"""
finetune_models.py
------------------
Fine-tunes MesoNet and Xception on FaceForensics++ c40.
Uses face detection and cropping before training — critical for
deepfake detection models which look for facial artifacts.

Run OVERNIGHT on CPU:
  MesoNet : ~2–3 hours
  Xception: ~8–12 hours

USAGE:
    python finetune_models.py --data_dir ./data --model both --mesonet_weights Meso4_DF.pt

OUTPUT:
    mesonet_finetuned.pt
    xception_finetuned.pt
"""

import os
import time
import argparse
import random
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from face_crop import extract_face

# ── Config ────────────────────────────────────────────────────
EPOCHS      = 15
BATCH_SIZE  = 8
LR          = 1e-4
VAL_SPLIT   = 0.2
FRAMES      = 8       # frames per video — more frames with face crop
SEED        = 42
DEVICE      = torch.device("cpu")

torch.set_num_threads(8)
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


# ════════════════════════════════════════════════════════════════
# DATASET — with face cropping
# ════════════════════════════════════════════════════════════════

class DeepfakeDataset(Dataset):
    def __init__(self, video_list, transform, frames_per_video=FRAMES):
        self.items     = []
        self.transform = transform
        video_exts     = {".mp4", ".avi", ".mov", ".mkv"}
        face_found     = 0
        face_fallback  = 0

        print(f"[INFO] Extracting face crops from {len(video_list)} videos...")
        for video_path, label in tqdm(video_list, desc="Extracting faces"):
            if os.path.splitext(video_path)[1].lower() not in video_exts:
                continue
            cap   = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total == 0:
                cap.release()
                continue

            indices = np.linspace(0, total - 1, num=frames_per_video, dtype=int)
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face      = extract_face(frame_rgb)
                    if face.size > 0:
                        self.items.append((face, label))
                        face_found += 1
                    else:
                        face_fallback += 1
            cap.release()

        print(f"[INFO] Dataset: {len(self.items)} face crops "
              f"({sum(1 for _,l in self.items if l==0)} real, "
              f"{sum(1 for _,l in self.items if l==1)} fake)")
        print(f"[INFO] Face detection: {face_found} found, {face_fallback} fallback centre crops")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        face, label = self.items[idx]
        return self.transform(face), torch.tensor(label, dtype=torch.float32)


def get_video_list(data_dir):
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    items      = []
    for label, folder in [(0, "real"), (1, "fake")]:
        fp = os.path.join(data_dir, folder)
        if not os.path.isdir(fp):
            continue
        for fname in os.listdir(fp):
            if os.path.splitext(fname)[1].lower() in video_exts:
                items.append((os.path.join(fp, fname), label))
    random.shuffle(items)
    return items


def train_val_split(items, val_ratio=VAL_SPLIT):
    n = int(len(items) * val_ratio)
    return items[n:], items[:n]


# ════════════════════════════════════════════════════════════════
# MESONET
# ════════════════════════════════════════════════════════════════

class Meso4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(3, 8, 3, padding=1)
        self.bn1     = nn.BatchNorm2d(8)
        self.pool1   = nn.MaxPool2d(2, 2)
        self.conv2   = nn.Conv2d(8, 8, 5, padding=2)
        self.bn2     = nn.BatchNorm2d(8)
        self.pool2   = nn.MaxPool2d(2, 2)
        self.conv3   = nn.Conv2d(8, 16, 5, padding=2)
        self.bn3     = nn.BatchNorm2d(16)
        self.pool3   = nn.MaxPool2d(2, 2)
        self.conv4   = nn.Conv2d(16, 16, 5, padding=2)
        self.bn4     = nn.BatchNorm2d(16)
        self.pool4   = nn.MaxPool2d(4, 4)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.fc1     = nn.Linear(16 * 8 * 8, 16)
        self.leaky   = nn.LeakyReLU(0.1)
        self.fc2     = nn.Linear(16, 1)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool4(torch.relu(self.bn4(self.conv4(x))))
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.leaky(self.fc1(x))
        return self.fc2(x)


mesonet_train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    transforms.ToTensor(),
    # MesoNet uses [0,1] — no normalisation
])

mesonet_val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


# ════════════════════════════════════════════════════════════════
# XCEPTION
# ════════════════════════════════════════════════════════════════

def get_xception():
    print("[INFO] Loading Xception (ImageNet pretrained via timm)...")
    model = timm.create_model("xception", pretrained=True)
    # Freeze early layers, unfreeze last block + classifier
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if any(x in name for x in ["block12", "conv4", "bn4", "fc"]):
            param.requires_grad = True
    in_features = model.fc.in_features
    model.fc    = nn.Linear(in_features, 1)
    trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Trainable parameters: {trainable:,}")
    return model


xception_train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((299, 299)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

xception_val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# ════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ════════════════════════════════════════════════════════════════

def train_model(model, train_loader, val_loader, model_name,
                output_path, epochs=EPOCHS, lr=LR, patience=4):

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )

    best_val_loss = float("inf")
    patience_cnt  = 0

    print(f"\n{'='*55}")
    print(f"  TRAINING {model_name.upper()} (with face crops)")
    print(f"  Epochs: {epochs} | LR: {lr} | Batch: {BATCH_SIZE}")
    print(f"  Train batches: {len(train_loader)} | Val: {len(val_loader)}")
    print(f"{'='*55}")

    for epoch in range(epochs):
        # Train
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        t0 = time.time()

        for frames, labels in tqdm(train_loader,
                                   desc=f"Ep {epoch+1}/{epochs} train",
                                   leave=False):
            frames, labels = frames.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out  = model(frames).squeeze(1)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            tr_loss    += loss.item() * frames.size(0)
            preds       = (torch.sigmoid(out) >= 0.5).float()
            tr_correct += (preds == labels).sum().item()
            tr_total   += labels.size(0)

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for frames, labels in tqdm(val_loader,
                                       desc=f"Ep {epoch+1}/{epochs} val",
                                       leave=False):
                frames, labels = frames.to(DEVICE), labels.to(DEVICE)
                out  = model(frames).squeeze(1)
                loss = criterion(out, labels)
                val_loss    += loss.item() * frames.size(0)
                preds        = (torch.sigmoid(out) >= 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        atl = tr_loss  / tr_total
        avl = val_loss / val_total
        ta  = tr_correct  / tr_total  * 100
        va  = val_correct / val_total * 100

        scheduler.step(avl)

        print(f"  Ep {epoch+1:2d}/{epochs} | "
              f"Train loss:{atl:.4f} acc:{ta:.1f}% | "
              f"Val loss:{avl:.4f} acc:{va:.1f}% | "
              f"{time.time()-t0:.0f}s")

        if avl < best_val_loss:
            best_val_loss = avl
            torch.save(model.state_dict(), output_path)
            print(f"  [✓] Best model saved → {output_path} (val acc: {va:.1f}%)")
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  [STOP] Early stopping at epoch {epoch+1}")
                break

    print(f"\n  Best val loss: {best_val_loss:.4f}")
    print(f"  Saved: {output_path}")


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def run(data_dir, model_choice, mesonet_weights, epochs):
    videos               = get_video_list(data_dir)
    train_vids, val_vids = train_val_split(videos)
    print(f"[INFO] Train: {len(train_vids)} | Val: {len(val_vids)} videos")

    if model_choice in ("mesonet", "both"):
        print("\n" + "="*55)
        print("  FINE-TUNING MESONET (with face crops)")
        print("="*55)
        model = Meso4()
        if mesonet_weights and os.path.exists(mesonet_weights):
            print(f"[INFO] Loading base weights: {mesonet_weights}")
            model.load_state_dict(torch.load(mesonet_weights, map_location=DEVICE))
        model.to(DEVICE)

        train_ds = DeepfakeDataset(train_vids, mesonet_train_transform)
        val_ds   = DeepfakeDataset(val_vids,   mesonet_val_transform)
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

        train_model(model, train_dl, val_dl,
                    "MesoNet", "mesonet_finetuned.pt", epochs=epochs)

    if model_choice in ("xception", "both"):
        print("\n" + "="*55)
        print("  FINE-TUNING XCEPTION (with face crops)")
        print("="*55)
        model = get_xception()
        model.to(DEVICE)

        train_ds = DeepfakeDataset(train_vids, xception_train_transform)
        val_ds   = DeepfakeDataset(val_vids,   xception_val_transform)
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

        train_model(model, train_dl, val_dl,
                    "Xception", "xception_finetuned.pt", epochs=epochs)

    print("\n" + "="*55)
    print("  ALL FINE-TUNING COMPLETE")
    print("  Run inference:")
    if model_choice in ("mesonet", "both"):
        print("  python mesonet_inference.py --data_dir ./data --weights mesonet_finetuned.pt")
    if model_choice in ("xception", "both"):
        print("  python xception_inference.py --data_dir ./data --weights xception_finetuned.pt")
    print("="*55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",        type=str, default="./data")
    parser.add_argument("--model",           type=str, default="both",
                        choices=["mesonet", "xception", "both"])
    parser.add_argument("--mesonet_weights", type=str, default="Meso4_DF.pt")
    parser.add_argument("--epochs",          type=int, default=EPOCHS)
    args = parser.parse_args()
    run(args.data_dir, args.model, args.mesonet_weights, args.epochs)
