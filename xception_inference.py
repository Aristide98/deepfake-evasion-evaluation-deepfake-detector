"""
xception_inference.py
---------------------
Xception deepfake detection with face cropping.

USAGE:
    python xception_inference.py --data_dir ./data --weights xception_finetuned.pt
"""

import os, time, argparse, csv, cv2, torch, torch.nn as nn
import timm, numpy as np
from torchvision import transforms
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from tqdm import tqdm
from face_crop import extract_frames_with_faces

FRAMES_PER_VIDEO = 10
IMAGE_SIZE       = 299
DEVICE           = torch.device("cpu")
torch.set_num_threads(8)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def load_model(weights_path=None):
    print("[INFO] Loading Xception via timm...")
    model = timm.create_model("xception", pretrained=(weights_path is None))
    in_features = model.fc.in_features
    model.fc    = nn.Linear(in_features, 1)
    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        print(f"[INFO] Finetuned weights loaded: {weights_path}")
    elif weights_path:
        print(f"[WARN] Weights not found: {weights_path} — using ImageNet.")
    model.eval()
    model.to(DEVICE)
    return model


def predict_video(model, video_path):
    faces = extract_frames_with_faces(video_path, FRAMES_PER_VIDEO)
    if not faces:
        return 0, 0.5
    probs = []
    with torch.no_grad():
        for face in faces:
            if face.size == 0:
                continue
            tensor = transform(face).unsqueeze(0).to(DEVICE)
            out    = model(tensor)
            prob   = torch.sigmoid(out)[0][0].item()
            probs.append(prob)
    if not probs:
        return 0, 0.5
    avg_prob = np.mean(probs)
    return (1 if avg_prob >= 0.5 else 0), avg_prob


def load_videos(data_dir):
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    items = []
    for label, folder in [(0, "real"), (1, "fake")]:
        fp = os.path.join(data_dir, folder)
        if not os.path.isdir(fp):
            continue
        for fname in os.listdir(fp):
            if os.path.splitext(fname)[1].lower() in video_exts:
                items.append((os.path.join(fp, fname), label))
    print(f"[INFO] {len(items)} videos "
          f"({sum(1 for _,l in items if l==0)} real, "
          f"{sum(1 for _,l in items if l==1)} fake)")
    return items


def compute_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return {"accuracy": round(acc*100,2), "auc": round(auc,4),
            "f1": round(f1,4), "fpr": round(fpr*100,2),
            "fnr": round(fnr*100,2), "asr": round(fnr*100,2)}


def run_inference(data_dir, output_csv, weights_path=None):
    model  = load_model(weights_path)
    videos = load_videos(data_dir)
    if not videos:
        print("[ERROR] No videos found.")
        return

    y_true, y_pred, y_prob = [], [], []
    total_frames = 0
    t0 = time.time()

    for video_path, label in tqdm(videos, desc="Xception inference"):
        pred, prob = predict_video(model, video_path)
        y_true.append(label); y_pred.append(pred); y_prob.append(prob)
        total_frames += FRAMES_PER_VIDEO

    elapsed   = time.time() - t0
    fps       = round(total_frames / elapsed, 2)
    per_video = round(elapsed / len(videos), 3)
    metrics   = compute_metrics(y_true, y_pred, y_prob)

    print("\n" + "="*50)
    print("  XCEPTION RESULTS")
    print("="*50)
    print(f"  Videos evaluated : {len(videos)}")
    print(f"  Weights          : {'Finetuned' if weights_path else 'ImageNet only'}")
    print(f"  Accuracy         : {metrics['accuracy']}%")
    print(f"  AUC              : {metrics['auc']}")
    print(f"  F1-score         : {metrics['f1']}")
    print(f"  False Positive R : {metrics['fpr']}%")
    print(f"  False Negative R : {metrics['fnr']}%")
    print(f"  Attack Success R : {metrics['asr']}%")
    print(f"  Inference Speed  : {fps} fps")
    print(f"  Time per video   : {per_video}s")
    print("="*50)

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model","condition","videos","accuracy","auc",
            "f1","fpr","fnr","asr","fps","time_per_video_s"])
        writer.writeheader()
        writer.writerow({
            "model": f"Xception ({'finetuned' if weights_path else 'ImageNet'})",
            "condition": os.path.basename(data_dir),
            "videos": len(videos), **metrics,
            "fps": fps, "time_per_video_s": per_video})
    print(f"[INFO] Saved: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output",   type=str, default="results_xception.csv")
    parser.add_argument("--weights",  type=str, default=None)
    args = parser.parse_args()
    run_inference(args.data_dir, args.output, args.weights)
