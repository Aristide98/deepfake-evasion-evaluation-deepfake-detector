"""
convert_weights.py
------------------
Converts MesoNet Keras weights (Meso4_DF.h5) to PyTorch (.pt) format.

USAGE:
    python convert_weights.py --input Meso4_DF.h5 --output Meso4_DF.pt

INSTALL (one time):
    pip install h5py torch

NOTE: TensorFlow is NOT required. We read the .h5 file directly using
h5py, which works on any Python version including 3.14+.

Place this script in the same folder as Meso4_DF.h5 and run it.
The output Meso4_DF.pt file is what you pass to mesonet_inference.py.
"""

import argparse
import torch
import torch.nn as nn
import numpy as np

# ── Meso4 PyTorch architecture (must match mesonet_inference.py) ──
class Meso4(nn.Module):
    def __init__(self):
        super(Meso4, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm2d(8)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(8, 16, kernel_size=5, padding=2)
        self.bn3   = nn.BatchNorm2d(16)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.bn4   = nn.BatchNorm2d(16)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(4, 4)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.fc1     = nn.Linear(16 * 8 * 8, 16)
        self.leaky   = nn.LeakyReLU(0.1)
        self.fc2     = nn.Linear(16, 1)  # original MesoNet uses sigmoid (1 output)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.leaky(self.fc1(x))
        x = self.fc2(x)
        return x


def convert(h5_path, output_path):
    # Import here so the script fails clearly if h5py not installed
    try:
        import h5py
    except ImportError:
        print("[ERROR] h5py not installed. Run: pip install h5py")
        return

    print(f"[INFO] Reading Keras weights from: {h5_path}")

    model = Meso4()
    state = model.state_dict()

    with h5py.File(h5_path, "r") as f:

        # ── Helper to extract kernel + bias from a Keras layer ──
        def get_weights(layer_name):
            """
            Keras h5 weight layout varies by version.
            Try both common structures.
            """
            # Structure 1: f[layer_name][layer_name]['kernel:0'] etc.
            try:
                grp = f[layer_name][layer_name]
                kernel = np.array(grp["kernel:0"])
                bias   = np.array(grp["bias:0"])
                return kernel, bias
            except (KeyError, TypeError):
                pass
            # Structure 2: f['model_weights'][layer_name][layer_name]
            try:
                grp = f["model_weights"][layer_name][layer_name]
                kernel = np.array(grp["kernel:0"])
                bias   = np.array(grp["bias:0"])
                return kernel, bias
            except (KeyError, TypeError):
                pass
            # Structure 3: flat keys
            try:
                kernel = np.array(f[f"{layer_name}/kernel:0"])
                bias   = np.array(f[f"{layer_name}/bias:0"])
                return kernel, bias
            except KeyError:
                raise KeyError(f"Could not find weights for layer: {layer_name}. "
                               f"Available keys: {list(f.keys())}")

        def get_bn_weights(layer_name):
            """Extract BatchNorm gamma, beta, moving_mean, moving_variance."""
            try:
                grp = f[layer_name][layer_name]
                gamma    = np.array(grp["gamma:0"])
                beta     = np.array(grp["beta:0"])
                mean     = np.array(grp["moving_mean:0"])
                variance = np.array(grp["moving_variance:0"])
                return gamma, beta, mean, variance
            except (KeyError, TypeError):
                pass
            try:
                grp = f["model_weights"][layer_name][layer_name]
                gamma    = np.array(grp["gamma:0"])
                beta     = np.array(grp["beta:0"])
                mean     = np.array(grp["moving_mean:0"])
                variance = np.array(grp["moving_variance:0"])
                return gamma, beta, mean, variance
            except (KeyError, TypeError):
                raise KeyError(f"Could not find BN weights for: {layer_name}")

        # ── Conv layers ──────────────────────────────────────────
        # Keras conv kernel shape: (H, W, C_in, C_out)
        # PyTorch conv weight shape: (C_out, C_in, H, W)
        conv_map = [
            ("conv2d_5", "conv1"),
            ("conv2d_6", "conv2"),
            ("conv2d_7", "conv3"),
            ("conv2d_8", "conv4"),
        ]
        for keras_name, pt_name in conv_map:
            kernel, bias = get_weights(keras_name)
            # Transpose from (H, W, C_in, C_out) → (C_out, C_in, H, W)
            kernel_t = np.transpose(kernel, (3, 2, 0, 1))
            state[f"{pt_name}.weight"] = torch.FloatTensor(kernel_t)
            state[f"{pt_name}.bias"]   = torch.FloatTensor(bias)
            print(f"  [✓] {keras_name} → {pt_name}  shape: {kernel_t.shape}")

        # ── BatchNorm layers ─────────────────────────────────────
        bn_map = [
            ("batch_normalization_5", "bn1"),
            ("batch_normalization_6", "bn2"),
            ("batch_normalization_7", "bn3"),
            ("batch_normalization_8", "bn4"),
        ]
        for keras_name, pt_name in bn_map:
            gamma, beta, mean, var = get_bn_weights(keras_name)
            state[f"{pt_name}.weight"]       = torch.FloatTensor(gamma)
            state[f"{pt_name}.bias"]         = torch.FloatTensor(beta)
            state[f"{pt_name}.running_mean"] = torch.FloatTensor(mean)
            state[f"{pt_name}.running_var"]  = torch.FloatTensor(var)
            print(f"  [✓] {keras_name} → {pt_name}")

        # ── Fully connected layers ───────────────────────────────
        # Keras dense kernel shape: (C_in, C_out)
        # PyTorch linear weight shape: (C_out, C_in)
        fc_map = [
            ("dense_3", "fc1"),
            ("dense_4", "fc2"),
        ]
        for keras_name, pt_name in fc_map:
            kernel, bias = get_weights(keras_name)
            kernel_t = np.transpose(kernel, (1, 0))
            state[f"{pt_name}.weight"] = torch.FloatTensor(kernel_t)
            state[f"{pt_name}.bias"]   = torch.FloatTensor(bias)
            print(f"  [✓] {keras_name} → {pt_name}  shape: {kernel_t.shape}")

    # Load converted weights into model and verify
    model.load_state_dict(state)
    model.eval()

    # Quick sanity check — run a dummy forward pass
    dummy = torch.zeros(1, 3, 256, 256)
    with torch.no_grad():
        out = model(dummy)
    print(f"\n[INFO] Sanity check passed — output shape: {out.shape}  (expected: torch.Size([1, 1]))")

    # Save
    torch.save(model.state_dict(), output_path)
    print(f"[INFO] Weights saved to: {output_path}")
    print(f"\n[DONE] Conversion complete.")
    print(f"       Now run:")
    print(f"       python mesonet_inference.py --data_dir ./data --weights {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MesoNet Keras weights to PyTorch")
    parser.add_argument("--input",  type=str, default="Meso4_DF.h5",
                        help="Path to the Keras .h5 weights file")
    parser.add_argument("--output", type=str, default="Meso4_DF.pt",
                        help="Output path for the PyTorch .pt weights file")
    args = parser.parse_args()
    convert(args.input, args.output)
