"""
diag.py
-------
Checks what the finetuned models actually output on known real/fake videos.
Run this to diagnose why inference results are near-random.
"""

import torch
import cv2
import numpy as np
import os

# ── Test MesoNet ──────────────────────────────────────────────
print("=" * 50)
print("  MESONET DIAGNOSTIC")
print("=" * 50)

try:
    from mesonet_inference import Meso4, transform as meso_transform, DEVICE

    model = Meso4()
    model.load_state_dict(torch.load("mesonet_finetuned.pt", map_location=DEVICE))
    model.eval()

    test_files = [
        ("data/fake/008_990.mp4", "FAKE"),
        ("data/fake/033_097.mp4", "FAKE"),
        ("data/real/008.mp4",     "REAL"),
        ("data/real/033.mp4",     "REAL"),
    ]

    for fpath, expected in test_files:
        if not os.path.exists(fpath):
            print(f"  [SKIP] {fpath} not found")
            continue
        cap = cv2.VideoCapture(fpath)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"  [SKIP] Could not read {fpath}")
            continue
        frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = meso_transform(frame).unsqueeze(0)
        with torch.no_grad():
            out  = model(tensor)
            raw  = out[0][0].item()
            prob = torch.sigmoid(out)[0][0].item()
            pred = "FAKE" if prob >= 0.5 else "REAL"
            ok   = "OK" if pred == expected else "WRONG"
        print(f"  [{ok}] Expected={expected} | raw={raw:.4f} | sigmoid={prob:.4f} | predicted={pred}")

except Exception as e:
    print(f"  [ERROR] {e}")

# ── Test Xception ─────────────────────────────────────────────
print()
print("=" * 50)
print("  XCEPTION DIAGNOSTIC")
print("=" * 50)

try:
    from xception_inference import load_model as load_xception
    from xception_inference import transform as xcept_transform

    model_x = load_xception("xception_finetuned.pt")

    for fpath, expected in test_files:
        if not os.path.exists(fpath):
            print(f"  [SKIP] {fpath} not found")
            continue
        cap = cv2.VideoCapture(fpath)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"  [SKIP] Could not read {fpath}")
            continue
        frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = xcept_transform(frame).unsqueeze(0)
        with torch.no_grad():
            out  = model_x(tensor)
            raw  = out[0][0].item()
            prob = torch.sigmoid(out)[0][0].item()
            pred = "FAKE" if prob >= 0.5 else "REAL"
            ok   = "OK" if pred == expected else "WRONG"
        print(f"  [{ok}] Expected={expected} | raw={raw:.4f} | sigmoid={prob:.4f} | predicted={pred}")

except Exception as e:
    print(f"  [ERROR] {e}")

print()
print("=" * 50)
print("  INTERPRETATION")
print("=" * 50)
print("  Good: fake videos score > 0.5, real videos score < 0.5")
print("  Bad : all videos score similarly (near 0 or near 1)")
print("        = model did not learn to distinguish real from fake")