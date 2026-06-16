"""
evasion_attacks.py
------------------
Applies three passive evasion attack types to a folder of videos.
Run this BEFORE inference to generate the attacked video sets.

ATTACKS INCLUDED:
  1. Compression  — ffmpeg CRF 35 and CRF 45
  2. Gaussian noise — sigma 10 and sigma 25
  3. Gamma shift  — gamma 0.6 (darken) and gamma 1.6 (brighten)

USAGE:
    # Apply all attacks to your data folder:
    python evasion_attacks.py --data_dir ./data --output_base ./data_attacked

    # Apply one specific attack only:
    python evasion_attacks.py --data_dir ./data --output_base ./data_attacked --attack compression_35

OUTPUT FOLDER STRUCTURE (one per condition):
    data_attacked/
      compression_crf35/real/  compression_crf35/fake/
      compression_crf45/real/  compression_crf45/fake/
      noise_sigma10/real/      noise_sigma10/fake/
      noise_sigma25/real/      noise_sigma25/fake/
      gamma_06/real/           gamma_06/fake/
      gamma_16/real/           gamma_16/fake/

Run your inference scripts on each output folder to get per-condition results.

INSTALL:
    pip install opencv-python tqdm
    ffmpeg must be installed: https://ffmpeg.org/download.html
"""

import os
import shutil
import argparse
import subprocess
import cv2
import numpy as np
from tqdm import tqdm

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


# ════════════════════════════════════════════════════════════════
# ATTACK 1 — Compression (ffmpeg)
# ════════════════════════════════════════════════════════════════

def apply_compression(input_path, output_path, crf):
    """
    Re-encode video using H.264 with given CRF value.
    CRF 35 = moderate social-media compression (YouTube, Twitter).
    CRF 45 = heavy compression (WhatsApp, Telegram, TikTok).
    Higher CRF = more compression = more quality loss.
    Cybersecurity rationale: attackers re-encode deepfakes before
    distribution to degrade detector input features.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vcodec", "libx264",
        "-crf", str(crf),
        "-preset", "fast",
        "-an",                  # strip audio — not needed for visual detection
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[WARN] ffmpeg failed on {input_path}: {result.stderr[:200]}")


# ════════════════════════════════════════════════════════════════
# ATTACK 2 — Gaussian Noise (OpenCV)
# ════════════════════════════════════════════════════════════════

def apply_gaussian_noise(input_path, output_path, sigma):
    """
    Add Gaussian noise to every frame of the video.
    sigma=10: low perturbation, basic obfuscation.
    sigma=25: higher perturbation, stronger evasion.
    Cybersecurity rationale: low-skill attackers add noise to
    disrupt high-frequency texture features that detectors rely on.
    No specialist knowledge required — one line of OpenCV code.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open {input_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Add Gaussian noise
        noise        = np.random.randn(*frame.shape) * sigma
        noisy_frame  = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        out.write(noisy_frame)

    cap.release()
    out.release()


# ════════════════════════════════════════════════════════════════
# ATTACK 3 — Gamma Shift (OpenCV LUT)
# ════════════════════════════════════════════════════════════════

def build_gamma_lut(gamma):
    """Build a lookup table for gamma correction."""
    inv_gamma = 1.0 / gamma
    table     = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in range(256)
    ]).astype("uint8")
    return table


def apply_gamma_shift(input_path, output_path, gamma):
    """
    Apply gamma correction to every frame of the video.
    gamma=0.6: darken the image.
    gamma=1.6: brighten the image.
    Cybersecurity rationale: simple brightness obfuscation
    with minimal visible quality degradation to the human eye,
    but measurable impact on pixel-level detector features.
    Requires no specialist tools — implementable in one line.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open {input_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    lut    = build_gamma_lut(gamma)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        corrected = cv2.LUT(frame, lut)
        out.write(corrected)

    cap.release()
    out.release()


# ════════════════════════════════════════════════════════════════
# RUNNER
# ════════════════════════════════════════════════════════════════

ATTACK_CONFIG = {
    "compression_crf35": {
        "fn":     apply_compression,
        "kwargs": {"crf": 35},
        "label":  "Attack 1a: Compression CRF 35 (moderate social-media)",
        "ext":    ".mp4",
    },
    "compression_crf45": {
        "fn":     apply_compression,
        "kwargs": {"crf": 45},
        "label":  "Attack 1b: Compression CRF 45 (heavy WhatsApp/TikTok)",
        "ext":    ".mp4",
    },
    "noise_sigma10": {
        "fn":     apply_gaussian_noise,
        "kwargs": {"sigma": 10},
        "label":  "Attack 2a: Gaussian Noise sigma=10 (low perturbation)",
        "ext":    ".mp4",
    },
    "noise_sigma25": {
        "fn":     apply_gaussian_noise,
        "kwargs": {"sigma": 25},
        "label":  "Attack 2b: Gaussian Noise sigma=25 (higher perturbation)",
        "ext":    ".mp4",
    },
    "gamma_06": {
        "fn":     apply_gamma_shift,
        "kwargs": {"gamma": 0.6},
        "label":  "Attack 3a: Gamma shift 0.6 (darken)",
        "ext":    ".mp4",
    },
    "gamma_16": {
        "fn":     apply_gamma_shift,
        "kwargs": {"gamma": 1.6},
        "label":  "Attack 3b: Gamma shift 1.6 (brighten)",
        "ext":    ".mp4",
    },
}


def get_videos(data_dir):
    """Collect all videos from data_dir/real and data_dir/fake."""
    items = []
    for label_folder in ["real", "fake"]:
        folder = os.path.join(data_dir, label_folder)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if os.path.splitext(fname)[1].lower() in VIDEO_EXTS:
                items.append((os.path.join(folder, fname), label_folder, fname))
    return items


def run_attack(attack_name, data_dir, output_base):
    cfg    = ATTACK_CONFIG[attack_name]
    videos = get_videos(data_dir)

    if not videos:
        print(f"[ERROR] No videos found in {data_dir}")
        return

    print(f"\n[ATTACK] {cfg['label']}")
    print(f"         Input : {data_dir}")
    print(f"         Output: {os.path.join(output_base, attack_name)}")
    print(f"         Videos: {len(videos)}")

    for video_path, label_folder, fname in tqdm(videos, desc=attack_name):
        out_dir  = os.path.join(output_base, attack_name, label_folder)
        out_name = os.path.splitext(fname)[0] + cfg["ext"]
        out_path = os.path.join(out_dir, out_name)

        cfg["fn"](video_path, out_path, **cfg["kwargs"])

    print(f"[DONE] {attack_name} complete.")


def run_all_attacks(data_dir, output_base, attack_filter=None):
    attacks = [attack_filter] if attack_filter else list(ATTACK_CONFIG.keys())
    print(f"\n{'='*55}")
    print(f"  EVASION ATTACK PIPELINE")
    print(f"  Attacks to run : {len(attacks)}")
    print(f"  Source         : {data_dir}")
    print(f"  Output base    : {output_base}")
    print(f"{'='*55}")

    for attack_name in attacks:
        if attack_name not in ATTACK_CONFIG:
            print(f"[WARN] Unknown attack: {attack_name}. Skipping.")
            continue
        run_attack(attack_name, data_dir, output_base)

    print(f"\n{'='*55}")
    print("  ALL ATTACKS COMPLETE")
    print(f"  Output folders created in: {output_base}")
    print(f"  Next step: run inference scripts on each subfolder")
    print(f"  Example:")
    for name in attacks:
        print(f"    python xception_inference.py "
              f"--data_dir {os.path.join(output_base, name)} "
              f"--output results_xception_{name}.csv")
    print(f"{'='*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake Evasion Attack Scripts")
    parser.add_argument("--data_dir",     type=str, required=True,
                        help="Source folder with real/ and fake/ subfolders")
    parser.add_argument("--output_base",  type=str, default="./data_attacked",
                        help="Base folder for attacked video output")
    parser.add_argument("--attack",       type=str, default=None,
                        help=f"Run one specific attack only. Options: {list(ATTACK_CONFIG.keys())}")
    args = parser.parse_args()
    run_all_attacks(args.data_dir, args.output_base, args.attack)
