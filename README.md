# Under Attack: Evaluating Deepfake Detection Models Against Real-World Cybersecurity Evasion Techniques

> MSc Cybersecurity Thesis — 2025

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU-orange.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This repository contains the experimental code for my MSc Cybersecurity thesis evaluating two deepfake detection models — **Xception** and **MesoNet** — against five passive evasion techniques grounded in four documented real-world attack domains: biometric authentication bypass, liveness detection evasion, KYC/AML identity fraud, and social engineering via executive impersonation.

A three-dimensional evaluation framework was proposed covering **detection accuracy**, **evasion robustness** (measured by Attack Success Rate), and **inference speed**. Both models were fine-tuned on the FaceForensics++ c40 dataset using a face detection and cropping pipeline.

A Chrome browser extension is included as a practical artefact, running the fine-tuned MesoNet model client-side via ONNX Runtime Web — fully offline, no server required.

---

## Key Results

| Condition | Xception ASR | MesoNet ASR |
|---|---|---|
| Baseline (clean) | 22.67% | 18.67% |
| Compression CRF 35 | 32.67% | **17.33%** |
| Compression CRF 45 | 14.0% | **9.33%** |
| Gaussian Noise σ=10 | **30.67%** | 40.0% |
| Gaussian Noise σ=25 | **28.0%** | 78.0% |
| Gamma shift γ=0.6 | **8.0%** | 10.0% |
| Gamma shift γ=1.6 | 30.0% | 20.67% |

**ASR = Attack Success Rate** — percentage of deepfakes misclassified as real after the attack is applied. Lower is better for the defender.

| Model | Accuracy | AUC | FPR | Inference Speed |
|---|---|---|---|---|
| Xception (finetuned) | 84.67% | 0.9465 | 8.0% | 3.65 fps |
| MesoNet (finetuned) | 79.33% | 0.8865 | 22.67% | 7.86 fps |

> Neither model meets the 24 fps real-time threshold — a critical finding for live video call social engineering detection.

---

## Repository Structure

```
├── finetune_models.py          # Fine-tune Xception & MesoNet on FF++ c40
├── xception_inference.py       # Xception inference + metrics
├── mesonet_inference.py        # MesoNet inference + metrics
├── evasion_attacks.py          # Apply 5 passive evasion conditions
├── run_all_experiments.py      # Master pipeline runner
├── face_crop.py                # Face detection + cropping (OpenCV Haar)
├── convert_weights.py          # Convert MesoNet Keras h5 → PyTorch pt
├── diag.py                     # Model output diagnostic tool
├── extension/                  # Chrome browser extension
│   ├── manifest.json
│   ├── popup.js                # MesoNet ONNX inference in browser
│   ├── popup.html
│   └── ...
└── README.md
```

---

## Requirements

```bash
pip install torch torchvision timm opencv-python scikit-learn tqdm h5py onnx onnxruntime
```

> **No GPU required** — all experiments run on CPU only. Tested on Python 3.14, Windows 11, Intel Core i7, 16GB RAM.

> **TensorFlow is not required** — MesoNet Keras weights are converted to PyTorch using `h5py` directly.

---

## Dataset

This project uses the **FaceForensics++ (c40)** dataset. Access requires submitting a request to the authors:

👉 [Request access here](https://github.com/ondyari/FaceForensics)

Once approved, download using:

```bash
python faceforensics_download_v4.py ./data/fake --server EU2 -d Deepfakes -c c40 -t videos -n 150
python faceforensics_download_v4.py ./data/real --server EU2 -d original -c c40 -t videos -n 150
```

Expected folder structure:

```
data/
├── real/   # 150 pristine videos
└── fake/   # 150 Deepfakes-type manipulated videos
```

---

## Usage

### 1. Convert MesoNet weights (one time)

Download `Meso4_DF.h5` from the [MesoNet GitHub repo](https://github.com/DariusAf/MesoNet), then:

```bash
pip install h5py
python convert_weights.py --input Meso4_DF.h5 --output Meso4_DF.pt
```

### 2. Fine-tune both models

```bash
python finetune_models.py --data_dir ./data --model both --mesonet_weights Meso4_DF.pt
```

Outputs: `mesonet_finetuned.pt` and `xception_finetuned.pt`

### 3. Run the full experiment pipeline

```bash
python run_all_experiments.py --data_dir ./data
```

This runs baseline inference, applies all 5 evasion attacks, re-runs inference on each attacked condition, and outputs `results/ALL_RESULTS.csv`.

### 4. Run individual inference

```bash
python xception_inference.py --data_dir ./data --weights xception_finetuned.pt
python mesonet_inference.py  --data_dir ./data --weights mesonet_finetuned.pt
```

### 5. Apply evasion attacks individually

```bash
python evasion_attacks.py --data_dir ./data --output_base ./data_attacked --attack compression_crf35
python evasion_attacks.py --data_dir ./data --output_base ./data_attacked --attack noise_sigma25
```

Available attacks: `compression_crf35`, `compression_crf45`, `noise_sigma10`, `noise_sigma25`, `gamma_06`, `gamma_16`

---

## Chrome Extension

A fully offline deepfake detection browser extension powered by MesoNet ONNX.

**Features:**
- Right-click any image on any webpage → "Check for deepfake"
- Face detection using YCbCr skin-tone analysis before inference
- MesoNet ONNX runs entirely in the browser — no server, no API
- Combined verdict: model score (60%) + visual checklist (40%)

**To load in Chrome:**
1. Export MesoNet to ONNX: `python -c "import torch; from mesonet_inference import Meso4, DEVICE; m=Meso4(); m.load_state_dict(torch.load('mesonet_finetuned.pt',map_location=DEVICE)); m.eval(); torch.onnx.export(m,torch.zeros(1,3,256,256),'extension/mesonet.onnx',opset_version=11)"`
2. Go to `chrome://extensions`
3. Enable **Developer mode**
4. Click **Load unpacked** → select the `extension/` folder

---

## Evasion Attack Details

| Attack | Tool | Parameters | Real-world Rationale |
|---|---|---|---|
| Compression CRF 35 | ffmpeg | `-crf 35 -vcodec libx264` | Social media re-encoding (YouTube, Twitter) |
| Compression CRF 45 | ffmpeg | `-crf 45 -vcodec libx264` | Heavy messaging compression (WhatsApp, Telegram) |
| Gaussian Noise σ=10 | OpenCV | `np.random.randn * 10` | Basic obfuscation — no specialist knowledge required |
| Gaussian Noise σ=25 | OpenCV | `np.random.randn * 25` | Strongest attack — MesoNet ASR 78.0% |
| Gamma shift | OpenCV | LUT γ=0.6 and γ=1.6 | Brightness obfuscation — invisible to human eye |

---

## Methodology Notes

- **Face cropping:** OpenCV Haar cascade classifier applied before all inference — models detect facial artifacts, not background patterns
- **MesoNet normalisation:** Input in `[0, 1]` range — divide by 255 only, no mean/std subtraction
- **Xception normalisation:** Standard `[-1, 1]` normalisation (mean=0.5, std=0.5)
- **Fine-tuning:** BCE loss, Adam optimiser, early stopping (patience=4), 80/20 train/val split
- **Random seed:** Fixed at 42 for reproducibility

---

## Threat Model

This thesis models an adversary who generates a deepfake and applies passive evasion techniques before distribution — requiring no specialist knowledge and no access to the target detection system. Four documented real-world attack domains are addressed:

1. **Biometric bypass** — virtual camera injection defeating Face ID, Windows Hello
2. **Liveness detection evasion** — real-time deepfakes replicating blink/head-turn commands
3. **KYC/AML fraud** — forged identities passing remote financial onboarding
4. **Social engineering** — CEO impersonation in live video calls ($25M lost, Hong Kong 2024)

---

## Citation

If you use this code or framework in your research, please cite:

```
Gnamiena, A. (2025). Under Attack: Evaluating Deepfake Detection Models Against 
Real-World Cybersecurity Evasion Techniques. MSc Cybersecurity Thesis.
```

---

## References

- Rössler et al. (2019) — FaceForensics++
- Afchar et al. (2018) — MesoNet
- Chollet (2017) — Xception
- Gragnaniello et al. (2021) — Are GAN Generated Images Easy to Detect?
- Yan et al. (2023) — DeepfakeBench

---

## License

MIT License — see [LICENSE](LICENSE) for details.

> **Note:** The FaceForensics++ dataset requires a separate access request from the dataset authors. Pretrained model weights are not included in this repository and must be obtained separately.
