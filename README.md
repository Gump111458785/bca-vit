# BCA-ViT: Bidirectional Cross-Attention Adaptation and Calibration for Transfer Vision Transformers

Official implementation of the paper **"Bidirectional Cross-Attention Adaptation and Calibration for Transfer Vision Transformers"**.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.12+](https://img.shields.io/badge/PyTorch-1.12+-orange.svg)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/CUDA-GPU_Required-brightgreen.svg)]()
[![Pretrained Weights](https://img.shields.io/badge/Foundation_Weights-Required-red.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Overview

**BCA-ViT** tackles the dual challenge of domain-shift miscalibration and high fine-tuning cost when adapting large foundation Vision Transformers to downstream visual tasks:
1. **Bidirectional Cross-Attention (BCA) Adapter**: Bridges frozen ViT-B/16 transformer layers with dynamic bottleneck prompts, adapting to target domains with < 2.2% trainable parameters.
2. **Uncertainty-Calibrated Margin (UCM) Loss**: Penalizes out-of-domain transfer overconfidence and restores posterior probability calibration.
3. **Multi-Dataset Transfer Benchmark**: Evaluated on Flowers-102, Oxford-IIIT Pets, and DTD across Accuracy, Macro-F1, ECE, and GPU Throughput.

---

## ⚙️ System Requirements

- **OS**: Linux / macOS
- **Hardware**: NVIDIA GPU (CUDA acceleration strictly required)
- **Pretrained Weights**: ImageNet-1k pretrained ViT-B/16 checkpoint (`vit_base_patch16_224.pth`) required.

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/Gump111458785/bca-vit.git
cd bca-vit
pip install -r requirements.txt
pip install -e .
```

### 2. Download Pretrained Weights (Required)

Before training or evaluation, download the foundation model weights:

```bash
bash scripts/download_weights.sh
```

### 3. Download Datasets

```bash
bash scripts/download_data.sh
```

---

## 🔬 Training & Evaluation

### Parameter-Efficient Training on GPU

Train BCA-ViT on Flowers-102:
```bash
python scripts/train.py \
    --model bca_vit \
    --dataset flowers102 \
    --weights_path ./weights/vit_base_patch16_224.pth \
    --epochs 20 \
    --batch_size 32
```

### Evaluating Checkpoints

```bash
python scripts/evaluate.py \
    --model bca_vit \
    --dataset flowers102 \
    --checkpoint ./checkpoints/bca_vit_flowers102_seed42_best.pth
```

### Profiling Parameter Budget & Hardware Efficiency

```bash
python scripts/benchmark_efficiency.py
```

---

## 📊 Experimental Results

### Table 1: Multi-Dataset Transfer Performance and Calibration Comparison

| Dataset | Method | Trainable Params (M) | Top-1 Acc (%) ↑ | Top-5 Acc (%) ↑ | Macro-F1 (%) ↑ | ECE (%) ↓ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Flowers-102** | Linear Probe | 0.08 | 91.80 | 97.45 | 91.68 | 7.85 |
| | Full FT | 85.88 | 96.25 | 99.30 | 96.18 | 6.42 |
| | LoRA | 1.18 | 96.80 | 99.45 | 96.72 | 5.10 |
| | VPT | 0.72 | 96.10 | 99.20 | 96.02 | 5.88 |
| | **BCA-ViT (Ours)**| **1.86** | **98.15** | **99.80** | **98.10** | **2.25** |
| **Oxford-Pets** | Linear Probe | 0.03 | 90.20 | 97.80 | 90.05 | 8.12 |
| | Full FT | 85.84 | 93.45 | 98.60 | 93.38 | 6.95 |
| | LoRA | 1.15 | 94.10 | 98.90 | 94.02 | 5.48 |
| | VPT | 0.69 | 93.65 | 98.75 | 93.58 | 6.15 |
| | **BCA-ViT (Ours)**| **1.82** | **95.60** | **99.35** | **95.54** | **2.68** |
| **DTD** | Linear Probe | 0.04 | 71.30 | 91.20 | 71.15 | 10.45 |
| | Full FT | 85.85 | 74.85 | 92.65 | 74.72 | 9.80 |
| | LoRA | 1.16 | 76.20 | 93.40 | 76.08 | 7.92 |
| | VPT | 0.70 | 75.40 | 93.10 | 75.25 | 8.45 |
| | **BCA-ViT (Ours)**| **1.83** | **78.90** | **95.10** | **78.78** | **3.85** |

### Table 2: Parameter Budget and GPU Inference Efficiency

| Architecture | Total Params (M) | Trainable Params (M) | FLOPs (G) | GPU Latency (ms) | Throughput (img/s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Full FT | 85.88 | 85.88 | 17.58 | 7.85 | 680 |
| Linear Probe | 85.88 | 0.08 | 17.58 | 7.82 | 685 |
| LoRA | 87.06 | 1.18 | 17.65 | 8.12 | 655 |
| VPT | 86.60 | 0.72 | 17.62 | 8.05 | 660 |
| **BCA-ViT (Ours)** | **87.74** | **1.86** | **17.82** | **8.40** | **635** |

### Table 3: Ablation Study on Flowers-102

| Variant | BCA Adapter | UCM Loss | Temp Calibration | Top-1 Acc (%) | Macro-F1 (%) | ECE (%) ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| (a) Baseline (Linear Probe) | ✗ | ✗ | ✗ | 91.80 | 91.68 | 7.85 |
| (b) + BCA Adapter | ✓ | ✗ | ✗ | 97.45 | 97.38 | 4.60 |
| (c) + UCM Loss | ✗ | ✓ | ✗ | 93.20 | 93.12 | 4.95 |
| (d) BCA + UCM | ✓ | ✓ | ✗ | 97.90 | 97.82 | 2.50 |
| (e) Full BCA-ViT | ✓ | ✓ | ✓ | **98.15** | **98.10** | **2.25** |

---

## 🔁 Automated Reproduction Verification

```bash
python scripts/verify_reproduction.py
```
