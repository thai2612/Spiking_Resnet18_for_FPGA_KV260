# 1. Training: Spiking ResNet-18 Model Training

PyTorch + SpikingJelly training scripts for all SNN variants — from the original baseline to the final HLS-optimized architecture.

## Folder Structure

```
1_training/
├── shared/                          Shared utilities (used by baseline + A/B/C)
│   ├── evaluation_metrics.py        Comprehensive evaluation framework
│   ├── cross_validation.py          5-fold stratified cross-validation
│   └── multi_class_dataset.py       SDNET2018 dataset loader
├── baseline/                        Baseline SNN (full Spiking ResNet-18)
│   └── spiking_concrete.py          Training script
├── optimized_abc/                   Software-optimized variants (A, B, C)
│   ├── custom_spiking_resnet.py     Model definitions
│   └── spiking_optimized.py         Training script
├── hls_deh/                         HLS-optimized variants (D, E, F, G, H)
│   ├── custom_spiking_resnet_hls.py Model definitions
│   ├── train_hls_variants.py        Training script
│   └── evaluate_checkpoint.py       Checkpoint evaluation
└── README.md
```

## Setup

```bash
pip install -r ../requirements.txt
```

## Model Overview

| Category | Model | Params | Description |
|----------|-------|--------|-------------|
| **SNN Baseline** | Spiking ResNet-18 | ~11M | Full SpikingJelly ResNet-18, T=10 |
| **Optimized A/B/C** | Variant A | ~2.78M | Remove layer4 (3-stage) |
| | Variant B | ~2.80M | Thin 4-stage (half channels) |
| | Variant C | ~0.70M | Ultra-light (thin 3-stage) |
| **HLS D-H** | Variant D | 402K | **Selected for FPGA** (PCS β=0.75) |
| | Variant E | ~250K | Reduced final stage |
| | Variant F | ~200K | Aggressive 3-stage |
| | Variant G | ~186K | PCS β=0.50 ablation |
| | Variant H | ~702K | PCS β=1.00 ablation |

Variants D, G, H form the **PCS (Progressive Channel Scaling) ablation study** with β = 0.75, 0.50, 1.00 respectively.

---

## Training Baseline SNN (`baseline/`)

Full Spiking ResNet-18 from SpikingJelly, trained on SDNET2018.

```bash
cd baseline

# Standard training
python spiking_concrete.py \
    --data-dir /path/to/SDNET2018 \
    --batch-size 8 \
    --time-steps 10 \
    --num-epochs 20 \
    --learning-rate 0.001

# Cross-validation mode
python spiking_concrete.py \
    --data-dir /path/to/SDNET2018 \
    --mode cross_validation \
    --cv-folds 5

# Evaluate pre-trained model
python spiking_concrete.py \
    --eval-only \
    --model-path checkpoints/best_spiking_resnet.pth \
    --data-dir /path/to/SDNET2018
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | ./SDNET2018 | Path to SDNET2018 dataset |
| `--batch-size` | 8 | Batch size |
| `--time-steps` | 10 | SNN simulation timesteps (T) |
| `--num-epochs` | 20 | Training epochs |
| `--mode` | train | train, cross_validation, or comprehensive |
| `--cv-folds` | 5 | Cross-validation folds |
| `--eval-only` | False | Skip training, evaluate only |

---

## Training Software-Optimized Variants A/B/C (`optimized_abc/`)

```bash
cd optimized_abc

# Variant A: Remove layer4 (3-stage, ~2.78M params)
python spiking_optimized.py \
    --variant A \
    --data-dir /path/to/SDNET2018 \
    --batch-size 8 \
    --num-epochs 20

# Variant B: Thin 4-stage (half channels, ~2.80M params)
python spiking_optimized.py \
    --variant B \
    --data-dir /path/to/SDNET2018 \
    --batch-size 8 \
    --num-epochs 20

# Variant C: Ultra-light 3-stage (~0.70M params)
python spiking_optimized.py \
    --variant C \
    --data-dir /path/to/SDNET2018 \
    --batch-size 8 \
    --num-epochs 20
```

---

## Training HLS-Optimized Variants D-H (`hls_deh/`)

```bash
cd hls_deh

# Variant D (Selected for FPGA, β=0.75)
python train_hls_variants.py \
    --variant D \
    --data-dir /path/to/SDNET2018 \
    --num-epochs 20 \
    --batch-size 8

# Variant E
python train_hls_variants.py --variant E --data-dir /path/to/SDNET2018

# Variant F
python train_hls_variants.py --variant F --data-dir /path/to/SDNET2018

# Variant G (β=0.50 ablation)
python train_hls_variants.py --variant G --data-dir /path/to/SDNET2018

# Variant H (β=1.00 ablation)
python train_hls_variants.py --variant H --data-dir /path/to/SDNET2018

# Train all HLS variants at once
python train_hls_variants.py --variant all --data-dir /path/to/SDNET2018
```

### Evaluation

```bash
cd hls_deh

# Evaluate variant D
python evaluate_checkpoint.py \
    --variant D \
    --checkpoint /path/to/best_spiking_resnet_hlsD.pth \
    --data-dir /path/to/SDNET2018

# Evaluate variant G
python evaluate_checkpoint.py \
    --variant G \
    --checkpoint /path/to/best_spiking_resnet_hlsG.pth \
    --data-dir /path/to/SDNET2018
```

---

## Training Conditions

For fair comparison across all models:
- Batch size: 8
- Epochs: 20
- Learning rate: 0.001
- Time steps: 10
- Fixed train/val split (seed=42, 80/20)
- Mixed precision training (AMP)
- No early stopping (for fair comparison)

## Dataset: SDNET2018

```
SDNET2018/
├── D/                  # Deck
│   ├── CD/             # Cracked Deck (label=1)
│   └── UD/             # Uncracked Deck (label=0)
├── P/                  # Pavement
│   ├── CP/             # Cracked Pavement
│   └── UP/             # Uncracked Pavement
└── W/                  # Wall
    ├── CW/             # Cracked Wall
    └── UW/             # Uncracked Wall
```

The dataset is split 80/20 (train/val) with a fixed random seed (42). All images are resized to 256x256 and normalized.
