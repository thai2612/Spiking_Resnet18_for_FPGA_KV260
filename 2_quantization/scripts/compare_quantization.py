#!/usr/bin/env python3
"""
Phase A3: Quantization Comparison

Compare INT8 vs INT16 quantization to choose the best option.
Evaluates accuracy, memory, and provides recommendation.

Usage:
    python compare_quantization.py --checkpoint <path>
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from datetime import datetime
from sklearn.metrics import accuracy_score, confusion_matrix

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
TRAINING_DIR = os.path.join(REPO_ROOT, '1_training')
sys.path.insert(0, TRAINING_DIR)

from custom_spiking_resnet_hls import create_hls_variant, HLS_VARIANT_CONFIGS
from spikingjelly.activation_based import functional

# Settings
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SDNET2018Dataset(Dataset):
    """SDNET2018 Dataset loader"""

    def __init__(self, root_dir, split='val', transform=None, train_ratio=0.8):
        self.transform = transform
        self.images = []
        self.labels = []

        subdirs = {'D': ['CD', 'UD'], 'P': ['CP', 'UP'], 'W': ['CW', 'UW']}

        for main_dir, sub_list in subdirs.items():
            for sub_dir in sub_list:
                label = 1 if sub_dir.startswith('C') else 0
                dir_path = os.path.join(root_dir, main_dir, sub_dir)
                if os.path.exists(dir_path):
                    for img_name in os.listdir(dir_path):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            self.images.append(os.path.join(dir_path, img_name))
                            self.labels.append(label)

        total = len(self.images)
        indices = np.arange(total)
        rng = np.random.RandomState(42)
        rng.shuffle(indices)
        train_size = int(total * train_ratio)

        if split == 'train':
            indices = indices[:train_size]
        else:
            indices = indices[train_size:]

        self.images = [self.images[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


def compute_scale_zp(min_val, max_val, num_bits, symmetric=True):
    """Compute quantization scale and zero point"""
    if symmetric:
        abs_max = max(abs(min_val), abs(max_val))
        qmax = 2 ** (num_bits - 1) - 1
        scale = abs_max / qmax if abs_max > 0 else 1.0
        zero_point = 0
    else:
        qmin = 0
        qmax = 2 ** num_bits - 1
        scale = (max_val - min_val) / (qmax - qmin) if max_val > min_val else 1.0
        zero_point = int(round(qmin - min_val / scale))
    return scale, zero_point


def quantize_tensor(tensor, scale, zero_point, num_bits, symmetric=True):
    """Quantize tensor"""
    if symmetric:
        qmax = 2 ** (num_bits - 1) - 1
        qmin = -qmax - 1
    else:
        qmin = 0
        qmax = 2 ** num_bits - 1

    quantized = torch.round(tensor / scale) + zero_point
    quantized = torch.clamp(quantized, qmin, qmax)
    return quantized


def dequantize_tensor(quantized, scale, zero_point):
    """Dequantize tensor back to float"""
    return (quantized.float() - zero_point) * scale


def simulate_quantized_inference(model, val_loader, device, num_bits, num_batches=None):
    """
    Simulate quantized inference by:
    1. Quantize weights
    2. Run inference with fake quantization
    3. Return accuracy metrics
    """
    model.eval()

    # Step 1: Quantize all weights
    original_weights = {}
    quant_info = {}

    for name, param in model.named_parameters():
        if 'weight' in name:
            original_weights[name] = param.data.clone()

            w = param.data
            w_min, w_max = w.min().item(), w.max().item()
            scale, zp = compute_scale_zp(w_min, w_max, num_bits, symmetric=True)

            # Quantize and dequantize (simulates quantization error)
            w_quant = quantize_tensor(w, scale, zp, num_bits, symmetric=True)
            w_dequant = dequantize_tensor(w_quant, scale, zp)

            # Replace weights with dequantized version
            param.data = w_dequant

            quant_info[name] = {
                'scale': scale,
                'min': w_min,
                'max': w_max,
                'quant_error': (w - w_dequant).abs().mean().item()
            }

    # Step 2: Run inference
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            if num_batches and batch_idx >= num_batches:
                break

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

            functional.reset_net(model.model)

            if (batch_idx + 1) % 50 == 0:
                print(f"    Batch {batch_idx + 1}...", flush=True)

    # Step 3: Restore original weights
    for name, param in model.named_parameters():
        if name in original_weights:
            param.data = original_weights[name]

    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds) * 100
    cm = confusion_matrix(all_labels, all_preds)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
    else:
        sensitivity = 0
        specificity = 0

    # Calculate average quantization error
    avg_quant_error = np.mean([info['quant_error'] for info in quant_info.values()])

    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'num_samples': len(all_labels),
        'avg_quant_error': avg_quant_error,
        'quant_info': quant_info
    }


def calculate_memory_usage(model, num_bits):
    """Calculate memory usage for weights"""
    total_params = sum(p.numel() for p in model.parameters() if 'weight' in str(p))

    # Count actual weight parameters
    weight_params = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            weight_params += param.numel()

    bytes_per_param = num_bits / 8
    memory_bytes = weight_params * bytes_per_param
    memory_kb = memory_bytes / 1024

    return {
        'num_weights': weight_params,
        'bits': num_bits,
        'memory_bytes': memory_bytes,
        'memory_kb': memory_kb
    }


def run_comparison(checkpoint_path, data_dir, variant='D', T=10):
    """Run INT8 vs INT16 comparison"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Variant: {variant}")
    print(f"Checkpoint: {checkpoint_path}")
    print()

    # Load model
    print("Loading model...", flush=True)
    model = create_hls_variant(variant=variant, num_classes=2, T=T)
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"Loaded from epoch {checkpoint.get('epoch', 'unknown')}")

    # Create validation loader
    val_transform = transforms.Compose([
        transforms.Resize(288),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset = SDNET2018Dataset(data_dir, split='val', transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    print(f"Validation samples: {len(val_dataset)}")
    print()

    results = {}

    # ============================================================
    # FP32 Baseline
    # ============================================================
    print("=" * 60)
    print("EVALUATING FP32 BASELINE")
    print("=" * 60)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            functional.reset_net(model.model)

            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx + 1}/{len(val_loader)}", flush=True)

    fp32_accuracy = accuracy_score(all_labels, all_preds) * 100
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    fp32_sensitivity = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    fp32_specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0

    results['FP32'] = {
        'accuracy': fp32_accuracy,
        'sensitivity': fp32_sensitivity,
        'specificity': fp32_specificity,
        'memory_kb': calculate_memory_usage(model, 32)['memory_kb'],
        'bits': 32
    }

    print(f"\nFP32 Results:")
    print(f"  Accuracy:    {fp32_accuracy:.2f}%")
    print(f"  Sensitivity: {fp32_sensitivity:.2f}%")
    print(f"  Specificity: {fp32_specificity:.2f}%")
    print()

    # ============================================================
    # INT8 Quantization
    # ============================================================
    print("=" * 60)
    print("EVALUATING INT8 QUANTIZATION")
    print("=" * 60)

    int8_results = simulate_quantized_inference(model, val_loader, device, num_bits=8)
    int8_memory = calculate_memory_usage(model, 8)

    results['INT8'] = {
        'accuracy': int8_results['accuracy'],
        'sensitivity': int8_results['sensitivity'],
        'specificity': int8_results['specificity'],
        'memory_kb': int8_memory['memory_kb'],
        'bits': 8,
        'accuracy_drop': fp32_accuracy - int8_results['accuracy'],
        'avg_quant_error': int8_results['avg_quant_error']
    }

    print(f"\nINT8 Results:")
    print(f"  Accuracy:    {int8_results['accuracy']:.2f}% (drop: {fp32_accuracy - int8_results['accuracy']:.2f}%)")
    print(f"  Sensitivity: {int8_results['sensitivity']:.2f}%")
    print(f"  Specificity: {int8_results['specificity']:.2f}%")
    print(f"  Memory:      {int8_memory['memory_kb']:.1f} KB")
    print(f"  Quant Error: {int8_results['avg_quant_error']:.6f}")
    print()

    # ============================================================
    # INT16 Quantization
    # ============================================================
    print("=" * 60)
    print("EVALUATING INT16 QUANTIZATION")
    print("=" * 60)

    int16_results = simulate_quantized_inference(model, val_loader, device, num_bits=16)
    int16_memory = calculate_memory_usage(model, 16)

    results['INT16'] = {
        'accuracy': int16_results['accuracy'],
        'sensitivity': int16_results['sensitivity'],
        'specificity': int16_results['specificity'],
        'memory_kb': int16_memory['memory_kb'],
        'bits': 16,
        'accuracy_drop': fp32_accuracy - int16_results['accuracy'],
        'avg_quant_error': int16_results['avg_quant_error']
    }

    print(f"\nINT16 Results:")
    print(f"  Accuracy:    {int16_results['accuracy']:.2f}% (drop: {fp32_accuracy - int16_results['accuracy']:.2f}%)")
    print(f"  Sensitivity: {int16_results['sensitivity']:.2f}%")
    print(f"  Specificity: {int16_results['specificity']:.2f}%")
    print(f"  Memory:      {int16_memory['memory_kb']:.1f} KB")
    print(f"  Quant Error: {int16_results['avg_quant_error']:.6f}")
    print()

    # ============================================================
    # Comparison Summary
    # ============================================================
    print("=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print()
    print(f"{'Metric':<20} {'FP32':>12} {'INT8':>12} {'INT16':>12}")
    print("-" * 60)
    print(f"{'Accuracy (%)':<20} {results['FP32']['accuracy']:>12.2f} {results['INT8']['accuracy']:>12.2f} {results['INT16']['accuracy']:>12.2f}")
    print(f"{'Acc Drop (%)':<20} {'-':>12} {results['INT8']['accuracy_drop']:>12.2f} {results['INT16']['accuracy_drop']:>12.2f}")
    print(f"{'Sensitivity (%)':<20} {results['FP32']['sensitivity']:>12.2f} {results['INT8']['sensitivity']:>12.2f} {results['INT16']['sensitivity']:>12.2f}")
    print(f"{'Specificity (%)':<20} {results['FP32']['specificity']:>12.2f} {results['INT8']['specificity']:>12.2f} {results['INT16']['specificity']:>12.2f}")
    print(f"{'Memory (KB)':<20} {results['FP32']['memory_kb']:>12.1f} {results['INT8']['memory_kb']:>12.1f} {results['INT16']['memory_kb']:>12.1f}")
    print("-" * 60)

    # Recommendation
    print()
    print("RECOMMENDATION:")
    print("-" * 60)

    int8_drop = results['INT8']['accuracy_drop']
    int16_drop = results['INT16']['accuracy_drop']

    if int8_drop <= 1.0:
        recommendation = "INT8"
        reason = f"INT8 accuracy drop ({int8_drop:.2f}%) <= 1%, memory efficient ({results['INT8']['memory_kb']:.0f}KB)"
    elif int16_drop <= 1.0:
        recommendation = "INT16"
        reason = f"INT8 drop too high ({int8_drop:.2f}%), INT16 acceptable ({int16_drop:.2f}%)"
    else:
        recommendation = "QAT Required"
        reason = f"Both INT8 ({int8_drop:.2f}%) and INT16 ({int16_drop:.2f}%) drop > 1%"

    print(f"  Selected: {recommendation}")
    print(f"  Reason:   {reason}")
    print()

    # Save report
    output_dir = os.path.join(os.path.dirname(SCRIPT_DIR), 'quantization_output')
    os.makedirs(output_dir, exist_ok=True)

    report_path = os.path.join(output_dir, 'quantization_comparison.txt')
    with open(report_path, 'w') as f:
        f.write("QUANTIZATION COMPARISON REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Variant: {variant}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Samples: {len(val_dataset)}\n\n")

        f.write(f"{'Metric':<20} {'FP32':>12} {'INT8':>12} {'INT16':>12}\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Accuracy (%)':<20} {results['FP32']['accuracy']:>12.2f} {results['INT8']['accuracy']:>12.2f} {results['INT16']['accuracy']:>12.2f}\n")
        f.write(f"{'Acc Drop (%)':<20} {'-':>12} {results['INT8']['accuracy_drop']:>12.2f} {results['INT16']['accuracy_drop']:>12.2f}\n")
        f.write(f"{'Sensitivity (%)':<20} {results['FP32']['sensitivity']:>12.2f} {results['INT8']['sensitivity']:>12.2f} {results['INT16']['sensitivity']:>12.2f}\n")
        f.write(f"{'Specificity (%)':<20} {results['FP32']['specificity']:>12.2f} {results['INT8']['specificity']:>12.2f} {results['INT16']['specificity']:>12.2f}\n")
        f.write(f"{'Memory (KB)':<20} {results['FP32']['memory_kb']:>12.1f} {results['INT8']['memory_kb']:>12.1f} {results['INT16']['memory_kb']:>12.1f}\n")
        f.write("-" * 60 + "\n\n")

        f.write(f"RECOMMENDATION: {recommendation}\n")
        f.write(f"Reason: {reason}\n")

    print(f"Report saved to: {report_path}")
    print("=" * 60)

    return results, recommendation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare INT8 vs INT16 Quantization')
    parser.add_argument('--variant', type=str, default='D', choices=['D', 'E', 'F'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data-dir', type=str,
                        default='./SDNET2018')
    parser.add_argument('--T', type=int, default=10)

    args = parser.parse_args()

    results, recommendation = run_comparison(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        variant=args.variant,
        T=args.T
    )
