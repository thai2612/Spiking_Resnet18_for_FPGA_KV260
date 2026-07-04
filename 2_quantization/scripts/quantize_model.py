#!/usr/bin/env python3
"""
Phase A3: Model Quantization

Convert floating-point model to fixed-point representation.
Supports Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT).

Usage:
    python quantize_model.py --checkpoint <path> --method ptq
    python quantize_model.py --checkpoint <path> --method qat --epochs 5
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
from sklearn.metrics import accuracy_score

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


class SDNET2018Dataset(Dataset):
    """SDNET2018 Dataset loader"""

    def __init__(self, root_dir, split='train', transform=None, train_ratio=0.8):
        self.transform = transform
        self.images = []
        self.labels = []

        subdirs = {'D': ['CD', 'UD'], 'P': ['CP', 'UP'], 'W': ['CW', 'UW']}

        for main_dir, sub_list in subdirs.items():
            for sub_dir in sub_list:
                label = 1 if sub_dir.startswith('C') else 0
                dir_path = os.path.join(root_dir, main_dir, sub_dir)
                if os.path.exists(dir_path):
                    for img_name in sorted(os.listdir(dir_path)):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            self.images.append(os.path.join(dir_path, img_name))
                            self.labels.append(label)

        total = len(self.images)
        indices = np.arange(total)
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(indices)
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


def analyze_weight_distribution(model, output_dir):
    """Analyze weight distribution for quantization"""
    print("\nAnalyzing weight distribution...")

    stats = {}
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            w = param.detach().cpu().numpy().flatten()
            stats[name] = {
                'shape': list(param.shape),
                'min': float(w.min()),
                'max': float(w.max()),
                'mean': float(w.mean()),
                'std': float(w.std()),
                'abs_max': float(np.abs(w).max())
            }

    # Print summary
    print("\nWeight Statistics:")
    print("-" * 80)
    print(f"{'Layer':<40} {'Shape':<20} {'Min':>10} {'Max':>10} {'AbsMax':>10}")
    print("-" * 80)
    for name, s in stats.items():
        shape_str = str(s['shape'])
        print(f"{name[:40]:<40} {shape_str:<20} {s['min']:>10.4f} {s['max']:>10.4f} {s['abs_max']:>10.4f}")

    # Save to file
    stats_path = os.path.join(output_dir, 'weight_statistics.txt')
    with open(stats_path, 'w') as f:
        f.write("Weight Distribution Analysis\n")
        f.write("=" * 80 + "\n")
        for name, s in stats.items():
            f.write(f"\n{name}\n")
            f.write(f"  Shape: {s['shape']}\n")
            f.write(f"  Range: [{s['min']:.6f}, {s['max']:.6f}]\n")
            f.write(f"  Mean: {s['mean']:.6f}, Std: {s['std']:.6f}\n")
            f.write(f"  Abs Max: {s['abs_max']:.6f}\n")

    print(f"\nSaved to: {stats_path}")
    return stats


def compute_scale_and_zero_point(min_val, max_val, num_bits=8, symmetric=True):
    """Compute quantization scale and zero point"""
    if symmetric:
        # Symmetric quantization (zero_point = 0)
        abs_max = max(abs(min_val), abs(max_val))
        qmax = 2 ** (num_bits - 1) - 1  # 127 for int8
        scale = abs_max / qmax if abs_max > 0 else 1.0
        zero_point = 0
    else:
        # Asymmetric quantization
        qmin = 0
        qmax = 2 ** num_bits - 1  # 255 for uint8
        scale = (max_val - min_val) / (qmax - qmin) if max_val > min_val else 1.0
        zero_point = int(round(qmin - min_val / scale))

    return scale, zero_point


def quantize_tensor(tensor, scale, zero_point, num_bits=8, symmetric=True):
    """Quantize tensor to fixed-point"""
    if symmetric:
        qmax = 2 ** (num_bits - 1) - 1
        qmin = -qmax - 1
    else:
        qmin = 0
        qmax = 2 ** num_bits - 1

    quantized = torch.round(tensor / scale) + zero_point
    quantized = torch.clamp(quantized, qmin, qmax)
    return quantized.to(torch.int8 if symmetric else torch.uint8)


def dequantize_tensor(quantized, scale, zero_point):
    """Dequantize tensor back to float"""
    return (quantized.float() - zero_point) * scale


def manual_ptq_quantization(model, calibration_loader, device, output_dir, num_bits=8):
    """
    Manual Post-Training Quantization

    Since SpikingJelly neurons don't support PyTorch's quantization directly,
    we implement manual quantization for HLS export.
    """
    print("\n" + "=" * 60)
    print("MANUAL POST-TRAINING QUANTIZATION")
    print("=" * 60)

    model.eval()
    quant_params = {}

    # Step 1: Collect activation statistics
    print("\nStep 1: Collecting activation statistics...")
    activation_stats = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            if output.dim() > 0:
                if name not in activation_stats:
                    activation_stats[name] = {'min': float('inf'), 'max': float('-inf')}
                activation_stats[name]['min'] = min(activation_stats[name]['min'], output.min().item())
                activation_stats[name]['max'] = max(activation_stats[name]['max'], output.max().item())
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Run calibration
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(calibration_loader):
            images = images.to(device)
            _ = model(images)
            functional.reset_net(model.model)

            if batch_idx >= 50:  # Use 50 batches for calibration
                break
            if (batch_idx + 1) % 10 == 0:
                print(f"  Calibration batch {batch_idx + 1}/50", flush=True)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Step 2: Quantize weights
    print("\nStep 2: Quantizing weights...")
    for name, param in model.named_parameters():
        if 'weight' in name:
            w = param.detach().cpu()
            w_min, w_max = w.min().item(), w.max().item()
            scale, zp = compute_scale_and_zero_point(w_min, w_max, num_bits, symmetric=True)

            # Quantize
            w_quant = quantize_tensor(w, scale, zp, num_bits, symmetric=True)

            # Store quantization params
            quant_params[name] = {
                'scale': scale,
                'zero_point': zp,
                'num_bits': num_bits,
                'shape': list(param.shape),
                'min': w_min,
                'max': w_max,
                'quantized_min': w_quant.min().item(),
                'quantized_max': w_quant.max().item()
            }

            print(f"  {name}: scale={scale:.6f}, range=[{w_quant.min().item()}, {w_quant.max().item()}]")

    # Step 3: Compute activation scales
    print("\nStep 3: Computing activation scales...")
    for name, stats in activation_stats.items():
        scale, zp = compute_scale_and_zero_point(stats['min'], stats['max'], num_bits, symmetric=True)
        quant_params[f'act_{name}'] = {
            'scale': scale,
            'zero_point': zp,
            'num_bits': num_bits,
            'min': stats['min'],
            'max': stats['max']
        }

    # Save quantization parameters
    quant_params_path = os.path.join(output_dir, 'quantization_params.txt')
    with open(quant_params_path, 'w') as f:
        f.write("QUANTIZATION PARAMETERS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Bits: {num_bits}\n")
        f.write(f"Mode: Symmetric\n\n")

        f.write("WEIGHT QUANTIZATION:\n")
        f.write("-" * 60 + "\n")
        for name, params in quant_params.items():
            if not name.startswith('act_'):
                f.write(f"\n{name}\n")
                f.write(f"  Shape: {params['shape']}\n")
                f.write(f"  Scale: {params['scale']:.8f}\n")
                f.write(f"  Zero Point: {params['zero_point']}\n")
                f.write(f"  FP Range: [{params['min']:.6f}, {params['max']:.6f}]\n")
                f.write(f"  INT Range: [{params['quantized_min']}, {params['quantized_max']}]\n")

        f.write("\n\nACTIVATION QUANTIZATION:\n")
        f.write("-" * 60 + "\n")
        for name, params in quant_params.items():
            if name.startswith('act_'):
                f.write(f"\n{name[4:]}\n")  # Remove 'act_' prefix
                f.write(f"  Scale: {params['scale']:.8f}\n")
                f.write(f"  Zero Point: {params['zero_point']}\n")
                f.write(f"  FP Range: [{params['min']:.6f}, {params['max']:.6f}]\n")

    print(f"\nSaved quantization params to: {quant_params_path}")

    return quant_params


def evaluate_quantized_accuracy(model, val_loader, device, quant_params=None):
    """Evaluate model accuracy (simulated quantization)"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            # If we have quant_params, we would apply fake quantization here
            # For now, just evaluate FP32 model as baseline
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            functional.reset_net(model.model)

    accuracy = accuracy_score(all_labels, all_preds) * 100
    return accuracy


def export_quantized_weights(model, quant_params, output_dir, num_bits=8):
    """Export quantized weights as C headers"""
    print("\nExporting quantized weights to C headers...")

    weights_dir = os.path.join(output_dir, '..', 'weights')
    os.makedirs(weights_dir, exist_ok=True)

    # Header file
    header_path = os.path.join(weights_dir, 'weights.h')
    with open(header_path, 'w') as f:
        f.write("// Auto-generated quantized weights for HLS\n")
        f.write(f"// Generated: {datetime.now().isoformat()}\n")
        f.write(f"// Quantization: {num_bits}-bit symmetric\n\n")
        f.write("#ifndef WEIGHTS_H\n")
        f.write("#define WEIGHTS_H\n\n")
        f.write("#include <stdint.h>\n\n")

        # Write scale factors
        f.write("// Scale factors (multiply INT result by scale to get FP value)\n")
        for name, params in quant_params.items():
            if not name.startswith('act_'):
                safe_name = name.replace('.', '_')
                f.write(f"#define {safe_name.upper()}_SCALE {params['scale']:.8e}f\n")

        f.write("\n// Weight arrays (INT8, Q7 format)\n")
        f.write("// Defined in weights.c\n\n")

        # Declare arrays
        for name, param in model.named_parameters():
            if 'weight' in name:
                safe_name = name.replace('.', '_')
                shape = list(param.shape)
                total_size = np.prod(shape)
                f.write(f"extern const int8_t {safe_name}[{total_size}];  // {shape}\n")

        f.write("\n#endif // WEIGHTS_H\n")

    # Source file with actual weights
    source_path = os.path.join(weights_dir, 'weights.c')
    with open(source_path, 'w') as f:
        f.write("// Auto-generated quantized weights\n")
        f.write(f"// Generated: {datetime.now().isoformat()}\n\n")
        f.write('#include "weights.h"\n\n')

        for name, param in model.named_parameters():
            if 'weight' in name:
                safe_name = name.replace('.', '_')
                w = param.detach().cpu()

                # Get quantization params
                if name in quant_params:
                    scale = quant_params[name]['scale']
                    zp = quant_params[name]['zero_point']
                else:
                    # Fallback
                    w_min, w_max = w.min().item(), w.max().item()
                    scale, zp = compute_scale_and_zero_point(w_min, w_max, num_bits, symmetric=True)

                # Quantize
                w_quant = quantize_tensor(w, scale, zp, num_bits, symmetric=True)
                w_flat = w_quant.numpy().flatten()

                # Write array
                shape = list(param.shape)
                total_size = len(w_flat)
                f.write(f"// Shape: {shape}\n")
                f.write(f"const int8_t {safe_name}[{total_size}] = {{\n")

                # Write values in rows of 16
                for i in range(0, len(w_flat), 16):
                    row = w_flat[i:i+16]
                    row_str = ', '.join(f"{v:4d}" for v in row)
                    f.write(f"    {row_str},\n")

                f.write("};\n\n")

    # Format documentation
    format_path = os.path.join(weights_dir, 'format.md')
    with open(format_path, 'w') as f:
        f.write("# Quantized Weight Format\n\n")
        f.write("## Overview\n\n")
        f.write(f"- **Quantization**: {num_bits}-bit symmetric\n")
        f.write("- **Format**: Q7 (1 sign bit, 7 fractional bits for INT8)\n")
        f.write("- **Data Type**: `int8_t` (range: -128 to 127)\n\n")
        f.write("## Conversion\n\n")
        f.write("```c\n")
        f.write("// To convert INT8 back to float:\n")
        f.write("float fp_value = (float)int8_value * SCALE;\n")
        f.write("\n")
        f.write("// For MAC operations:\n")
        f.write("int32_t acc = 0;\n")
        f.write("for (int i = 0; i < N; i++) {\n")
        f.write("    acc += (int32_t)input[i] * (int32_t)weight[i];\n")
        f.write("}\n")
        f.write("// Scale result: output_scale = input_scale * weight_scale\n")
        f.write("```\n\n")
        f.write("## Layer Scales\n\n")
        f.write("| Layer | Scale | Range |\n")
        f.write("|-------|-------|-------|\n")
        for name, params in quant_params.items():
            if not name.startswith('act_'):
                f.write(f"| {name} | {params['scale']:.6e} | [{params['quantized_min']}, {params['quantized_max']}] |\n")

    print(f"  weights.h: {header_path}")
    print(f"  weights.c: {source_path}")
    print(f"  format.md: {format_path}")


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = os.path.join(os.path.dirname(SCRIPT_DIR), 'quantization_output')
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"\nLoading Variant {args.variant} model...")
    model = create_hls_variant(variant=args.variant, num_classes=2, T=args.T)
    checkpoint = torch.load(args.checkpoint, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Create data loaders
    train_transform = transforms.Compose([
        transforms.Resize(288),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(288),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = SDNET2018Dataset(args.data_dir, split='train', transform=train_transform)
    val_dataset = SDNET2018Dataset(args.data_dir, split='val', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Step 1: Evaluate FP32 baseline
    print("\n" + "=" * 60)
    print("EVALUATING FP32 BASELINE")
    print("=" * 60)
    fp32_accuracy = evaluate_quantized_accuracy(model, val_loader, device)
    print(f"FP32 Accuracy: {fp32_accuracy:.2f}%")

    # Step 2: Analyze weights
    analyze_weight_distribution(model, output_dir)

    # Step 3: Quantization
    if args.method == 'ptq':
        quant_params = manual_ptq_quantization(model, train_loader, device, output_dir, args.bits)
    else:
        print("\nQAT not yet implemented. Using PTQ...")
        quant_params = manual_ptq_quantization(model, train_loader, device, output_dir, args.bits)

    # Step 4: Export weights
    export_quantized_weights(model, quant_params, output_dir, args.bits)

    # Summary
    print("\n" + "=" * 60)
    print("QUANTIZATION SUMMARY")
    print("=" * 60)
    print(f"Variant: {args.variant}")
    print(f"Method: {args.method.upper()}")
    print(f"Bits: {args.bits}")
    print(f"FP32 Accuracy: {fp32_accuracy:.2f}%")
    print(f"Output: {output_dir}")
    print()
    print("Next steps:")
    print("  1. Review weight_statistics.txt and quantization_params.txt")
    print("  2. Check weights/ folder for C headers")
    print("  3. Proceed to Phase B (Bit-True C Model)")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quantize SNN Model')
    parser.add_argument('--variant', type=str, default='D', choices=['D', 'E', 'F'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--method', type=str, default='ptq', choices=['ptq', 'qat'])
    parser.add_argument('--bits', type=int, default=8, help='Quantization bits')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--T', type=int, default=10)
    parser.add_argument('--data-dir', type=str,
                        default='./SDNET2018')

    args = parser.parse_args()
    main(args)
