#!/usr/bin/env python3
"""
Phase A2: Golden Reference Generation

Export test inputs and layer-by-layer outputs from trained PyTorch model.
These outputs serve as ground truth for verifying HLS implementation.

Usage:
    python export_golden_ref.py --checkpoint <path> --num-samples 100
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
import csv
from datetime import datetime

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
TRAINING_DIR = os.path.join(REPO_ROOT, '1_training')
sys.path.insert(0, TRAINING_DIR)

from custom_spiking_resnet_hls import create_hls_variant
from spikingjelly.activation_based import functional

# Settings
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


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
                    for img_name in sorted(os.listdir(dir_path)):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            self.images.append(os.path.join(dir_path, img_name))
                            self.labels.append(label)

        total = len(self.images)
        indices = np.arange(total)
        np.random.seed(RANDOM_SEED)  # Ensure reproducibility
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
        return image, self.labels[idx], self.images[idx]


class HookManager:
    """Manages forward hooks to capture intermediate layer outputs"""

    def __init__(self):
        self.outputs = {}
        self.hooks = []

    def _get_hook(self, name):
        def hook(module, input, output):
            # Handle spiking neuron outputs (may be tensor or tuple)
            if isinstance(output, tuple):
                output = output[0]
            self.outputs[name] = output.detach().cpu()
        return hook

    def register_hooks(self, model):
        """Register hooks on key layers"""
        # Hook on stem output
        self.hooks.append(
            model.model.stem.register_forward_hook(self._get_hook('stem'))
        )

        # Hook on each stage output
        for i, stage in enumerate(model.model.stages):
            self.hooks.append(
                stage.register_forward_hook(self._get_hook(f'stage{i+1}'))
            )

        # Hook on avgpool output
        self.hooks.append(
            model.model.avgpool.register_forward_hook(self._get_hook('avgpool'))
        )

        # Hook on fc output (before averaging over T)
        self.hooks.append(
            model.model.fc.register_forward_hook(self._get_hook('fc'))
        )

    def clear_outputs(self):
        self.outputs = {}

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def export_golden_reference(
    checkpoint_path: str,
    output_dir: str,
    data_dir: str,
    variant: str = 'D',
    num_samples: int = 100,
    T: int = 10
):
    """
    Export golden reference data from trained model.

    Args:
        checkpoint_path: Path to trained checkpoint
        output_dir: Output directory for golden reference
        data_dir: SDNET2018 dataset directory
        variant: Model variant (D, E, F)
        num_samples: Number of samples to export
        T: Number of timesteps
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Exporting {num_samples} samples for Variant {variant}")
    print(f"Checkpoint: {checkpoint_path}")
    print()

    # Create output directories
    inputs_dir = os.path.join(output_dir, 'inputs')
    layer_outputs_dir = os.path.join(output_dir, 'layer_outputs')
    labels_dir = os.path.join(output_dir, 'labels')

    os.makedirs(inputs_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Create subdirs for each layer
    layer_names = ['stem', 'stage1', 'stage2', 'stage3', 'stage4', 'avgpool', 'fc']
    if variant == 'F':  # Variant F has only 3 stages
        layer_names = ['stem', 'stage1', 'stage2', 'stage3', 'avgpool', 'fc']

    for layer_name in layer_names:
        os.makedirs(os.path.join(layer_outputs_dir, layer_name), exist_ok=True)

    # Load model
    print("Loading model...")
    model = create_hls_variant(variant=variant, num_classes=2, T=T)
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Setup hooks
    hook_manager = HookManager()
    hook_manager.register_hooks(model)

    # Create dataset
    val_transform = transforms.Compose([
        transforms.Resize(288),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = SDNET2018Dataset(data_dir, split='val', transform=val_transform)
    print(f"Validation set size: {len(dataset)}")

    # Limit samples
    num_samples = min(num_samples, len(dataset))
    print(f"Exporting {num_samples} samples...")
    print()

    # Results tracking
    results = []

    # Process samples
    with torch.no_grad():
        for idx in range(num_samples):
            image, label, img_path = dataset[idx]
            image = image.unsqueeze(0).to(device)  # [1, 3, 256, 256]

            # Clear previous outputs
            hook_manager.clear_outputs()

            # Forward pass
            output = model(image)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(output, dim=1).item()
            confidence = probs[0, pred].item()

            # Save input tensor
            input_np = image.cpu().numpy().astype(np.float32)
            input_path = os.path.join(inputs_dir, f'input_{idx:04d}.bin')
            input_np.tofile(input_path)

            # Save layer outputs
            for layer_name, layer_output in hook_manager.outputs.items():
                # Take last timestep output if temporal
                if layer_output.dim() > 4:
                    layer_output = layer_output[-1]  # Last timestep

                output_np = layer_output.numpy().astype(np.float32)
                output_path = os.path.join(layer_outputs_dir, layer_name, f'output_{idx:04d}.bin')
                output_np.tofile(output_path)

            # Save final output (logits)
            logits_np = output.cpu().numpy().astype(np.float32)
            logits_path = os.path.join(layer_outputs_dir, 'fc', f'logits_{idx:04d}.bin')
            logits_np.tofile(logits_path)

            # Track result
            results.append({
                'idx': idx,
                'image_path': img_path,
                'true_label': label,
                'predicted_label': pred,
                'confidence': confidence,
                'correct': int(pred == label)
            })

            # Progress
            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"  [{idx+1}/{num_samples}] label={label}, pred={pred}, conf={confidence:.4f}", flush=True)

            # Reset spiking neurons
            functional.reset_net(model.model)

    # Remove hooks
    hook_manager.remove_hooks()

    # Save labels/results
    csv_path = os.path.join(labels_dir, 'golden_labels.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['idx', 'image_path', 'true_label', 'predicted_label', 'confidence', 'correct'])
        writer.writeheader()
        writer.writerows(results)

    # Calculate accuracy
    correct = sum(r['correct'] for r in results)
    accuracy = correct / len(results) * 100

    # Save summary
    summary = {
        'variant': variant,
        'checkpoint': checkpoint_path,
        'num_samples': num_samples,
        'T': T,
        'accuracy': accuracy,
        'correct': correct,
        'timestamp': datetime.now().isoformat(),
        'layer_shapes': {}
    }

    # Get layer shapes from first sample
    for layer_name in layer_names:
        sample_file = os.path.join(layer_outputs_dir, layer_name, 'output_0000.bin')
        if os.path.exists(sample_file):
            data = np.fromfile(sample_file, dtype=np.float32)
            summary['layer_shapes'][layer_name] = data.shape

    summary_path = os.path.join(output_dir, 'golden_ref_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("GOLDEN REFERENCE SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Variant: {variant}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Samples: {num_samples}\n")
        f.write(f"Timesteps (T): {T}\n")
        f.write(f"Accuracy: {accuracy:.2f}% ({correct}/{num_samples})\n")
        f.write(f"Timestamp: {summary['timestamp']}\n")
        f.write("\nLayer output shapes:\n")
        for layer_name, shape in summary['layer_shapes'].items():
            f.write(f"  {layer_name}: {shape}\n")
        f.write("\nFiles generated:\n")
        f.write(f"  inputs/: {num_samples} files (input_XXXX.bin)\n")
        for layer_name in layer_names:
            f.write(f"  layer_outputs/{layer_name}/: {num_samples} files\n")
        f.write(f"  labels/golden_labels.csv\n")
        f.write("=" * 60 + "\n")

    print()
    print("=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Samples exported: {num_samples}")
    print(f"Accuracy on exported samples: {accuracy:.2f}%")
    print()
    print("Files generated:")
    print(f"  - inputs/: {num_samples} binary files")
    print(f"  - layer_outputs/: outputs for each layer")
    print(f"  - labels/golden_labels.csv: ground truth")
    print(f"  - golden_ref_summary.txt: summary report")
    print("=" * 60)

    return summary


def verify_golden_reference(output_dir: str, num_check: int = 5):
    """Verify saved golden reference files can be loaded correctly"""
    print("\nVerifying golden reference files...")

    # Check inputs
    inputs_dir = os.path.join(output_dir, 'inputs')
    input_files = sorted([f for f in os.listdir(inputs_dir) if f.endswith('.bin')])
    print(f"  Input files: {len(input_files)}")

    if input_files:
        # Load and check first few
        for i in range(min(num_check, len(input_files))):
            data = np.fromfile(os.path.join(inputs_dir, input_files[i]), dtype=np.float32)
            expected_size = 1 * 3 * 256 * 256  # [1, 3, 256, 256]
            if data.size == expected_size:
                print(f"    {input_files[i]}: OK (shape can be reshaped to [1,3,256,256])")
            else:
                print(f"    {input_files[i]}: ERROR (size={data.size}, expected={expected_size})")

    # Check labels
    labels_file = os.path.join(output_dir, 'labels', 'golden_labels.csv')
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            print(f"  Labels file: {len(rows)} entries")

    print("Verification complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export Golden Reference for HLS')
    parser.add_argument('--variant', type=str, default='D', choices=['D', 'E', 'F'],
                        help='Model variant')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained checkpoint (.pth)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: 2_quantization/golden_ref)')
    parser.add_argument('--data-dir', type=str,
                        default='./SDNET2018',
                        help='SDNET2018 dataset directory')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of samples to export')
    parser.add_argument('--T', type=int, default=10,
                        help='Number of timesteps')
    parser.add_argument('--verify', action='store_true',
                        help='Verify exported files after generation')

    args = parser.parse_args()

    # Set default output dir
    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(SCRIPT_DIR),
            'golden_ref'
        )

    # Run export
    export_golden_reference(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        variant=args.variant,
        num_samples=args.num_samples,
        T=args.T
    )

    # Verify if requested
    if args.verify:
        verify_golden_reference(args.output_dir)
