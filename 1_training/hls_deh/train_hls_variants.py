#!/usr/bin/env python3
"""
Training Script for HLS-Optimized Spiking ResNet Variants

This script trains variants D, E, F, G, H which are optimized for FPGA deployment
with reduced feature map sizes to fit BRAM constraints.

IMPORTANT: Training conditions must match previous experiments for fair comparison:
- Batch size: 8
- Epochs: 20
- Learning rate: 0.001
- Time steps (T): 10
- No early stopping (for fair comparison)
- Random seed: 42

Usage:
    python train_hls_variants.py --variant D --num-epochs 20 --batch-size 8 --data-dir /path/to/SDNET2018

    # Or train all variants:
    python train_hls_variants.py --variant all --num-epochs 20 --batch-size 8
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# Note: Using torch.amp.autocast and torch.amp.GradScaler directly (new API)
from torchvision import transforms
from PIL import Image
import numpy as np
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

# Ensure local imports work from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import HLS-optimized models
from custom_spiking_resnet_hls import (
    SpikingResNetHLS, HLS_VARIANT_CONFIGS, create_hls_variant,
    calculate_feature_map_sizes
)

# SpikingJelly imports
from spikingjelly.activation_based import functional

# Fixed split seed — same train/val partition for ALL runs (fair comparison)
SPLIT_SEED = 42

# Training seed — set per run via --seed argument
RANDOM_SEED = 42


def set_seed(seed):
    """Set all random seeds for a training run."""
    global RANDOM_SEED
    RANDOM_SEED = seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class SDNET2018Dataset(Dataset):
    """Dataset for SDNET2018 concrete crack images (256x256)."""

    def __init__(self, root_dir: str, split: str = 'train',
                 transform=None, train_ratio: float = 0.8):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.images = []
        self.labels = []

        subdirs = {
            'D': ['CD', 'UD'],
            'P': ['CP', 'UP'],
            'W': ['CW', 'UW']
        }

        for main_dir, sub_list in subdirs.items():
            for sub_dir in sub_list:
                label = 1 if sub_dir.startswith('C') else 0
                dir_path = os.path.join(root_dir, main_dir, sub_dir)
                if os.path.exists(dir_path):
                    for img_name in os.listdir(dir_path):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            self.images.append(os.path.join(dir_path, img_name))
                            self.labels.append(label)

        # Split dataset — fixed partition (SPLIT_SEED=42) for ALL runs
        # so validation set is identical across seeds (fair comparison)
        total_images = len(self.images)
        indices = np.arange(total_images)
        rng = np.random.RandomState(SPLIT_SEED)
        rng.shuffle(indices)

        train_size = int(total_images * train_ratio)

        if split == 'train':
            indices = indices[:train_size]
        else:
            indices = indices[train_size:]

        self.images = [self.images[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(image_size: int = 256):
    """
    Get train and val transforms for HLS variants.

    NOTE: For 256x256 input (HLS requirement), we use:
    - RandomResizedCrop(256) instead of RandomResizedCrop(224)
    - Same augmentation as original for fair comparison

    Original (224x224): RandomResizedCrop(224), Resize(256)+CenterCrop(224)
    HLS (256x256): RandomResizedCrop(256), Resize(288)+CenterCrop(256)
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),  # Match original style
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Match original
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Validation: Resize larger then center crop (match original pattern)
    val_transform = transforms.Compose([
        transforms.Resize(image_size + 32),  # 256 -> 288 (same ratio as 224 -> 256)
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def evaluate_model(model, dataloader, device, criterion=None):
    """Evaluate model and return comprehensive metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            if criterion:
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                num_batches += 1

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds) * 100,
        'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0) * 100,
        'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0) * 100,
        'f1_score': f1_score(all_labels, all_preds, average='weighted', zero_division=0) * 100,
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
    }

    # Specificity (True Negative Rate)
    cm = confusion_matrix(all_labels, all_preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['specificity'] = (tn / (tn + fp) * 100) if (tn + fp) > 0 else 0.0
        metrics['sensitivity'] = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0

    # AUC-ROC
    try:
        metrics['auc_roc'] = roc_auc_score(all_labels, all_probs)
    except:
        metrics['auc_roc'] = 0.0

    if criterion and num_batches > 0:
        metrics['loss'] = total_loss / num_batches

    return metrics


def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
    """Train for one epoch - matches original spiking_optimized.py"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        if scaler:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pred = outputs.argmax(dim=1, keepdim=True)  # Match original style
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += labels.size(0)

        # Batch logging every 10 batches - same as original
        if batch_idx % 10 == 0:
            print(f'Batch [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}', flush=True)

        # Reset spiking states
        functional.reset_net(model)

    return {
        'loss': total_loss / len(dataloader),
        'accuracy': 100. * correct / total
    }


def train_variant(variant: str, args) -> Dict:
    """Train a single HLS variant and return results."""

    print("\n" + "=" * 80)
    print(f"Training HLS Variant {variant}")
    print("=" * 80)

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create model
    print(f"\nCreating model...")
    model = create_hls_variant(
        variant=variant,
        num_classes=2,
        T=args.time_steps,
        dropout_rate=args.dropout_rate
    )
    model = model.to(device)

    # Print model info
    info = model.get_variant_info()
    print(f"\nModel: {info['name']}")
    print(f"  Description: {info['description']}")
    print(f"  Parameters: {info['total_parameters']:,} ({info['total_parameters']/1e6:.3f}M)")
    print(f"  Peak Feature Map: {info['peak_feature_map_kb']:.1f} KB")
    print(f"  Stem Type: {info['stem_type']}")
    print(f"  Channels: {info['stage_channels']}")

    # Create datasets
    print(f"\nLoading dataset from: {args.data_dir}")
    train_transform, val_transform = get_transforms(args.image_size)

    train_dataset = SDNET2018Dataset(
        args.data_dir, split='train', transform=train_transform,
        train_ratio=args.train_ratio
    )
    val_dataset = SDNET2018Dataset(
        args.data_dir, split='val', transform=val_transform,
        train_ratio=args.train_ratio
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,
                          weight_decay=args.weight_decay)

    # Learning rate scheduler - match original (no eta_min)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs
    )
    print(f"Using CosineAnnealingLR scheduler", flush=True)

    # Mixed precision - use new API to avoid FutureWarning
    use_amp = args.use_amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    print(f"Mixed precision training: {use_amp}", flush=True)

    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_metrics': []
    }

    best_val_acc = 0.0
    best_epoch = 0
    best_metrics = None

    # Create checkpoint directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"snn_hls{variant}_binary_s{args.seed}_{timestamp}"
    checkpoint_dir = os.path.join(args.checkpoint_dir, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    results_dir = os.path.join(args.results_dir, run_name)
    os.makedirs(results_dir, exist_ok=True)

    # Training loop
    print(f"\nStarting training for {args.num_epochs} epochs...", flush=True)
    print("-" * 80, flush=True)

    start_time = time.time()

    for epoch in range(args.num_epochs):
        epoch_start = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )

        # Validate
        val_metrics = evaluate_model(model, val_loader, device, criterion)

        # Update scheduler
        scheduler.step()

        # Record history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_metrics'].append(val_metrics)

        epoch_time = time.time() - epoch_start

        # Print progress
        print(f"Epoch [{epoch+1:2d}/{args.num_epochs}] "
              f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.2f}% | "
              f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.2f}% | "
              f"Time: {epoch_time:.1f}s", flush=True)

        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch
            best_metrics = val_metrics.copy()

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': best_val_acc,
                'val_metrics': best_metrics,
            }, os.path.join(checkpoint_dir, f'best_spiking_resnet_hls{variant}.pth'))

    total_time = time.time() - start_time

    # Save final model
    torch.save({
        'epoch': args.num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': history['val_acc'][-1],
    }, os.path.join(checkpoint_dir, f'final_spiking_resnet_hls{variant}.pth'))

    # Final evaluation with best model
    print("\n" + "-" * 80)
    print("Final Evaluation with Best Model")
    print("-" * 80)

    checkpoint = torch.load(
        os.path.join(checkpoint_dir, f'best_spiking_resnet_hls{variant}.pth'),
        weights_only=False  # Our own checkpoint, safe to load
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    final_metrics = evaluate_model(model, val_loader, device, criterion)

    print(f"\nBest Validation Results (Epoch {best_epoch + 1}):")
    print(f"  Accuracy:    {final_metrics['accuracy']:.2f}%")
    print(f"  Precision:   {final_metrics['precision']:.2f}%")
    print(f"  Recall:      {final_metrics['recall']:.2f}%")
    print(f"  F1-Score:    {final_metrics['f1_score']:.2f}%")
    print(f"  Specificity: {final_metrics.get('specificity', 0):.2f}%")
    print(f"  AUC-ROC:     {final_metrics['auc_roc']:.4f}")

    # Calculate overfitting gap
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    overfitting_gap = final_train_acc - final_val_acc

    print(f"\nOverfitting Analysis:")
    print(f"  Final Train Acc: {final_train_acc:.2f}%")
    print(f"  Final Val Acc:   {final_val_acc:.2f}%")
    print(f"  Train-Val Gap:   {overfitting_gap:.2f}%")

    # Compile results
    results = {
        'run_name': run_name,
        'variant': variant,
        'variant_info': info,
        'training_config': {
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'time_steps': args.time_steps,
            'image_size': args.image_size,
            'train_ratio': args.train_ratio,
            'random_seed': RANDOM_SEED,
            'split_seed': SPLIT_SEED,
        },
        'best_epoch': best_epoch + 1,
        'best_val_accuracy': best_val_acc,
        'final_metrics': final_metrics,
        'overfitting_gap': overfitting_gap,
        'total_training_time_seconds': total_time,
        'history': history,
        'checkpoint_dir': checkpoint_dir,
        'results_dir': results_dir,
    }

    # Save results
    results_file = os.path.join(results_dir, f'training_results_{variant}.json')
    with open(results_file, 'w') as f:
        # Convert numpy types to Python types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        json.dump(convert(results), f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Also save summary
    summary_file = os.path.join(args.checkpoint_dir, f'{run_name}_summary.json')
    summary = {
        'run_name': run_name,
        'variant': variant,
        'variant_info': info,
        'best_accuracy': best_val_acc,
        'best_epoch': best_epoch + 1,
        'final_metrics': final_metrics,
        'overfitting_gap': overfitting_gap,
        'total_training_time': total_time,
        'training_config': results['training_config'],
    }
    with open(summary_file, 'w') as f:
        json.dump(convert(summary), f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Train HLS-Optimized Spiking ResNet Variants',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument('--data-dir', type=str,
                        default='./SDNET2018',
                        help='Path to SDNET2018 dataset')
    parser.add_argument('--image-size', type=int, default=256,
                        help='Input image size (256 for HLS variants)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Train/val split ratio')

    # Model arguments
    parser.add_argument('--variant', type=str, default='D',
                        choices=['D', 'E', 'F', 'G', 'H', 'all'],
                        help='HLS variant to train (D, E, F, G, H, or all)')
    parser.add_argument('--time-steps', type=int, default=10,
                        help='Number of SNN time steps (T)')
    parser.add_argument('--dropout-rate', type=float, default=0.0,
                        help='Dropout rate')

    # Training arguments
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size (use 8 for fair comparison)')
    parser.add_argument('--num-epochs', type=int, default=20,
                        help='Number of epochs (use 20 for fair comparison)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for training (split is always 42)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--use-amp', action='store_true', default=True,
                        help='Use automatic mixed precision')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable automatic mixed precision')

    # Output arguments
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Results directory')

    # Device
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use')

    args = parser.parse_args()

    if args.no_amp:
        args.use_amp = False

    # Set training seed
    set_seed(args.seed)

    # Print configuration
    print("=" * 80)
    print("HLS-Optimized Spiking ResNet Training")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Variant(s):     {args.variant}")
    print(f"  Image size:     {args.image_size}x{args.image_size}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Epochs:         {args.num_epochs}")
    print(f"  Learning rate:  {args.learning_rate}")
    print(f"  Time steps (T): {args.time_steps}")
    print(f"  Random seed:    {args.seed} (split always {SPLIT_SEED})")
    print()

    # Verify fair comparison settings
    if args.batch_size != 8:
        print("WARNING: batch_size != 8, results may not be directly comparable to variants A/B/C")
    if args.num_epochs != 20:
        print("WARNING: num_epochs != 20, results may not be directly comparable to variants A/B/C")
    if args.time_steps != 10:
        print("WARNING: time_steps != 10, results may not be directly comparable to variants A/B/C")

    # Train variant(s)
    all_results = {}

    if args.variant == 'all':
        variants = ['D', 'E', 'F', 'G', 'H']
    else:
        variants = [args.variant]

    for variant in variants:
        results = train_variant(variant, args)
        all_results[variant] = results

    # Print summary comparison
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("Training Summary - All Variants")
        print("=" * 80)
        print(f"\n{'Variant':<12} {'Accuracy':<12} {'Params':<12} {'Peak FM':<12} {'Gap':<10}")
        print("-" * 58)
        for v, r in all_results.items():
            info = r['variant_info']
            print(f"{info['name'][:11]:<12} "
                  f"{r['best_val_accuracy']:.2f}%{'':<5} "
                  f"{info['total_parameters']/1e6:.3f}M{'':<5} "
                  f"{info['peak_feature_map_kb']:.0f}KB{'':<6} "
                  f"{r['overfitting_gap']:.2f}%")

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
