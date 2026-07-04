#!/usr/bin/env python3
"""
Optimized Spiking ResNet Training for SDNET2018 Concrete Crack Detection

Identical training pipeline to spiking_concrete.py but uses optimized
Spiking ResNet variants with reduced parameters.

Variants:
  A - Remove layer4 (3-stage, ~2.78M params, -75%)
  B - Thin 4-stage (half channels, ~2.80M params, -75%)
  C - Ultra-light (thin 3-stage, ~0.70M params, -94%)

Usage:
  python spiking_optimized.py --variant A --data-dir <path> --batch-size 8 --num-epochs 20
  python spiking_optimized.py --variant B --data-dir <path> --batch-size 8 --num-epochs 20
  python spiking_optimized.py --variant C --data-dir <path> --batch-size 8 --num-epochs 20
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple, Optional
import json
import time
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix

# Add shared modules to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared'))

# Local imports
from evaluation_metrics import ComprehensiveEvaluator
from cross_validation import CrossValidator, run_cross_validation_experiment
from multi_class_dataset import create_multi_class_datasets, print_dataset_summary
from custom_spiking_resnet import SpikingResNetOptimized, VARIANT_CONFIGS, create_variant

from spikingjelly.activation_based import neuron, functional, surrogate, layer

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience=7, min_delta=0.0001, restore_best_weights=True, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        self.best_loss = float('inf')
        self.best_epoch = 0
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0

    def __call__(self, val_loss, model, epoch):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if self.verbose:
                print(f"Validation loss improved to {val_loss:.6f}")
        else:
            self.wait += 1
            if self.verbose and self.wait >= self.patience // 2:
                print(f"No improvement for {self.wait}/{self.patience} epochs")

            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.verbose:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    print(f"   Best validation loss: {self.best_loss:.6f} at epoch {self.best_epoch + 1}")

                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict({k: v.to(model.device if hasattr(model, 'device') else 'cpu')
                                         for k, v in self.best_weights.items()})
                    if self.verbose:
                        print(f"   Restored best weights from epoch {self.best_epoch + 1}")

                return True

        return False


class TrainingMonitor:
    """Monitor training progress and detect potential issues"""

    def __init__(self, patience_overfitting=5, divergence_threshold=0.1):
        self.patience_overfitting = patience_overfitting
        self.divergence_threshold = divergence_threshold

        self.train_losses = []
        self.val_losses = []
        self.overfitting_count = 0

    def update(self, train_loss, val_loss, epoch):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        if len(self.train_losses) >= 3:
            recent_train = np.mean(self.train_losses[-3:])
            recent_val = np.mean(self.val_losses[-3:])

            if len(self.train_losses) >= 6:
                older_train = np.mean(self.train_losses[-6:-3])
                older_val = np.mean(self.val_losses[-6:-3])

                if (recent_train < older_train) and (recent_val > older_val):
                    self.overfitting_count += 1
                    if self.overfitting_count >= self.patience_overfitting:
                        print(f"Potential overfitting detected at epoch {epoch + 1}")
                        print(f"   Train loss trend: {older_train:.4f} -> {recent_train:.4f}")
                        print(f"   Val loss trend: {older_val:.4f} -> {recent_val:.4f}")
                else:
                    self.overfitting_count = max(0, self.overfitting_count - 1)

    def get_divergence(self):
        if len(self.train_losses) >= 3:
            recent_train = np.mean(self.train_losses[-3:])
            recent_val = np.mean(self.val_losses[-3:])
            return abs(recent_val - recent_train) / recent_train
        return 0.0


class SDNET2018Dataset(Dataset):
    """
    Custom Dataset for SDNET2018 concrete crack images

    Directory structure expected:
    root/
        D/ (Decks)
            CD/ (Cracked)
            UD/ (Uncracked)
        P/ (Pavements)
            CP/
            UP/
        W/ (Walls)
            CW/
            UW/
    """
    def __init__(self, root_dir: str, split: str = 'train', transform=None, train_ratio: float = 0.8):
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

        total_images = len(self.images)
        indices = np.arange(total_images)
        np.random.shuffle(indices)

        train_size = int(total_images * train_ratio)

        if split == 'train':
            indices = indices[:train_size]
        else:
            indices = indices[train_size:]

        self.images = [self.images[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]

        print(f"{split} dataset: {len(self.images)} images")
        print(f"Cracked: {sum(self.labels)}, Uncracked: {len(self.labels) - sum(self.labels)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def create_data_loaders(data_dir: str, batch_size: int = 32, num_workers: int = 4, train_ratio: float = 0.8):
    """Create data loaders with appropriate transforms"""

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(288),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = SDNET2018Dataset(data_dir, split='train', transform=train_transform, train_ratio=train_ratio)
    val_dataset = SDNET2018Dataset(data_dir, split='val', transform=val_transform, train_ratio=train_ratio)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                output = model(data)
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        if batch_idx % 10 == 0:
            print(f'Batch [{batch_idx}/{len(loader)}] Loss: {loss.item():.4f}')

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def train_epoch_with_regularization(model, loader, criterion, optimizer, device, scaler=None, args=None):
    """Train for one epoch with SNN-specific regularization"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    membrane_reg_weight = args.membrane_reg_weight if args else 0.0
    synaptic_reg_weight = args.synaptic_reg_weight if args else 0.0
    spike_rate_reg_weight = args.spike_rate_reg_weight if args else 0.0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                output = model(data)
                classification_loss = criterion(output, target)

                total_loss_batch = classification_loss

                if membrane_reg_weight > 0:
                    membrane_reg = model.get_membrane_regularization()
                    total_loss_batch += membrane_reg_weight * membrane_reg

                if synaptic_reg_weight > 0:
                    synaptic_reg = model.get_synaptic_regularization()
                    total_loss_batch += synaptic_reg_weight * synaptic_reg

                if spike_rate_reg_weight > 0:
                    spike_rate_reg = model.get_spike_rate_regularization()
                    total_loss_batch += spike_rate_reg_weight * spike_rate_reg

            scaler.scale(total_loss_batch).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            classification_loss = criterion(output, target)

            total_loss_batch = classification_loss

            if membrane_reg_weight > 0:
                membrane_reg = model.get_membrane_regularization()
                total_loss_batch += membrane_reg_weight * membrane_reg

            if synaptic_reg_weight > 0:
                synaptic_reg = model.get_synaptic_regularization()
                total_loss_batch += synaptic_reg_weight * synaptic_reg

            if spike_rate_reg_weight > 0:
                spike_rate_reg = model.get_spike_rate_regularization()
                total_loss_batch += spike_rate_reg_weight * spike_rate_reg

            total_loss_batch.backward()
            optimizer.step()

        total_loss += total_loss_batch.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        if batch_idx % 10 == 0:
            log_msg = f'Batch [{batch_idx}/{len(loader)}] Loss: {total_loss_batch.item():.4f}'
            if membrane_reg_weight > 0 or synaptic_reg_weight > 0 or spike_rate_reg_weight > 0:
                log_msg += f' (Class: {classification_loss.item():.4f}'
                if membrane_reg_weight > 0:
                    log_msg += f', Mem: {membrane_reg.item():.6f}'
                if synaptic_reg_weight > 0:
                    log_msg += f', Syn: {synaptic_reg.item():.6f}'
                if spike_rate_reg_weight > 0:
                    log_msg += f', SpkRate: {spike_rate_reg.item():.6f}'
                log_msg += ')'
            print(log_msg)

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def evaluate(model, loader, criterion, device, class_names=None):
    """Evaluate model performance"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

            prob = torch.softmax(output, dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(prob.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy, np.array(all_preds), np.array(all_targets), np.array(all_probs)


def comprehensive_evaluate(model, loader, device, class_names=None):
    """Perform comprehensive evaluation with all metrics"""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    inference_start = time.time()

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            pred = output.argmax(dim=1)
            prob = torch.softmax(output, dim=1)

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(prob.cpu().numpy())

    inference_time = time.time() - inference_start

    if class_names is not None:
        evaluator = ComprehensiveEvaluator(class_names=class_names)
    else:
        evaluator = ComprehensiveEvaluator()

    metrics = evaluator.calculate_comprehensive_metrics(
        np.array(all_targets),
        np.array(all_preds),
        np.array(all_probs)
    )

    metrics['inference_time'] = inference_time
    metrics['inference_time_per_image'] = inference_time / len(all_targets)

    metrics['y_true'] = np.array(all_targets)
    metrics['y_pred'] = np.array(all_preds)
    metrics['y_prob'] = np.array(all_probs)

    return metrics


def save_model_checkpoint(model, optimizer, epoch, val_acc, args, model_name, save_dir, add_timestamp=True):
    """Save model checkpoint with metadata and optional timestamp"""
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') if add_timestamp else None

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'args': vars(args),
        'model_type': 'snn_optimized',
        'model_name': model_name,
        'variant': args.variant,
        'variant_info': model.get_variant_info() if hasattr(model, 'get_variant_info') else None,
        'save_time': str(datetime.now()),
        'timestamp': timestamp
    }

    if timestamp:
        filename = f'{model_name}_{timestamp}.pth'
    else:
        filename = f'{model_name}.pth'

    model_path = os.path.join(save_dir, filename)
    torch.save(checkpoint, model_path)
    print(f'Saved {model_name} to {model_path}')

    if add_timestamp:
        latest_path = os.path.join(save_dir, f'{model_name}_latest.pth')
        torch.save(checkpoint, latest_path)
        print(f'Also saved as latest: {latest_path}')

    return model_path


def save_evaluation_results(results, save_dir, filename, add_timestamp=True):
    """Save evaluation results to JSON file with optional timestamp"""
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') if add_timestamp else None

    json_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            json_results[key] = value.tolist()
        else:
            json_results[key] = value

    json_results['save_metadata'] = {
        'save_time': str(datetime.now()),
        'timestamp': timestamp,
        'original_filename': filename
    }

    if timestamp and add_timestamp:
        base_name = filename.replace('.json', '')
        timestamped_filename = f'{base_name}_{timestamp}.json'
    else:
        timestamped_filename = filename

    results_path = os.path.join(save_dir, timestamped_filename)
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)

    print(f"Results saved to {results_path}")

    if add_timestamp:
        base_name = filename.replace('.json', '')
        latest_path = os.path.join(save_dir, f'{base_name}_latest.json')
        with open(latest_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        print(f"Also saved as latest: {latest_path}")

    return results_path


def print_comprehensive_results(metrics, model_name="Spiking ResNet"):
    """Print comprehensive evaluation results"""
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE EVALUATION RESULTS - {model_name}")
    print(f"{'='*60}")

    print(f"Accuracy:     {metrics.get('accuracy', 0):.4f}")
    print(f"Precision:    {metrics.get('precision', 0):.4f}")
    print(f"Recall:       {metrics.get('recall', 0):.4f}")
    print(f"F1-Score:     {metrics.get('f1_score', 0):.4f}")
    print(f"Specificity:  {metrics.get('specificity', 0):.4f}")
    print(f"AUC-ROC:      {metrics.get('auc_roc', 0):.4f}")

    if 'training_time' in metrics:
        print(f"\nTraining Time:      {metrics['training_time']:.2f} seconds")
    if 'inference_time' in metrics:
        print(f"Inference Time:     {metrics['inference_time']:.4f} seconds")
        print(f"Per Image:          {metrics['inference_time_per_image']*1000:.2f} ms")

    if 'total_parameters' in metrics:
        print(f"\nModel Parameters:   {metrics['total_parameters']:,}")
        print(f"Model Size:         {metrics.get('model_size', 0):.2f} MB")

    print(f"{'='*60}")


def run_cross_validation(args, device):
    """Run cross-validation evaluation"""
    print("\n" + "="*60)
    print("RUNNING CROSS-VALIDATION EVALUATION")
    print("="*60)

    full_dataset = SDNET2018Dataset(
        args.data_dir,
        split='train',
        transform=transforms.Compose([
            transforms.Resize(288),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        train_ratio=1.0
    )

    config = {
        'n_folds': args.cv_folds,
        'train_params': {
            'model_params': {
                'variant': args.variant,
                'spiking_neuron': neuron.IFNode,
                'surrogate_function': surrogate.ATan(),
                'num_classes': args.num_classes,
                'T': args.time_steps
            },
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'num_epochs': args.num_epochs
        },
        'data_params': {
            'batch_size': args.batch_size,
            'num_workers': args.num_workers
        },
        'save_path': os.path.join(args.results_dir, f'snn_variant{args.variant}_cv_results.json')
    }

    results = run_cross_validation_experiment(
        SpikingResNetOptimized, full_dataset, config
    )

    return results


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Optimized Spiking ResNet Training for SDNET2018 Crack Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Variant selection
    parser.add_argument('--variant', type=str, default='A',
                       choices=['A', 'B', 'C'],
                       help='Model variant: A (no layer4, ~2.78M), B (thin 4-stage, ~2.80M), C (ultra-light, ~0.70M)')

    # Data parameters
    parser.add_argument('--data-dir', type=str,
                        default='./SDNET2018',
                        help='Path to SDNET2018 dataset directory')
    parser.add_argument('--class-scheme', type=str, default='binary',
                       choices=['binary', '3class', '6class'],
                       help='Classification scheme: binary, 3class, or 6class')
    parser.add_argument('--dataset-type', type=str, default='all',
                       choices=['all', 'deck', 'pavement', 'wall'],
                       help='Dataset type: all (all structures), deck, pavement, or wall only')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Training batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')

    # Training parameters
    parser.add_argument('--num-epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', '--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay (L2 regularization)')

    # Model parameters
    parser.add_argument('--time-steps', '-T', type=int, default=10,
                       help='Number of time steps for spiking network')
    parser.add_argument('--num-classes', type=int, default=None,
                       help='Number of output classes (auto-determined by class-scheme if not set)')

    # Training options
    parser.add_argument('--no-amp', action='store_true',
                       help='Disable automatic mixed precision training')
    parser.add_argument('--use-weighted-sampling', action='store_true',
                       help='Use weighted sampling for class balancing')
    parser.add_argument('--use-original-dataset', action='store_true',
                       help='Use original dataset implementation instead of multi-class')
    parser.add_argument('--use-enhanced-augmentation', action='store_true',
                       help='Use crack-aware enhanced data augmentation')
    parser.add_argument('--no-enhanced-augmentation', action='store_true',
                       help='Disable enhanced augmentation (use standard augmentation)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')

    # Anti-overfitting parameters
    parser.add_argument('--early-stopping', action='store_true',
                       help='Enable early stopping to prevent overfitting')
    parser.add_argument('--patience', type=int, default=7,
                       help='Number of epochs to wait before early stopping')
    parser.add_argument('--min-delta', type=float, default=0.0001,
                       help='Minimum change in validation loss to qualify as improvement')
    parser.add_argument('--dropout-rate', type=float, default=0.0,
                       help='Dropout rate for regularization (0.0 = no dropout)')
    parser.add_argument('--lr-scheduler', type=str, default='cosine',
                       choices=['cosine', 'plateau', 'step', 'none'],
                       help='Learning rate scheduler type')
    parser.add_argument('--lr-patience', type=int, default=5,
                       help='Patience for ReduceLROnPlateau scheduler')
    parser.add_argument('--lr-factor', type=float, default=0.5,
                       help='Factor by which to reduce learning rate')
    parser.add_argument('--lr-threshold', type=float, default=0.01,
                       help='Threshold for measuring new optimum for ReduceLROnPlateau')
    parser.add_argument('--membrane-reg-weight', type=float, default=0.0,
                       help='Weight for membrane potential regularization')
    parser.add_argument('--synaptic-reg-weight', type=float, default=0.0,
                       help='Weight for synaptic strength regularization')
    parser.add_argument('--spike-rate-reg-weight', type=float, default=0.0,
                       help='Weight for spike rate regularization')
    parser.add_argument('--regularization-preset', type=str, default='none',
                       choices=['none', 'light', 'medium', 'heavy'],
                       help='Preset regularization configuration')

    # Checkpoint and output
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                       help='Directory to save model checkpoints')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory to save evaluation results')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')

    # Data splitting
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Ratio of data to use for training (rest for validation)')

    # Evaluation modes
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'cross_validation', 'comprehensive'],
                       help='Execution mode')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of folds for cross-validation')

    # Evaluation only (no training)
    parser.add_argument('--eval-only', action='store_true',
                       help='Only evaluate pre-trained model, skip training')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to pre-trained model for evaluation')

    # Output options
    parser.add_argument('--no-timestamps', action='store_true',
                       help='Disable timestamps in output filenames')

    return parser.parse_args()


def apply_regularization_preset(args):
    """Apply regularization preset configurations"""
    if args.regularization_preset == 'none':
        return args

    print(f"Applying regularization preset: {args.regularization_preset}")

    if args.regularization_preset == 'light':
        args.early_stopping = True
        args.patience = 10
        args.dropout_rate = max(args.dropout_rate, 0.1)
        args.lr_scheduler = 'cosine'
        print("   Enabled early stopping (patience=10)")
        print("   Set dropout rate to 0.1")

    elif args.regularization_preset == 'medium':
        args.early_stopping = True
        args.patience = 7
        args.dropout_rate = max(args.dropout_rate, 0.15)
        args.lr_scheduler = 'plateau'
        args.lr_patience = 5
        args.membrane_reg_weight = max(args.membrane_reg_weight, 1e-5)
        args.synaptic_reg_weight = max(args.synaptic_reg_weight, 1e-6)
        print("   Enabled early stopping (patience=7)")
        print("   Set dropout rate to 0.15")
        print("   Using ReduceLROnPlateau scheduler")
        print("   Added light SNN-specific regularization")

    elif args.regularization_preset == 'heavy':
        args.early_stopping = True
        args.patience = 5
        args.dropout_rate = max(args.dropout_rate, 0.3)
        args.lr_scheduler = 'plateau'
        args.lr_patience = 3
        args.lr_factor = 0.3
        args.membrane_reg_weight = max(args.membrane_reg_weight, 1e-4)
        args.synaptic_reg_weight = max(args.synaptic_reg_weight, 1e-5)
        args.spike_rate_reg_weight = max(args.spike_rate_reg_weight, 1e-4)
        print("   Enabled aggressive early stopping (patience=5)")
        print("   Set high dropout rate (0.3)")
        print("   Using ReduceLROnPlateau with fast reduction")
        print("   Added strong SNN-specific regularization")

    return args


def create_timestamped_directories(base_save_dir: str, base_results_dir: str,
                                   use_timestamps: bool = True,
                                   class_scheme: str = 'binary',
                                   variant: str = 'A'):
    """Create timestamped directories for organizing outputs"""
    if use_timestamps:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"snn_opt{variant}_{class_scheme}_{timestamp}"

        save_dir = os.path.join(base_save_dir, run_name)
        results_dir = os.path.join(base_results_dir, run_name)

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        print(f"Created timestamped directories for Variant {variant} ({class_scheme}):")
        print(f"  Checkpoints: {save_dir}")
        print(f"  Results: {results_dir}")

        return save_dir, results_dir, run_name
    else:
        save_dir = os.path.join(base_save_dir, f"opt{variant}_{class_scheme}")
        results_dir = os.path.join(base_results_dir, f"opt{variant}_{class_scheme}")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        return save_dir, results_dir, f"snn_opt{variant}_{class_scheme}"


def main():
    """Main training and evaluation function"""
    args = parse_args()

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Print variant info
    variant_config = VARIANT_CONFIGS[args.variant]
    print(f"\n{'='*60}")
    print(f"MODEL VARIANT: {variant_config['name']}")
    print(f"  {variant_config['description']}")
    print(f"  Channels: {variant_config['stage_channels']}")
    print(f"  Stages: {len(variant_config['layers'])}, Blocks per stage: {variant_config['layers']}")
    print(f"{'='*60}")

    print(f"\nArguments: {args}")

    # Apply regularization preset
    args = apply_regularization_preset(args)

    # Determine timestamp usage
    use_timestamps = not args.no_timestamps
    print(f"Using timestamps in folders: {use_timestamps}")

    # Create timestamped directories
    save_dir, results_dir, run_name = create_timestamped_directories(
        args.save_dir, args.results_dir, use_timestamps, args.class_scheme, args.variant
    )

    args.save_dir = save_dir
    args.results_dir = results_dir

    # Handle different modes
    if args.mode == 'cross_validation':
        print("\nRunning Cross-Validation Mode...")
        results = run_cross_validation(args, device)

        save_evaluation_results(
            results,
            args.results_dir,
            f'spiking_resnet_opt{args.variant}_cv_results.json',
            add_timestamp=False
        )

        print("\nCross-validation completed successfully!")
        return results

    elif args.mode == 'comprehensive':
        print("\nRunning Comprehensive Evaluation Mode...")

        print("Phase 1: Standard Training...")
        training_results = main_training_loop(args, device, use_timestamps, run_name)

        print("Phase 2: Cross-Validation...")
        cv_results = run_cross_validation(args, device)

        comprehensive_results = {
            'standard_training': training_results,
            'cross_validation': cv_results,
            'evaluation_date': str(datetime.now()),
            'configuration': vars(args)
        }

        save_evaluation_results(
            comprehensive_results,
            args.results_dir,
            f'spiking_resnet_opt{args.variant}_comprehensive.json',
            add_timestamp=False
        )

        print("\nComprehensive evaluation completed successfully!")
        return comprehensive_results

    else:
        return main_training_loop(args, device, use_timestamps, run_name)


def main_training_loop(args, device, use_timestamps=True, run_name=None):
    """Main training loop for standard mode"""

    # Determine number of classes based on scheme
    if args.num_classes is None:
        class_scheme_map = {'binary': 2, '3class': 3, '6class': 6}
        args.num_classes = class_scheme_map[args.class_scheme]

    dataset_info = None
    class_weights = None

    # Create data loaders
    if args.use_original_dataset:
        train_loader, val_loader = create_data_loaders(
            args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_ratio=args.train_ratio
        )
        print(f"Using original dataset implementation (binary only)")
    else:
        use_enhanced_aug = True
        if args.no_enhanced_augmentation:
            use_enhanced_aug = False
        elif args.use_enhanced_augmentation:
            use_enhanced_aug = True

        dataset_info = create_multi_class_datasets(
            data_dir=args.data_dir,
            class_scheme=args.class_scheme,
            dataset_type=args.dataset_type,
            test_size=1.0 - args.train_ratio,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_weighted_sampling=args.use_weighted_sampling,
            use_enhanced_augmentation=use_enhanced_aug
        )

        train_loader = dataset_info['train_loader']
        val_loader = dataset_info['test_loader']

        print_dataset_summary(dataset_info)

        args.num_classes = dataset_info['class_info']['num_classes']
        class_weights = dataset_info['class_weights']
        print(f"Using multi-class dataset: {args.dataset_type} structures, {args.class_scheme} scheme")
        print(f"Enhanced augmentation: {use_enhanced_aug}")
        print(f"Class weights: {class_weights.tolist()}")

    # Create optimized model
    model = SpikingResNetOptimized(
        variant=args.variant,
        spiking_neuron=neuron.IFNode,
        surrogate_function=surrogate.ATan(),
        num_classes=args.num_classes,
        T=args.time_steps,
        dropout_rate=args.dropout_rate
    ).to(device)

    # Print variant and model info
    variant_info = model.get_variant_info()
    print(f"\nCreated {variant_info['name']} with {args.num_classes} classes for {args.class_scheme} classification")
    print(f"  Architecture: {variant_info['stage_channels']}, {variant_info['num_stages']} stages")
    if args.dropout_rate > 0:
        print(f"  Dropout rate: {args.dropout_rate}")

    # Print anti-overfitting configuration
    if args.early_stopping or args.dropout_rate > 0 or any([args.membrane_reg_weight, args.synaptic_reg_weight, args.spike_rate_reg_weight]):
        print("\nAnti-Overfitting Configuration:")
        if args.early_stopping:
            print(f"   Early stopping: patience={args.patience}, min_delta={args.min_delta}")
        if args.dropout_rate > 0:
            print(f"   Dropout regularization: {args.dropout_rate}")
        if args.lr_scheduler != 'cosine':
            print(f"   Learning rate scheduler: {args.lr_scheduler}")
        if args.membrane_reg_weight > 0:
            print(f"   Membrane potential regularization: {args.membrane_reg_weight}")
        if args.synaptic_reg_weight > 0:
            print(f"   Synaptic strength regularization: {args.synaptic_reg_weight}")
        if args.spike_rate_reg_weight > 0:
            print(f"   Spike rate regularization: {args.spike_rate_reg_weight}")

    # Model size information
    total_params = variant_info['total_parameters']
    model_size_mb = variant_info['model_size_mb']
    print(f"Model parameters: {total_params:,}")
    print(f"Model size: {model_size_mb:.2f} MB")

    # Handle evaluation-only mode
    if args.eval_only:
        if args.model_path and os.path.exists(args.model_path):
            print(f"Loading pre-trained model from {args.model_path}")
            try:
                checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
            except Exception:
                print("Warning: Using unsafe loading due to checkpoint format compatibility")
                checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("Error: --eval-only requires --model-path to be specified and valid")
            return None

        print("Performing comprehensive evaluation...")
        eval_class_names = dataset_info['class_info']['class_names'] if dataset_info is not None else None
        metrics = comprehensive_evaluate(model, val_loader, device, eval_class_names)

        metrics['total_parameters'] = total_params
        metrics['model_size'] = model_size_mb
        metrics['variant'] = args.variant
        metrics['variant_info'] = variant_info

        print_comprehensive_results(metrics, f"Spiking ResNet {variant_info['name']} (Pre-trained)")
        save_evaluation_results(metrics, args.results_dir,
                              f'spiking_resnet_opt{args.variant}_eval_only.json',
                              add_timestamp=False)

        return metrics

    # Load from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming training from {args.resume}")
        try:
            checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        except Exception:
            print("Warning: Using unsafe loading due to checkpoint format compatibility")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    # Loss and optimizer
    if not args.use_original_dataset and class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print(f"Using weighted CrossEntropyLoss with class weights: {class_weights.tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using standard CrossEntropyLoss")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    if args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs
        )
        print(f"Using CosineAnnealingLR scheduler")
    elif args.lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.lr_factor,
            patience=args.lr_patience, threshold=args.lr_threshold
        )
        print(f"Using ReduceLROnPlateau scheduler (patience={args.lr_patience}, factor={args.lr_factor})")
    elif args.lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.num_epochs//3, gamma=0.1
        )
        print(f"Using StepLR scheduler")
    else:
        scheduler = None
        print("No learning rate scheduler")

    # Mixed precision scaler
    use_amp = not args.no_amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    print(f"Mixed precision training: {use_amp}")

    # Initialize anti-overfitting components
    early_stopping = None
    training_monitor = None

    if args.early_stopping:
        early_stopping = EarlyStopping(
            patience=args.patience,
            min_delta=args.min_delta,
            restore_best_weights=True,
            verbose=True
        )
        print(f"Early stopping enabled: patience={args.patience}, min_delta={args.min_delta}")

    if args.early_stopping or any([args.membrane_reg_weight, args.synaptic_reg_weight, args.spike_rate_reg_weight]):
        training_monitor = TrainingMonitor()
        print(f"Training monitoring enabled")

    # Training history
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0

    training_start_time = time.time()

    # Training loop
    model_label = f"Variant {args.variant}"
    print(f"\nStarting training ({model_label}) for {args.num_epochs} epochs...")
    print("="*60)

    for epoch in range(start_epoch, args.num_epochs):
        print(f'\nEpoch {epoch+1}/{args.num_epochs}')
        print('-' * 50)

        train_loss, train_acc = train_epoch_with_regularization(
            model, train_loader, criterion, optimizer,
            device, scaler, args
        )

        val_loss, val_acc, _, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        if scheduler is not None:
            if args.lr_scheduler == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if training_monitor:
            training_monitor.update(train_loss, val_loss, epoch)

        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current LR: {current_lr:.6f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model_checkpoint(
                model, optimizer, epoch, val_acc, args,
                f'best_spiking_resnet_opt{args.variant}', args.save_dir, add_timestamp=False
            )
            print(f'New best model saved with validation accuracy: {val_acc:.2f}%')

        if early_stopping and early_stopping(val_loss, model, epoch):
            print(f"Training stopped early at epoch {epoch + 1}")
            break

    total_training_time = time.time() - training_start_time

    # Final evaluation with best model
    print("\nPerforming final comprehensive evaluation...")
    try:
        best_model_path = os.path.join(args.save_dir, f'best_spiking_resnet_opt{args.variant}.pth')
        checkpoint = torch.load(best_model_path, weights_only=True)
    except Exception:
        print("Warning: Using unsafe loading due to checkpoint format compatibility")
        checkpoint = torch.load(best_model_path, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])

    eval_class_names = dataset_info['class_info']['class_names'] if dataset_info is not None else None
    final_metrics = comprehensive_evaluate(model, val_loader, device, eval_class_names)

    # Add training information
    final_metrics['training_time'] = total_training_time
    final_metrics['total_parameters'] = total_params
    final_metrics['model_size'] = model_size_mb
    final_metrics['best_val_acc'] = best_val_acc
    final_metrics['variant'] = args.variant
    final_metrics['variant_info'] = variant_info
    final_metrics['training_history'] = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }

    # Print comprehensive results
    print_comprehensive_results(final_metrics, f"Spiking ResNet {variant_info['name']}")

    # Classification report
    print("\nDetailed Classification Report:")
    if dataset_info is not None:
        class_names = dataset_info['class_info']['class_names']
    else:
        num_classes = len(np.unique(final_metrics['y_true']))
        if num_classes == 2:
            class_names = ['Uncracked', 'Cracked']
        elif num_classes == 3:
            class_names = ['Deck Cracks', 'Pavement Cracks', 'Wall Cracks']
        elif num_classes == 6:
            class_names = ['Cracked Decks', 'Uncracked Decks', 'Cracked Pavements',
                          'Uncracked Pavements', 'Cracked Walls', 'Uncracked Walls']
        else:
            class_names = [f'Class {i}' for i in range(num_classes)]

    print(classification_report(
        final_metrics['y_true'],
        final_metrics['y_pred'].flatten(),
        target_names=class_names
    ))

    # Save final model
    save_model_checkpoint(
        model, optimizer, args.num_epochs-1, best_val_acc, args,
        f'final_spiking_resnet_opt{args.variant}', args.save_dir, add_timestamp=False
    )

    # Save evaluation results
    save_evaluation_results(
        final_metrics,
        args.results_dir,
        f'spiking_resnet_opt{args.variant}_training_results.json',
        add_timestamp=False
    )

    print(f"\nTraining completed successfully!")
    print(f"Total training time: {total_training_time:.2f} seconds")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

    if run_name:
        create_run_summary(args, run_name, final_metrics)

    return final_metrics


def create_run_summary(args, run_name: str, metrics: dict):
    """Create a summary file in the base directory pointing to the timestamped run"""
    base_save_dir = os.path.dirname(args.save_dir)
    base_results_dir = os.path.dirname(args.results_dir)

    variant_info = metrics.get('variant_info', {})

    summary = {
        'run_name': run_name,
        'run_date': str(datetime.now()),
        'model_type': f'spiking_resnet_opt{args.variant}',
        'variant': args.variant,
        'variant_info': variant_info,
        'best_accuracy': metrics.get('best_val_acc', 0),
        'total_training_time': metrics.get('training_time', 0),
        'metrics': {
            'accuracy': metrics.get('accuracy', 0),
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0),
            'f1_score': metrics.get('f1_score', 0),
            'specificity': metrics.get('specificity', 0),
            'auc_roc': metrics.get('auc_roc', 0),
        },
        'directories': {
            'checkpoints': args.save_dir,
            'results': args.results_dir
        },
        'key_files': {
            'best_model': os.path.join(args.save_dir, f'best_spiking_resnet_opt{args.variant}.pth'),
            'final_model': os.path.join(args.save_dir, f'final_spiking_resnet_opt{args.variant}.pth'),
            'training_results': os.path.join(args.results_dir, f'spiking_resnet_opt{args.variant}_training_results.json')
        },
        'configuration': vars(args)
    }

    summary_path = os.path.join(base_save_dir, f'{run_name}_summary.json')
    os.makedirs(base_save_dir, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Run summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
