#!/usr/bin/env python3
"""
Spiking ResNet Training for SDNET2018 Concrete Crack Detection
Using SpikingJelly Framework

Focused on training and evaluation of Spiking Neural Networks only.
Separated from baseline comparisons and plotting functionality.

Dataset: SDNET2018 - 56,000+ annotated concrete crack images (256x256)
Framework: SpikingJelly with PyTorch backend
Architecture: Spiking ResNet-18 adapted for binary classification
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

# Import evaluation modules
from evaluation_metrics import ComprehensiveEvaluator
from cross_validation import CrossValidator, run_cross_validation_experiment
from multi_class_dataset import create_multi_class_datasets, print_dataset_summary

# SpikingJelly imports
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from spikingjelly.activation_based.model import spiking_resnet

# Fixed split seed — same train/val partition for ALL runs (fair comparison)
SPLIT_SEED = 42

def set_seed(seed):
    """Set all random seeds for a training run."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Default seed (overridden by --seed in main)
set_seed(42)

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=7, min_delta=0.0001, restore_best_weights=True, verbose=True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            restore_best_weights: Whether to restore best weights when stopping
            verbose: Whether to print early stopping messages
        """
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
        """
        Check if training should stop
        
        Returns:
            bool: True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if self.verbose:
                print(f"💾 Validation loss improved to {val_loss:.6f}")
        else:
            self.wait += 1
            if self.verbose and self.wait >= self.patience // 2:
                print(f"⚠️  No improvement for {self.wait}/{self.patience} epochs")
            
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.verbose:
                    print(f"🛑 Early stopping triggered after {epoch + 1} epochs")
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
        """
        Args:
            patience_overfitting: Epochs to wait before flagging overfitting
            divergence_threshold: Threshold for train/val loss divergence
        """
        self.patience_overfitting = patience_overfitting
        self.divergence_threshold = divergence_threshold
        
        self.train_losses = []
        self.val_losses = []
        self.overfitting_count = 0
        
    def update(self, train_loss, val_loss, epoch):
        """Update monitoring with new losses"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        # Check for overfitting (train loss decreasing while val loss increasing)
        if len(self.train_losses) >= 3:
            recent_train = np.mean(self.train_losses[-3:])
            recent_val = np.mean(self.val_losses[-3:])
            
            if len(self.train_losses) >= 6:
                older_train = np.mean(self.train_losses[-6:-3])
                older_val = np.mean(self.val_losses[-6:-3])
                
                # Train loss decreasing but val loss increasing
                if (recent_train < older_train) and (recent_val > older_val):
                    self.overfitting_count += 1
                    if self.overfitting_count >= self.patience_overfitting:
                        print(f"⚠️  Potential overfitting detected at epoch {epoch + 1}")
                        print(f"   Train loss trend: {older_train:.4f} → {recent_train:.4f}")
                        print(f"   Val loss trend: {older_val:.4f} → {recent_val:.4f}")
                else:
                    self.overfitting_count = max(0, self.overfitting_count - 1)
    
    def get_divergence(self):
        """Calculate current train/val loss divergence"""
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
        
        # Define subdirectories
        subdirs = {
            'D': ['CD', 'UD'],  # Decks
            'P': ['CP', 'UP'],  # Pavements
            'W': ['CW', 'UW']   # Walls
        }
        
        # Load all image paths and labels
        for main_dir, sub_list in subdirs.items():
            for sub_dir in sub_list:
                # Label: 1 for cracked (C*), 0 for uncracked (U*)
                label = 1 if sub_dir.startswith('C') else 0
                
                dir_path = os.path.join(root_dir, main_dir, sub_dir)
                if os.path.exists(dir_path):
                    for img_name in os.listdir(dir_path):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            self.images.append(os.path.join(dir_path, img_name))
                            self.labels.append(label)
        
        # Split dataset — fixed partition (SPLIT_SEED=42) for ALL runs
        total_images = len(self.images)
        indices = np.arange(total_images)
        rng = np.random.RandomState(SPLIT_SEED)
        rng.shuffle(indices)
        
        train_size = int(total_images * train_ratio)
        
        if split == 'train':
            indices = indices[:train_size]
        else:  # validation
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

class SpikingResNetCrackDetector(nn.Module):
    """
    Spiking ResNet for crack detection with dropout and regularization
    Modified from SpikingJelly's implementation for binary classification
    """
    def __init__(self, 
                 spiking_neuron: callable = neuron.IFNode,
                 surrogate_function: callable = surrogate.ATan(),
                 detach_reset: bool = True,
                 num_classes: int = 2,
                 zero_init_residual: bool = False,
                 T: int = 4,  # Time steps
                 dropout_rate: float = 0.0):
        super().__init__()
        
        self.T = T
        self.dropout_rate = dropout_rate
        
        # Use SpikingJelly's pre-built spiking ResNet
        # Modify for binary classification
        self.model = spiking_resnet.spiking_resnet18(
            pretrained=False,
            spiking_neuron=spiking_neuron,
            surrogate_function=surrogate_function,
            detach_reset=detach_reset,
            num_classes=num_classes,
            zero_init_residual=zero_init_residual
        )
        
        # Add dropout layers if specified
        if dropout_rate > 0.0:
            # Get the last layer for modification
            if hasattr(self.model, 'fc'):
                in_features = self.model.fc.in_features
                # Replace final layer with dropout + linear
                self.model.fc = nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(in_features, num_classes)
                )
            elif hasattr(self.model, 'classifier'):
                in_features = self.model.classifier.in_features
                self.model.classifier = nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(in_features, num_classes)
                )
        
        # Store neuron references for regularization
        self.spiking_neurons = []
        self._collect_spiking_neurons()
    
    def _collect_spiking_neurons(self):
        """Collect all spiking neurons for regularization"""
        for module in self.model.modules():
            if isinstance(module, neuron.BaseNode):
                self.spiking_neurons.append(module)
        
    def forward(self, x):
        # x shape: [N, C, H, W]
        # Repeat input for T timesteps
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [T, N, C, H, W]
        
        # Process through time steps
        out_spikes = []
        for t in range(self.T):
            out = self.model(x_seq[t])
            out_spikes.append(out)
        
        # Aggregate spikes over time
        out = torch.stack(out_spikes, dim=0).mean(dim=0)  # [N, num_classes]
        
        # Reset the network state
        functional.reset_net(self.model)
        
        return out
    
    def get_membrane_regularization(self):
        """Calculate membrane potential regularization loss"""
        membrane_loss = 0.0
        count = 0
        
        for neuron_module in self.spiking_neurons:
            if hasattr(neuron_module, 'v') and neuron_module.v is not None:
                # Convert to tensor if it's a scalar
                v = neuron_module.v
                if isinstance(v, (int, float)):
                    v = torch.tensor(v, device=next(self.parameters()).device)
                elif isinstance(v, torch.Tensor) and v.numel() > 0:
                    # L2 penalty on membrane potentials to prevent excessive spiking
                    membrane_loss += torch.mean(v ** 2)
                    count += 1
        
        return membrane_loss / max(count, 1) if count > 0 else torch.tensor(0.0, device=next(self.parameters()).device)
    
    def get_synaptic_regularization(self):
        """Calculate synaptic strength regularization loss"""
        synaptic_loss = 0.0
        count = 0
        
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.requires_grad:
                # L2 penalty on weights for stability
                synaptic_loss += torch.norm(param, 2)
                count += 1
        
        return synaptic_loss / max(count, 1) if count > 0 else torch.tensor(0.0, device=next(self.parameters()).device)
    
    def get_spike_rate_regularization(self):
        """Calculate spike rate regularization to prevent over-spiking"""
        spike_rate_loss = 0.0
        count = 0
        
        for neuron_module in self.spiking_neurons:
            if hasattr(neuron_module, 'spike') and neuron_module.spike is not None:
                spike = neuron_module.spike
                if isinstance(spike, (int, float)):
                    spike = torch.tensor(spike, device=next(self.parameters()).device)
                elif isinstance(spike, torch.Tensor) and spike.numel() > 0:
                    # Penalize extremely high or low spike rates
                    spike_rate = torch.mean(spike.float())
                    # Target spike rate around 0.1-0.3 for good information flow
                    target_rate = 0.2
                    spike_rate_loss += (spike_rate - target_rate) ** 2
                    count += 1
        
        return spike_rate_loss / max(count, 1) if count > 0 else torch.tensor(0.0, device=next(self.parameters()).device)

def create_data_loaders(data_dir: str, batch_size: int = 32, num_workers: int = 4, train_ratio: float = 0.8):
    """Create data loaders with appropriate transforms"""
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transform
    val_transform = transforms.Compose([
        transforms.Resize(288),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = SDNET2018Dataset(data_dir, split='train', transform=train_transform, train_ratio=train_ratio)
    val_dataset = SDNET2018Dataset(data_dir, split='val', transform=val_transform, train_ratio=train_ratio)
    
    # Create data loaders
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
            # Mixed precision training
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
    
    # Regularization weights
    membrane_reg_weight = args.membrane_reg_weight if args else 0.0
    synaptic_reg_weight = args.synaptic_reg_weight if args else 0.0
    spike_rate_reg_weight = args.spike_rate_reg_weight if args else 0.0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            # Mixed precision training with regularization
            with torch.amp.autocast('cuda'):
                output = model(data)
                classification_loss = criterion(output, target)
                
                # Add SNN-specific regularization losses
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
            
            # Add SNN-specific regularization losses
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
        
        # Enhanced batch logging with regularization info
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
            
            # Store predictions and probabilities for comprehensive evaluation
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
    
    # Calculate comprehensive metrics
    if class_names is not None:
        evaluator = ComprehensiveEvaluator(class_names=class_names)
    else:
        evaluator = ComprehensiveEvaluator()  # Default binary class names
    
    metrics = evaluator.calculate_comprehensive_metrics(
        np.array(all_targets), 
        np.array(all_preds),
        np.array(all_probs)
    )
    
    # Add timing information
    metrics['inference_time'] = inference_time
    metrics['inference_time_per_image'] = inference_time / len(all_targets)
    
    # Store data for analysis
    metrics['y_true'] = np.array(all_targets)
    metrics['y_pred'] = np.array(all_preds)
    metrics['y_prob'] = np.array(all_probs)
    
    return metrics

def save_model_checkpoint(model, optimizer, epoch, val_acc, args, model_name, save_dir, add_timestamp=True):
    """Save model checkpoint with metadata and optional timestamp"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') if add_timestamp else None
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'args': vars(args),
        'model_type': 'snn',
        'model_name': model_name,
        'save_time': str(datetime.now()),
        'timestamp': timestamp
    }
    
    # Create filename with timestamp
    if timestamp:
        filename = f'{model_name}_{timestamp}.pth'
    else:
        filename = f'{model_name}.pth'
    
    model_path = os.path.join(save_dir, filename)
    torch.save(checkpoint, model_path)
    print(f'Saved {model_name} to {model_path}')
    
    # Also save as latest (without timestamp) for easy access
    if add_timestamp:
        latest_path = os.path.join(save_dir, f'{model_name}_latest.pth')
        torch.save(checkpoint, latest_path)
        print(f'Also saved as latest: {latest_path}')
    
    return model_path

def save_evaluation_results(results, save_dir, filename, add_timestamp=True):
    """Save evaluation results to JSON file with optional timestamp"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') if add_timestamp else None
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            json_results[key] = value.tolist()
        else:
            json_results[key] = value
    
    # Add metadata to results
    json_results['save_metadata'] = {
        'save_time': str(datetime.now()),
        'timestamp': timestamp,
        'original_filename': filename
    }
    
    # Create filename with timestamp
    if timestamp and add_timestamp:
        base_name = filename.replace('.json', '')
        timestamped_filename = f'{base_name}_{timestamp}.json'
    else:
        timestamped_filename = filename
    
    results_path = os.path.join(save_dir, timestamped_filename)
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"Results saved to {results_path}")
    
    # Also save as latest (without timestamp) for easy access
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
    
    # Core metrics
    print(f"Accuracy:     {metrics.get('accuracy', 0):.4f}")
    print(f"Precision:    {metrics.get('precision', 0):.4f}")
    print(f"Recall:       {metrics.get('recall', 0):.4f}")
    print(f"F1-Score:     {metrics.get('f1_score', 0):.4f}")
    print(f"Specificity:  {metrics.get('specificity', 0):.4f}")
    print(f"AUC-ROC:      {metrics.get('auc_roc', 0):.4f}")
    
    # Performance metrics
    if 'training_time' in metrics:
        print(f"\nTraining Time:      {metrics['training_time']:.2f} seconds")
    if 'inference_time' in metrics:
        print(f"Inference Time:     {metrics['inference_time']:.4f} seconds")
        print(f"Per Image:          {metrics['inference_time_per_image']*1000:.2f} ms")
    
    # Model size
    if 'total_parameters' in metrics:
        print(f"\nModel Parameters:   {metrics['total_parameters']:,}")
        print(f"Model Size:         {metrics.get('model_size', 0):.2f} MB")
    
    print(f"{'='*60}")

def run_cross_validation(args, device):
    """Run cross-validation evaluation"""
    print("\n" + "="*60)
    print("RUNNING CROSS-VALIDATION EVALUATION")
    print("="*60)
    
    # Create full dataset for CV
    full_dataset = SDNET2018Dataset(
        args.data_dir, 
        split='train',  # We'll handle splitting in CV
        transform=transforms.Compose([
            transforms.Resize(288),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        train_ratio=1.0  # Use full dataset
    )
    
    # CV configuration
    config = {
        'n_folds': args.cv_folds,
        'train_params': {
            'model_params': {
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
        'save_path': os.path.join(args.results_dir, 'snn_cv_results.json')
    }
    
    # Run CV evaluation
    results = run_cross_validation_experiment(
        SpikingResNetCrackDetector, full_dataset, config
    )
    
    return results

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Spiking ResNet Training for SDNET2018 Concrete Crack Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
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
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for training (split is always 42)')
    
    return parser.parse_args()

def apply_regularization_preset(args):
    """Apply regularization preset configurations"""
    if args.regularization_preset == 'none':
        return args
    
    print(f"🔧 Applying regularization preset: {args.regularization_preset}")
    
    if args.regularization_preset == 'light':
        # Light regularization: basic early stopping + minimal dropout
        args.early_stopping = True
        args.patience = 10
        args.dropout_rate = max(args.dropout_rate, 0.1)
        args.lr_scheduler = 'cosine'
        print("   ✓ Enabled early stopping (patience=10)")
        print("   ✓ Set dropout rate to 0.1")
        
    elif args.regularization_preset == 'medium':
        # Medium regularization: early stopping + dropout + advanced LR + light SNN reg
        args.early_stopping = True
        args.patience = 7
        args.dropout_rate = max(args.dropout_rate, 0.15)
        args.lr_scheduler = 'plateau'
        args.lr_patience = 5
        args.membrane_reg_weight = max(args.membrane_reg_weight, 1e-5)
        args.synaptic_reg_weight = max(args.synaptic_reg_weight, 1e-6)
        print("   ✓ Enabled early stopping (patience=7)")
        print("   ✓ Set dropout rate to 0.15")
        print("   ✓ Using ReduceLROnPlateau scheduler")
        print("   ✓ Added light SNN-specific regularization")
        
    elif args.regularization_preset == 'heavy':
        # Heavy regularization: aggressive early stopping + strong dropout + all regularizations
        args.early_stopping = True
        args.patience = 5
        args.dropout_rate = max(args.dropout_rate, 0.3)
        args.lr_scheduler = 'plateau'
        args.lr_patience = 3
        args.lr_factor = 0.3
        args.membrane_reg_weight = max(args.membrane_reg_weight, 1e-4)
        args.synaptic_reg_weight = max(args.synaptic_reg_weight, 1e-5)
        args.spike_rate_reg_weight = max(args.spike_rate_reg_weight, 1e-4)
        print("   ✓ Enabled aggressive early stopping (patience=5)")
        print("   ✓ Set high dropout rate (0.3)")
        print("   ✓ Using ReduceLROnPlateau with fast reduction")
        print("   ✓ Added strong SNN-specific regularization")
    
    return args

def create_timestamped_directories(base_save_dir: str, base_results_dir: str, use_timestamps: bool = True, class_scheme: str = 'binary'):
    """Create timestamped directories for organizing outputs"""
    if use_timestamps:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"snn_{class_scheme}_{timestamp}"
        
        save_dir = os.path.join(base_save_dir, run_name)
        results_dir = os.path.join(base_results_dir, run_name)
        
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"Created timestamped directories for {class_scheme} classification:")
        print(f"  Checkpoints: {save_dir}")
        print(f"  Results: {results_dir}")
        
        return save_dir, results_dir, run_name
    else:
        # Use original directories without timestamping
        save_dir = os.path.join(base_save_dir, class_scheme)
        results_dir = os.path.join(base_results_dir, class_scheme)
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        return save_dir, results_dir, f"snn_{class_scheme}"

def main():
    """Main training and evaluation function"""
    # Parse command line arguments
    args = parse_args()
    
    # Set training seed from args
    set_seed(args.seed)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Arguments: {args}")
    
    # Apply regularization preset
    args = apply_regularization_preset(args)
    
    # Determine timestamp usage
    use_timestamps = not args.no_timestamps
    print(f"Using timestamps in folders: {use_timestamps}")
    
    # Create timestamped directories
    save_dir, results_dir, run_name = create_timestamped_directories(
        args.save_dir, args.results_dir, use_timestamps, args.class_scheme
    )
    
    # Update args with new directories
    args.save_dir = save_dir
    args.results_dir = results_dir
    
    # Handle different modes
    if args.mode == 'cross_validation':
        print("\nRunning Cross-Validation Mode...")
        results = run_cross_validation(args, device)
        
        # Save CV results
        save_evaluation_results(
            results, 
            args.results_dir, 
            'spiking_resnet_cv_results.json',
            add_timestamp=False  # Folder already timestamped
        )
        
        print("\nCross-validation completed successfully!")
        return results
    
    elif args.mode == 'comprehensive':
        print("\nRunning Comprehensive Evaluation Mode...")
        
        # First run regular training
        print("Phase 1: Standard Training...")
        training_results = main_training_loop(args, device, use_timestamps, run_name)
        
        # Then run cross-validation
        print("Phase 2: Cross-Validation...")
        cv_results = run_cross_validation(args, device)
        
        # Combine results
        comprehensive_results = {
            'standard_training': training_results,
            'cross_validation': cv_results,
            'evaluation_date': str(datetime.now()),
            'configuration': vars(args)
        }
        
        # Save comprehensive results
        save_evaluation_results(
            comprehensive_results,
            args.results_dir,
            'spiking_resnet_comprehensive.json',
            add_timestamp=False  # Folder already timestamped
        )
        
        print("\nComprehensive evaluation completed successfully!")
        return comprehensive_results
    
    else:
        # Standard training mode
        return main_training_loop(args, device, use_timestamps, run_name)

def main_training_loop(args, device, use_timestamps=True, run_name=None):
    """Main training loop for standard mode"""
    
    # Determine number of classes based on scheme
    if args.num_classes is None:
        class_scheme_map = {'binary': 2, '3class': 3, '6class': 6}
        args.num_classes = class_scheme_map[args.class_scheme]
    
    # Initialize variables for scope
    dataset_info = None
    class_weights = None
    
    # Create data loaders
    if args.use_original_dataset:
        # Use original dataset implementation
        train_loader, val_loader = create_data_loaders(
            args.data_dir, 
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_ratio=args.train_ratio
        )
        print(f"Using original dataset implementation (binary only)")
    else:
        # Determine augmentation settings
        use_enhanced_aug = True  # Default to enhanced augmentation
        if args.no_enhanced_augmentation:
            use_enhanced_aug = False
        elif args.use_enhanced_augmentation:
            use_enhanced_aug = True
        
        # Use multi-class dataset implementation
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
        
        # Print dataset summary
        print_dataset_summary(dataset_info)
        
        # Update args with actual class info
        args.num_classes = dataset_info['class_info']['num_classes']
        class_weights = dataset_info['class_weights']
        print(f"Using multi-class dataset: {args.dataset_type} structures, {args.class_scheme} scheme")
        print(f"Enhanced augmentation: {use_enhanced_aug}")
        print(f"Class weights: {class_weights.tolist()}")
    
    # Create model with dropout support
    model = SpikingResNetCrackDetector(
        spiking_neuron=neuron.IFNode,
        surrogate_function=surrogate.ATan(),
        num_classes=args.num_classes,
        T=args.time_steps,
        dropout_rate=args.dropout_rate
    ).to(device)
    
    print(f"Created Spiking ResNet with {args.num_classes} classes for {args.class_scheme} classification")
    if args.dropout_rate > 0:
        print(f"Dropout rate: {args.dropout_rate}")
    
    # Print anti-overfitting configuration
    if args.early_stopping or args.dropout_rate > 0 or any([args.membrane_reg_weight, args.synaptic_reg_weight, args.spike_rate_reg_weight]):
        print("\n🛡️  Anti-Overfitting Configuration:")
        if args.early_stopping:
            print(f"   ✓ Early stopping: patience={args.patience}, min_delta={args.min_delta}")
        if args.dropout_rate > 0:
            print(f"   ✓ Dropout regularization: {args.dropout_rate}")
        if args.lr_scheduler != 'cosine':
            print(f"   ✓ Learning rate scheduler: {args.lr_scheduler}")
        if args.membrane_reg_weight > 0:
            print(f"   ✓ Membrane potential regularization: {args.membrane_reg_weight}")
        if args.synaptic_reg_weight > 0:
            print(f"   ✓ Synaptic strength regularization: {args.synaptic_reg_weight}")
        if args.spike_rate_reg_weight > 0:
            print(f"   ✓ Spike rate regularization: {args.spike_rate_reg_weight}")
    
    # Model size information
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
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
        
        # Perform comprehensive evaluation
        print("Performing comprehensive evaluation...")
        # Get class names if using multi-class dataset
        eval_class_names = dataset_info['class_info']['class_names'] if dataset_info is not None else None
        metrics = comprehensive_evaluate(model, val_loader, device, eval_class_names)
        
        # Add model info
        metrics['total_parameters'] = total_params
        metrics['model_size'] = model_size_mb
        
        # Print and save results
        print_comprehensive_results(metrics, "Spiking ResNet (Pre-trained)")
        save_evaluation_results(metrics, args.results_dir, 'spiking_resnet_eval_only.json', add_timestamp=False)
        
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
        # Use class weights for multi-class balanced training
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
    else:  # none
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
        print(f"🛑 Early stopping enabled: patience={args.patience}, min_delta={args.min_delta}")
    
    if args.early_stopping or any([args.membrane_reg_weight, args.synaptic_reg_weight, args.spike_rate_reg_weight]):
        training_monitor = TrainingMonitor()
        print(f"📊 Training monitoring enabled")
    
    # Training history
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0
    
    training_start_time = time.time()
    
    # Training loop  
    print(f"\nStarting training for {args.num_epochs} epochs...")
    print("="*60)
    
    for epoch in range(start_epoch, args.num_epochs):
        print(f'\nEpoch {epoch+1}/{args.num_epochs}')
        print('-' * 50)
        
        # Train with SNN-specific regularization
        train_loss, train_acc = train_epoch_with_regularization(
            model, train_loader, criterion, optimizer, 
            device, scaler, args
        )
        
        # Evaluate
        val_loss, val_acc, _, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler (handle different scheduler types)
        if scheduler is not None:
            if args.lr_scheduler == 'plateau':
                scheduler.step(val_loss)  # ReduceLROnPlateau needs validation loss
            else:
                scheduler.step()
        
        # Save history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Update training monitor
        if training_monitor:
            training_monitor.update(train_loss, val_loss, epoch)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Show current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current LR: {current_lr:.6f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model_checkpoint(
                model, optimizer, epoch, val_acc, args, 
                'best_spiking_resnet', args.save_dir, add_timestamp=False  # Folder already timestamped
            )
            print(f'New best model saved with validation accuracy: {val_acc:.2f}%')
        
        # Check for early stopping
        if early_stopping and early_stopping(val_loss, model, epoch):
            print(f"🛑 Training stopped early at epoch {epoch + 1}")
            break
    
    total_training_time = time.time() - training_start_time
    
    # Final evaluation with best model
    print("\nPerforming final comprehensive evaluation...")
    try:
        # Load the best model from the current run directory
        best_model_path = os.path.join(args.save_dir, 'best_spiking_resnet.pth')
        checkpoint = torch.load(best_model_path, weights_only=True)
    except Exception:
        print("Warning: Using unsafe loading due to checkpoint format compatibility")
        checkpoint = torch.load(best_model_path, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Comprehensive evaluation
    # Get class names for final evaluation
    eval_class_names = dataset_info['class_info']['class_names'] if dataset_info is not None else None
    final_metrics = comprehensive_evaluate(model, val_loader, device, eval_class_names)
    
    # Add training information
    final_metrics['training_time'] = total_training_time
    final_metrics['total_parameters'] = total_params
    final_metrics['model_size'] = model_size_mb
    final_metrics['best_val_acc'] = best_val_acc
    final_metrics['training_history'] = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }
    
    # Print comprehensive results
    print_comprehensive_results(final_metrics, "Spiking ResNet")
    
    # Classification report
    print("\nDetailed Classification Report:")
    # Determine class names based on actual number of classes
    if dataset_info is not None:
        class_names = dataset_info['class_info']['class_names']
    else:
        # Fallback: determine from actual data
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
        'final_spiking_resnet', args.save_dir, add_timestamp=False  # Folder already timestamped
    )
    
    # Save evaluation results
    save_evaluation_results(
        final_metrics, 
        args.results_dir, 
        'spiking_resnet_training_results.json',
        add_timestamp=False  # Folder already timestamped
    )
    
    print(f"\nTraining completed successfully!")
    print(f"Total training time: {total_training_time:.2f} seconds")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Create a summary link in the base directory
    if run_name:
        create_run_summary(args, run_name, final_metrics)
    
    return final_metrics

def create_run_summary(args, run_name: str, metrics: dict):
    """Create a summary file in the base directory pointing to the timestamped run"""
    base_save_dir = os.path.dirname(args.save_dir)
    base_results_dir = os.path.dirname(args.results_dir)
    
    summary = {
        'run_name': run_name,
        'run_date': str(datetime.now()),
        'model_type': 'spiking_resnet',
        'best_accuracy': metrics.get('best_val_acc', 0),
        'total_training_time': metrics.get('training_time', 0),
        'directories': {
            'checkpoints': args.save_dir,
            'results': args.results_dir
        },
        'key_files': {
            'best_model': os.path.join(args.save_dir, 'best_spiking_resnet.pth'),
            'final_model': os.path.join(args.save_dir, 'final_spiking_resnet.pth'),
            'training_results': os.path.join(args.results_dir, 'spiking_resnet_training_results.json')
        },
        'configuration': vars(args)
    }
    
    # Save in base directories
    summary_path = os.path.join(base_save_dir, f'{run_name}_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"Run summary saved to: {summary_path}")

if __name__ == '__main__':
    main()