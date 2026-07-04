#!/usr/bin/env python3
"""
Quick evaluation script for saved HLS variant checkpoints.
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from custom_spiking_resnet_hls import create_hls_variant

# Settings
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class SDNET2018Dataset(Dataset):
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
                    for img_name in os.listdir(dir_path):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            self.images.append(os.path.join(dir_path, img_name))
                            self.labels.append(label)

        total = len(self.images)
        indices = np.arange(total)
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


def evaluate(variant, checkpoint_path, data_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    model = create_hls_variant(variant=variant, num_classes=2, T=10)
    model = model.to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Create val dataset
    val_transform = transforms.Compose([
        transforms.Resize(288),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset = SDNET2018Dataset(data_dir, split='val', transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    print(f"Val samples: {len(val_dataset)}")

    # Evaluate
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0) * 100

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = (tn / (tn + fp) * 100) if (tn + fp) > 0 else 0
    sensitivity = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0

    try:
        auc_roc = roc_auc_score(all_labels, all_probs)
    except:
        auc_roc = 0.0

    # Print results
    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS - Variant {variant}")
    print("=" * 60)
    print(f"  Accuracy:    {accuracy:.2f}%")
    print(f"  Precision:   {precision:.2f}%")
    print(f"  Recall:      {recall:.2f}%")
    print(f"  F1-Score:    {f1:.2f}%")
    print(f"  Specificity: {specificity:.2f}%")
    print(f"  Sensitivity: {sensitivity:.2f}%")
    print(f"  AUC-ROC:     {auc_roc:.4f}")
    print("\nConfusion Matrix:")
    print(f"  TN={tn}, FP={fp}")
    print(f"  FN={fn}, TP={tp}")
    print("=" * 60)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'auc_roc': auc_roc
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, required=True, choices=['D', 'E', 'F', 'G', 'H'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data-dir', type=str,
                        default='./SDNET2018')
    args = parser.parse_args()

    evaluate(args.variant, args.checkpoint, args.data_dir)
