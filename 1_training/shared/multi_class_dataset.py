#!/usr/bin/env python3
"""
Multi-Class SDNET2018 Dataset Implementation
Supports both 3-class and 6-class crack detection schemes with proper class balancing

Author: Multi-class crack detection implementation
"""

import os
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter
import json
import random

class CrackAwareTransform:
    """Transform that applies different augmentation intensity based on crack presence"""
    
    def __init__(self, base_transform, enhanced_transform):
        self.base_transform = base_transform
        self.enhanced_transform = enhanced_transform
    
    def __call__(self, img, is_cracked=None):
        # If is_cracked is not provided, apply standard transform
        if is_cracked is None:
            return self.base_transform(img)
            
        if is_cracked:
            # Apply enhanced augmentation for cracked samples with higher probability
            if random.random() < 0.8:  # 80% chance for cracked samples
                return self.enhanced_transform(img)
            else:
                return self.base_transform(img)
        else:
            # Apply standard augmentation for uncracked samples with lower probability
            if random.random() < 0.5:  # 50% chance for uncracked samples
                return self.base_transform(img)
            else:
                # Minimal augmentation for uncracked samples
                minimal_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                return minimal_transform(img)

def create_enhanced_transforms(input_size: int = 224):
    """Create enhanced transforms for crack-aware augmentation"""
    
    # Standard augmentation (for uncracked and baseline)
    base_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Enhanced augmentation (for cracked samples)
    enhanced_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(0.7),  # Higher probability
        transforms.RandomRotation(20),  # More rotation
        transforms.RandomVerticalFlip(0.3),  # Add vertical flip
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),  # Stronger color jitter
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Add translation and scaling
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Test transform (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return base_transform, enhanced_transform, test_transform

class MultiClassSDNET2018Dataset(Dataset):
    """
    Multi-class SDNET2018 Dataset with configurable class schemes
    
    Supports three classification schemes:
    - 'binary': Traditional cracked (1) vs uncracked (0) - 2 classes
    - '3class': Structure-based crack detection - 3 classes
      - 0: Deck cracks, 1: Pavement cracks, 2: Wall cracks
    - '6class': Full classification - 6 classes  
      - 0: CD, 1: UD, 2: CP, 3: UP, 4: CW, 5: UW
    """
    
    def __init__(self, 
                 data_dir: str,
                 class_scheme: str = 'binary',
                 dataset_type: str = 'all',
                 split: str = 'train',
                 test_size: float = 0.2,
                 random_state: int = 42,
                 transform: Optional[transforms.Compose] = None):
        """
        Initialize multi-class dataset
        
        Args:
            data_dir: Path to SDNET2018 dataset directory
            class_scheme: 'binary', '3class', or '6class'
            dataset_type: 'all', 'deck', 'pavement', or 'wall' - which structures to include
            split: 'train', 'test', or 'all'
            test_size: Proportion of data for test split
            random_state: Random seed for reproducible splits
            transform: Optional image transformations
        """
        
        self.data_dir = data_dir
        self.class_scheme = class_scheme
        self.dataset_type = dataset_type
        self.split = split
        self.transform = transform
        self.random_state = random_state
        
        # Define class schemes
        self.class_schemes = {
            'binary': {
                'num_classes': 2,
                'class_names': ['Uncracked', 'Cracked'],
                'description': 'Binary crack detection'
            },
            '3class': {
                'num_classes': 3,
                'class_names': ['Deck Cracks', 'Pavement Cracks', 'Wall Cracks'],
                'description': 'Structure-based crack detection'
            },
            '6class': {
                'num_classes': 6,
                'class_names': ['Cracked Decks', 'Uncracked Decks', 'Cracked Pavements', 
                               'Uncracked Pavements', 'Cracked Walls', 'Uncracked Walls'],
                'description': 'Full structural classification'
            }
        }
        
        if class_scheme not in self.class_schemes:
            raise ValueError(f"class_scheme must be one of {list(self.class_schemes.keys())}")
        
        # Load and process dataset
        self.image_paths = []
        self.labels = []
        self.category_labels = []  # Store original categories for reference
        self.is_cracked = []  # Store crack status for each sample (for augmentation)
        
        self._load_dataset()
        self._create_train_test_split(test_size)
        
        # Calculate class weights for balanced training
        self.class_weights = self._calculate_class_weights()
        
        print(f"Loaded {len(self.image_paths)} images for {split} split")
        print(f"Dataset type: {dataset_type} | Class scheme: {class_scheme} ({self.class_schemes[class_scheme]['description']})")
        print(f"Class distribution: {self._get_class_distribution()}")
    
    def _load_dataset(self):
        """Load dataset and assign labels based on class scheme"""
        
        # Define category mapping
        category_mapping = {
            'CD': ('D', 'C'),  # Cracked Decks
            'UD': ('D', 'U'),  # Uncracked Decks
            'CP': ('P', 'C'),  # Cracked Pavements
            'UP': ('P', 'U'),  # Uncracked Pavements
            'CW': ('W', 'C'),  # Cracked Walls
            'UW': ('W', 'U')   # Uncracked Walls
        }
        
        # Filter categories based on dataset_type
        if self.dataset_type == 'all':
            allowed_structures = ['D', 'P', 'W']
        elif self.dataset_type == 'deck':
            allowed_structures = ['D']
        elif self.dataset_type == 'pavement':
            allowed_structures = ['P']
        elif self.dataset_type == 'wall':
            allowed_structures = ['W']
        else:
            raise ValueError(f"Invalid dataset_type: {self.dataset_type}. Must be 'all', 'deck', 'pavement', or 'wall'")
        
        # Load images from each category
        for category, (structure, crack_status) in category_mapping.items():
            # Skip if this structure is not allowed for current dataset_type
            if structure not in allowed_structures:
                continue
                
            category_dir = os.path.join(self.data_dir, structure, category)
            
            if not os.path.exists(category_dir):
                print(f"Warning: Category directory {category_dir} not found")
                continue
            
            # Get all image files
            image_files = [f for f in os.listdir(category_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            for image_file in image_files:
                image_path = os.path.join(category_dir, image_file)
                
                # Assign label based on class scheme and determine if we should include this sample
                include_sample = True
                label = None
                
                if self.class_scheme == 'binary':
                    # Binary: 0 = Uncracked, 1 = Cracked
                    label = 1 if crack_status == 'C' else 0
                elif self.class_scheme == '3class':
                    # 3-class: Only cracked samples, classified by structure
                    if crack_status == 'C':
                        if structure == 'D':
                            label = 0  # Deck cracks
                        elif structure == 'P':
                            label = 1  # Pavement cracks
                        elif structure == 'W':
                            label = 2  # Wall cracks
                    else:
                        # Skip uncracked samples for 3-class
                        include_sample = False
                elif self.class_scheme == '6class':
                    # 6-class: Full classification
                    category_to_label = {
                        'CD': 0, 'UD': 1, 'CP': 2, 'UP': 3, 'CW': 4, 'UW': 5
                    }
                    label = category_to_label[category]
                
                # Only add to dataset if we should include this sample
                if include_sample and label is not None:
                    self.image_paths.append(image_path)
                    self.category_labels.append(category)
                    self.labels.append(label)
                    self.is_cracked.append(crack_status == 'C')  # True if cracked, False if uncracked
        
        print(f"Loaded {len(self.image_paths)} images with {self.class_scheme} scheme")
    
    def _create_train_test_split(self, test_size: float):
        """Create stratified train/test split"""
        if self.split == 'all':
            return
        
        from sklearn.model_selection import train_test_split
        
        # Stratified split to maintain class distribution
        train_indices, test_indices = train_test_split(
            range(len(self.image_paths)),
            test_size=test_size,
            stratify=self.labels,
            random_state=self.random_state
        )
        
        if self.split == 'train':
            indices = train_indices
        else:  # test
            indices = test_indices
        
        # Filter data based on split
        self.image_paths = [self.image_paths[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        self.category_labels = [self.category_labels[i] for i in indices]
        self.is_cracked = [self.is_cracked[i] for i in indices]
    
    def _calculate_class_weights(self) -> torch.Tensor:
        """Calculate class weights for balanced training"""
        class_counts = Counter(self.labels)
        total_samples = len(self.labels)
        num_classes = self.class_schemes[self.class_scheme]['num_classes']
        
        # Calculate inverse frequency weights
        weights = []
        for class_idx in range(num_classes):
            count = class_counts.get(class_idx, 1)  # Avoid division by zero
            weight = total_samples / (num_classes * count)
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def _get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution for current split"""
        class_counts = Counter(self.labels)
        class_names = self.class_schemes[self.class_scheme]['class_names']
        
        distribution = {}
        for class_idx, class_name in enumerate(class_names):
            distribution[class_name] = class_counts.get(class_idx, 0)
        
        return distribution
    
    def get_weighted_sampler(self) -> WeightedRandomSampler:
        """Create weighted sampler for balanced training"""
        sample_weights = []
        class_weights = self.class_weights
        
        for label in self.labels:
            sample_weights.append(class_weights[label].item())
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    
    def get_class_info(self) -> Dict:
        """Get comprehensive class information"""
        return {
            'scheme': self.class_scheme,
            'num_classes': self.class_schemes[self.class_scheme]['num_classes'],
            'class_names': self.class_schemes[self.class_scheme]['class_names'],
            'description': self.class_schemes[self.class_scheme]['description'],
            'class_weights': self.class_weights.tolist(),
            'class_distribution': self._get_class_distribution(),
            'total_samples': len(self.labels)
        }
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get image and label with crack-aware augmentation"""
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        is_sample_cracked = self.is_cracked[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (256, 256), color='black')
        
        # Apply transformations
        if self.transform:
            # Check if using crack-aware transform
            if isinstance(self.transform, CrackAwareTransform):
                image = self.transform(image, is_sample_cracked)
            else:
                # Standard transform (for backward compatibility)
                image = self.transform(image)
        
        return image, label

def create_multi_class_datasets(data_dir: str, 
                               class_scheme: str = 'binary',
                               dataset_type: str = 'all',
                               test_size: float = 0.2,
                               random_state: int = 42,
                               batch_size: int = 32,
                               num_workers: int = 4,
                               use_weighted_sampling: bool = True,
                               use_enhanced_augmentation: bool = True) -> Dict:
    """
    Create train and test datasets with appropriate transforms and samplers
    
    Args:
        data_dir: Path to SDNET2018 dataset
        class_scheme: 'binary', '3class', or '6class'
        dataset_type: 'all', 'deck', 'pavement', or 'wall' - which structures to include
        test_size: Proportion for test split
        random_state: Random seed
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        use_weighted_sampling: Whether to use weighted sampling for balancing
        use_enhanced_augmentation: Whether to use crack-aware enhanced augmentation
    
    Returns:
        Dictionary containing datasets, loaders, and class information
    """
    
    # Define transforms
    if use_enhanced_augmentation:
        # Use crack-aware enhanced augmentation
        base_transform, enhanced_transform, test_transform = create_enhanced_transforms()
        train_transform = CrackAwareTransform(base_transform, enhanced_transform)
        print("Using enhanced crack-aware augmentation")
    else:
        # Use standard augmentation
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Test transforms without augmentation
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print("Using standard augmentation")
    
    # Create datasets
    train_dataset = MultiClassSDNET2018Dataset(
        data_dir=data_dir,
        class_scheme=class_scheme,
        dataset_type=dataset_type,
        split='train',
        test_size=test_size,
        random_state=random_state,
        transform=train_transform
    )
    
    test_dataset = MultiClassSDNET2018Dataset(
        data_dir=data_dir,
        class_scheme=class_scheme,
        dataset_type=dataset_type,
        split='test',
        test_size=test_size,
        random_state=random_state,
        transform=test_transform
    )
    
    # Create data loaders
    train_loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True,
        'drop_last': True
    }
    
    test_loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True,
        'shuffle': False
    }
    
    # Add weighted sampler for training if requested
    if use_weighted_sampling:
        train_loader_kwargs['sampler'] = train_dataset.get_weighted_sampler()
        train_loader_kwargs['shuffle'] = False  # Can't use shuffle with sampler
    else:
        train_loader_kwargs['shuffle'] = True
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    test_loader = DataLoader(test_dataset, **test_loader_kwargs)
    
    # Get class information
    class_info = train_dataset.get_class_info()
    
    return {
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'train_loader': train_loader,
        'test_loader': test_loader,
        'class_info': class_info,
        'class_weights': train_dataset.class_weights
    }

def print_dataset_summary(dataset_info: Dict):
    """Print comprehensive dataset summary"""
    
    class_info = dataset_info['class_info']
    
    print("\n" + "="*60)
    print("MULTI-CLASS DATASET SUMMARY")
    print("="*60)
    
    print(f"Class Scheme: {class_info['scheme']}")
    print(f"Description: {class_info['description']}")
    print(f"Number of Classes: {class_info['num_classes']}")
    
    print(f"\nClass Names:")
    for i, name in enumerate(class_info['class_names']):
        print(f"  {i}: {name}")
    
    print(f"\nClass Distribution:")
    for class_name, count in class_info['class_distribution'].items():
        percentage = (count / class_info['total_samples']) * 100
        print(f"  {class_name}: {count:,} ({percentage:.1f}%)")
    
    print(f"\nClass Weights for Balanced Training:")
    for i, (name, weight) in enumerate(zip(class_info['class_names'], class_info['class_weights'])):
        print(f"  {i} ({name}): {weight:.3f}")
    
    print(f"\nDataset Statistics:")
    print(f"  Total Training Samples: {len(dataset_info['train_dataset']):,}")
    print(f"  Total Test Samples: {len(dataset_info['test_dataset']):,}")
    print(f"  Total Samples: {class_info['total_samples']:,}")
    
    # Calculate imbalance ratio
    max_count = max(class_info['class_distribution'].values())
    min_count = min(class_info['class_distribution'].values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print(f"\nImbalance Analysis:")
    print(f"  Imbalance Ratio: {imbalance_ratio:.2f}:1")
    if imbalance_ratio < 3:
        print("  ✅ Acceptable imbalance")
    elif imbalance_ratio < 5:
        print("  ⚠️  Moderate imbalance - consider class weights")
    else:
        print("  ❌ Severe imbalance - class weights strongly recommended")

# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test multi-class dataset implementation')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to SDNET2018 dataset directory')
    parser.add_argument('--class-scheme', type=str, default='binary',
                       choices=['binary', '3class', '6class'],
                       help='Classification scheme to test')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for testing')
    
    args = parser.parse_args()
    
    print(f"Testing multi-class dataset with {args.class_scheme} scheme...")
    
    # Create dataset
    dataset_info = create_multi_class_datasets(
        data_dir=args.data_dir,
        class_scheme=args.class_scheme,
        batch_size=args.batch_size,
        use_weighted_sampling=True
    )
    
    # Print summary
    print_dataset_summary(dataset_info)
    
    # Test data loading
    print(f"\nTesting data loading...")
    train_loader = dataset_info['train_loader']
    
    for i, (images, labels) in enumerate(train_loader):
        print(f"Batch {i+1}: Images shape: {images.shape}, Labels: {labels.tolist()}")
        if i >= 2:  # Test first 3 batches
            break
    
    print(f"\n✅ Multi-class dataset implementation working correctly!")