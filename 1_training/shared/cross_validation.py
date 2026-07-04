"""
Cross-Validation Framework for SNN Crack Detection
Following CrackVision methodology with 5-fold stratified CV
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import copy
import os
from typing import Dict, List, Tuple, Any
import json
from evaluation_metrics import ComprehensiveEvaluator

class CrossValidator:
    """5-fold stratified cross-validation following CrackVision methodology"""
    
    def __init__(self, n_folds: int = 5, random_state: int = 42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        self.evaluator = ComprehensiveEvaluator()
        
    def prepare_stratified_folds(self, dataset: Dataset) -> List[Tuple[List[int], List[int]]]:
        """
        Create stratified folds ensuring class balance
        
        Args:
            dataset: PyTorch dataset with labels
            
        Returns:
            List of (train_indices, val_indices) tuples for each fold
        """
        # Extract all labels
        labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.append(label)
        
        labels = np.array(labels)
        
        # Create stratified folds
        folds = []
        for train_idx, val_idx in self.skf.split(np.zeros(len(labels)), labels):
            folds.append((train_idx.tolist(), val_idx.tolist()))
        
        print(f"Created {self.n_folds} stratified folds")
        for i, (train_idx, val_idx) in enumerate(folds):
            train_labels = labels[train_idx]
            val_labels = labels[val_idx]
            
            print(f"Fold {i+1}:")
            print(f"  Training: {len(train_idx)} samples, "
                  f"Cracked: {np.sum(train_labels)}, Uncracked: {len(train_labels) - np.sum(train_labels)}")
            print(f"  Validation: {len(val_idx)} samples, "
                  f"Cracked: {np.sum(val_labels)}, Uncracked: {len(val_labels) - np.sum(val_labels)}")
        
        return folds
    
    def run_cv_evaluation(self, model_class: Any, dataset: Dataset, 
                         train_params: Dict, data_params: Dict) -> Dict:
        """
        Run cross-validation evaluation
        
        Args:
            model_class: Model class to instantiate
            dataset: Full dataset
            train_params: Training parameters
            data_params: Data loading parameters
            
        Returns:
            Dictionary containing CV results
        """
        folds = self.prepare_stratified_folds(dataset)
        fold_results = []
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Running CV on device: {device}")
        
        for fold_idx, (train_indices, val_indices) in enumerate(folds):
            print(f"\n{'='*50}")
            print(f"FOLD {fold_idx + 1}/{self.n_folds}")
            print(f"{'='*50}")
            
            # Create fold datasets
            train_subset = Subset(dataset, train_indices)
            val_subset = Subset(dataset, val_indices)
            
            # Create data loaders
            train_loader = DataLoader(
                train_subset, 
                batch_size=data_params['batch_size'],
                shuffle=True,
                num_workers=data_params.get('num_workers', 4)
            )
            
            val_loader = DataLoader(
                val_subset,
                batch_size=data_params['batch_size'],
                shuffle=False,
                num_workers=data_params.get('num_workers', 4)
            )
            
            # Initialize model
            model = model_class(**train_params['model_params']).to(device)
            
            # Train model for this fold
            fold_result = self._train_and_evaluate_fold(
                model, train_loader, val_loader, train_params, fold_idx + 1
            )
            
            fold_results.append(fold_result)
        
        # Aggregate results
        cv_results = self._aggregate_fold_results(fold_results)
        
        return cv_results
    
    def _train_and_evaluate_fold(self, model: nn.Module, train_loader: DataLoader,
                                val_loader: DataLoader, train_params: Dict, 
                                fold_num: int) -> Dict:
        """
        Train and evaluate model for a single fold
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            train_params: Training parameters
            fold_num: Fold number (for logging)
            
        Returns:
            Dictionary containing fold results
        """
        device = next(model.parameters()).device
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_params.get('learning_rate', 1e-3),
            weight_decay=train_params.get('weight_decay', 1e-4)
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_params.get('num_epochs', 20)
        )
        
        # Training loop
        best_val_acc = 0
        best_model_state = None
        
        for epoch in range(train_params.get('num_epochs', 20)):
            # Training phase
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pred = output.argmax(dim=1)
                train_correct += pred.eq(target).sum().item()
                train_total += target.size(0)
            
            scheduler.step()
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            all_preds = []
            all_targets = []
            all_probs = []
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    pred = output.argmax(dim=1)
                    val_correct += pred.eq(target).sum().item()
                    val_total += target.size(0)
                    
                    all_preds.extend(pred.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
                    all_probs.extend(torch.softmax(output, dim=1).cpu().numpy())
            
            val_acc = val_correct / val_total
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
            
            if epoch % 5 == 0:
                print(f"Fold {fold_num}, Epoch {epoch+1}: "
                      f"Train Acc: {train_correct/train_total:.4f}, "
                      f"Val Acc: {val_acc:.4f}")
        
        # Load best model and get final predictions
        model.load_state_dict(best_model_state)
        model.eval()
        
        final_preds = []
        final_targets = []
        final_probs = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                
                final_preds.extend(pred.cpu().numpy())
                final_targets.extend(target.cpu().numpy())
                final_probs.extend(torch.softmax(output, dim=1).cpu().numpy())
        
        # Calculate comprehensive metrics
        fold_metrics = self.evaluator.calculate_comprehensive_metrics(
            np.array(final_targets), 
            np.array(final_preds),
            np.array(final_probs)
        )
        
        print(f"Fold {fold_num} Final Results:")
        print(f"  Accuracy: {fold_metrics['accuracy']:.4f}")
        print(f"  Precision: {fold_metrics['precision']:.4f}")
        print(f"  Recall: {fold_metrics['recall']:.4f}")
        print(f"  F1-Score: {fold_metrics['f1_score']:.4f}")
        if fold_metrics['auc_roc']:
            print(f"  AUC-ROC: {fold_metrics['auc_roc']:.4f}")
        
        return fold_metrics
    
    def _aggregate_fold_results(self, fold_results: List[Dict]) -> Dict:
        """
        Aggregate results across all folds
        
        Args:
            fold_results: List of fold result dictionaries
            
        Returns:
            Aggregated CV results with mean and std
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        
        aggregated = {
            'fold_results': fold_results,
            'mean_metrics': {},
            'std_metrics': {},
            'individual_metrics': {}
        }
        
        for metric in metrics:
            values = []
            for fold_result in fold_results:
                if fold_result.get(metric) is not None:
                    values.append(fold_result[metric])
            
            if values:
                aggregated['mean_metrics'][metric] = np.mean(values)
                aggregated['std_metrics'][metric] = np.std(values)
                aggregated['individual_metrics'][metric] = values
        
        # Print summary
        print(f"\n{'='*60}")
        print("CROSS-VALIDATION SUMMARY")
        print(f"{'='*60}")
        
        for metric in metrics:
            if metric in aggregated['mean_metrics']:
                mean_val = aggregated['mean_metrics'][metric]
                std_val = aggregated['std_metrics'][metric]
                print(f"{metric.replace('_', ' ').title():<15}: "
                      f"{mean_val:.4f} ± {std_val:.4f}")
        
        return aggregated
    
    def save_cv_results(self, cv_results: Dict, save_path: str) -> None:
        """
        Save cross-validation results to file
        
        Args:
            cv_results: CV results dictionary
            save_path: Path to save results
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = copy.deepcopy(cv_results)
        
        for fold_result in serializable_results['fold_results']:
            if 'confusion_matrix' in fold_result:
                fold_result['confusion_matrix'] = fold_result['confusion_matrix'].tolist()
        
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"CV results saved to {save_path}")
    
    def compare_cv_results(self, results1: Dict, results2: Dict, 
                          model1_name: str = "Model 1", 
                          model2_name: str = "Model 2") -> Dict:
        """
        Compare cross-validation results between two models
        
        Args:
            results1: CV results for first model
            results2: CV results for second model
            model1_name: Name of first model
            model2_name: Name of second model
            
        Returns:
            Comparison results including statistical tests
        """
        comparison = {
            'models': [model1_name, model2_name],
            'metrics_comparison': {},
            'statistical_tests': {}
        }
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        
        for metric in metrics:
            if (metric in results1['individual_metrics'] and 
                metric in results2['individual_metrics']):
                
                values1 = results1['individual_metrics'][metric]
                values2 = results2['individual_metrics'][metric]
                
                comparison['metrics_comparison'][metric] = {
                    model1_name: {
                        'mean': np.mean(values1),
                        'std': np.std(values1),
                        'values': values1
                    },
                    model2_name: {
                        'mean': np.mean(values2),
                        'std': np.std(values2),
                        'values': values2
                    }
                }
                
                # Statistical significance test
                stat_test = self.evaluator.statistical_significance_test(
                    values1, values2, test='ttest'
                )
                comparison['statistical_tests'][metric] = stat_test
        
        # Print comparison
        print(f"\n{'='*70}")
        print(f"MODEL COMPARISON: {model1_name} vs {model2_name}")
        print(f"{'='*70}")
        
        for metric in metrics:
            if metric in comparison['metrics_comparison']:
                comp = comparison['metrics_comparison'][metric]
                stat = comparison['statistical_tests'][metric]
                
                print(f"\n{metric.replace('_', ' ').title()}:")
                print(f"  {model1_name}: {comp[model1_name]['mean']:.4f} ± {comp[model1_name]['std']:.4f}")
                print(f"  {model2_name}: {comp[model2_name]['mean']:.4f} ± {comp[model2_name]['std']:.4f}")
                print(f"  Difference: {stat['mean_diff']:.4f}")
                print(f"  P-value: {stat['p_value']:.4f} {'*' if stat['significant'] else ''}")
        
        return comparison

def run_cross_validation_experiment(model_class, dataset, config):
    """
    Convenience function to run a complete CV experiment
    
    Args:
        model_class: Model class to evaluate
        dataset: Dataset to use
        config: Configuration dictionary
        
    Returns:
        CV results
    """
    cv = CrossValidator(n_folds=config.get('n_folds', 5))
    
    results = cv.run_cv_evaluation(
        model_class=model_class,
        dataset=dataset,
        train_params=config['train_params'],
        data_params=config['data_params']
    )
    
    # Save results if path provided
    if 'save_path' in config:
        cv.save_cv_results(results, config['save_path'])
    
    return results