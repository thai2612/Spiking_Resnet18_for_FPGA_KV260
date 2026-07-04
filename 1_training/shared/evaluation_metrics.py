"""
Comprehensive Evaluation Metrics for SNN Crack Detection
Inspired by CrackVision paper methodology
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.preprocessing import label_binarize
from scipy import stats
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveEvaluator:
    """Comprehensive evaluation metrics following CrackVision methodology"""
    
    def __init__(self, class_names: List[str] = None):
        self.class_names = class_names or ['Uncracked', 'Cracked']
        
    def calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                      y_prob: np.ndarray = None) -> Dict:
        """
        Calculate comprehensive metrics including precision, recall, F1, specificity, AUC
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (for AUC calculation)
            
        Returns:
            Dictionary containing all metrics
        """
        results = {}
        
        # Basic metrics
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['precision'] = precision_score(y_true, y_pred, average='weighted')
        results['recall'] = recall_score(y_true, y_pred, average='weighted')
        results['f1_score'] = f1_score(y_true, y_pred, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = cm
        
        # Specificity calculation
        if len(np.unique(y_true)) == 2:  # Binary classification
            tn, fp, fn, tp = cm.ravel()
            results['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            results['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # AUC calculation
        if y_prob is not None:
            try:
                if len(np.unique(y_true)) == 2:  # Binary
                    results['auc_roc'] = roc_auc_score(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
                else:  # Multi-class
                    results['auc_roc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
            except Exception as e:
                print(f"Warning: Could not calculate AUC - {e}")
                results['auc_roc'] = None
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        results['per_class_metrics'] = {
            'precision': dict(zip(self.class_names, precision_per_class)),
            'recall': dict(zip(self.class_names, recall_per_class)),
            'f1_score': dict(zip(self.class_names, f1_per_class))
        }
        
        return results
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: str = None, 
                            title: str = "Confusion Matrix") -> None:
        """
        Plot confusion matrix following CrackVision style
        
        Args:
            cm: Confusion matrix
            save_path: Path to save the plot
            title: Plot title
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    cbar_kws={'label': 'Count'})
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, y_true: np.ndarray, y_prob: np.ndarray, 
                       save_path: str = None, title: str = "ROC Curves") -> Dict:
        """
        Plot ROC curves for binary or multi-class classification
        
        Args:
            y_true: True labels
            y_prob: Prediction probabilities
            save_path: Path to save the plot
            title: Plot title
            
        Returns:
            Dictionary containing AUC values
        """
        plt.figure(figsize=(10, 8))
        auc_scores = {}
        
        if len(np.unique(y_true)) == 2:  # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
            auc_score = roc_auc_score(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
            
            plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc_score:.3f})')
            auc_scores['binary'] = auc_score
            
        else:  # Multi-class classification
            y_true_bin = label_binarize(y_true, classes=range(len(self.class_names)))
            
            for i, class_name in enumerate(self.class_names):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                auc_score = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
                
                plt.plot(fpr, tpr, linewidth=2, 
                        label=f'{class_name} (AUC = {auc_score:.3f})')
                auc_scores[class_name] = auc_score
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return auc_scores
    
    def calculate_multiclass_auc(self, y_true: np.ndarray, y_prob: np.ndarray, 
                               average: str = 'macro') -> float:
        """
        Calculate multi-class AUC following CrackVision equation (8)
        
        Args:
            y_true: True labels
            y_prob: Prediction probabilities
            average: Averaging method ('macro', 'weighted')
            
        Returns:
            Multi-class AUC score
        """
        try:
            return roc_auc_score(y_true, y_prob, multi_class='ovr', average=average)
        except Exception as e:
            print(f"Warning: Could not calculate multi-class AUC - {e}")
            return None
    
    def statistical_significance_test(self, results1: List[float], results2: List[float], 
                                    test: str = 'ttest') -> Dict:
        """
        Perform statistical significance testing between two sets of results
        
        Args:
            results1: First set of results (e.g., SNN accuracies)
            results2: Second set of results (e.g., CNN accuracies)
            test: Statistical test to use ('ttest', 'mannwhitney')
            
        Returns:
            Dictionary containing test results
        """
        results = {
            'mean_diff': np.mean(results1) - np.mean(results2),
            'std1': np.std(results1),
            'std2': np.std(results2)
        }
        
        if test == 'ttest':
            statistic, p_value = stats.ttest_ind(results1, results2)
            results['test'] = 'Independent t-test'
        elif test == 'mannwhitney':
            statistic, p_value = stats.mannwhitneyu(results1, results2, alternative='two-sided')
            results['test'] = 'Mann-Whitney U test'
        else:
            raise ValueError("Test must be 'ttest' or 'mannwhitney'")
        
        results['statistic'] = statistic
        results['p_value'] = p_value
        results['significant'] = p_value < 0.05
        
        return results
    
    def performance_comparison_table(self, results_dict: Dict[str, Dict], 
                                   save_path: str = None) -> pd.DataFrame:
        """
        Create performance comparison table following CrackVision style
        
        Args:
            results_dict: Dictionary of {model_name: metrics_dict}
            save_path: Path to save the table
            
        Returns:
            Pandas DataFrame with comparison results
        """
        comparison_data = []
        
        for model_name, metrics in results_dict.items():
            row = {
                'Model': model_name,
                'Accuracy (%)': f"{metrics.get('accuracy', 0) * 100:.2f}",
                'Precision (%)': f"{metrics.get('precision', 0) * 100:.2f}",
                'Recall (%)': f"{metrics.get('recall', 0) * 100:.2f}",
                'F1-Score (%)': f"{metrics.get('f1_score', 0) * 100:.2f}",
                'AUC-ROC': f"{metrics.get('auc_roc', 0):.3f}" if metrics.get('auc_roc') else 'N/A'
            }
            
            if 'specificity' in metrics:
                row['Specificity (%)'] = f"{metrics.get('specificity', 0) * 100:.2f}"
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"Performance comparison table saved to {save_path}")
        
        return df
    
    def plot_performance_comparison(self, results_dict: Dict[str, Dict], 
                                  metrics: List[str] = None,
                                  save_path: str = None) -> None:
        """
        Create performance comparison bar chart
        
        Args:
            results_dict: Dictionary of {model_name: metrics_dict}
            metrics: List of metrics to plot
            save_path: Path to save the plot
        """
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        models = list(results_dict.keys())
        x = np.arange(len(models))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, metric in enumerate(metrics):
            values = [results_dict[model].get(metric, 0) * 100 for model in models]
            ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title())
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Performance (%)')
        ax.set_title('Performance Comparison Across Models')
        ax.set_xticks(x + width * (len(metrics) - 1) / 2)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_classification_report_table(self, y_true: np.ndarray, y_pred: np.ndarray,
                                           save_path: str = None) -> str:
        """
        Generate detailed classification report following CrackVision format
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the report
            
        Returns:
            Classification report string
        """
        report = classification_report(y_true, y_pred, 
                                     target_names=self.class_names,
                                     digits=4)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Classification report saved to {save_path}")
        
        return report

def create_crackvision_style_table(results_dict: Dict[str, Dict], 
                                 dataset_name: str = "SDNET2018") -> None:
    """
    Create a CrackVision-style performance table for publication
    
    Args:
        results_dict: Dictionary of {model_name: metrics_dict}
        dataset_name: Name of the dataset
    """
    print(f"\n{'='*60}")
    print(f"Performance Comparison on {dataset_name} Dataset")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC-ROC':<10}")
    print(f"{'-'*60}")
    
    for model_name, metrics in results_dict.items():
        acc = f"{metrics.get('accuracy', 0)*100:.2f}%"
        prec = f"{metrics.get('precision', 0)*100:.2f}%"
        rec = f"{metrics.get('recall', 0)*100:.2f}%"
        f1 = f"{metrics.get('f1_score', 0)*100:.2f}%"
        auc = f"{metrics.get('auc_roc', 0):.3f}" if metrics.get('auc_roc') else 'N/A'
        
        print(f"{model_name:<20} {acc:<10} {prec:<10} {rec:<10} {f1:<10} {auc:<10}")
    
    print(f"{'='*60}")