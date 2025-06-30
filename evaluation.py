#!/usr/bin/env python3
"""
Comprehensive model evaluation script
"""

import torch
import numpy as np
import pandas as pd
import json
import argparse
import os
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
from scipy import stats

from utils.visualization import TrainingVisualizer, FeatureVisualizer
from inference import ModelInference

class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def evaluate_predictions(self, predictions_df: pd.DataFrame) -> Dict:
        """
        Evaluate model predictions comprehensively
        
        Args:
            predictions_df: DataFrame with predictions and ground truth
        
        Returns:
            Dictionary with evaluation metrics
        """
        results = {}
        
        # Evaluate OS6
        if 'os6_true' in predictions_df.columns:
            os6_mask = predictions_df['os6_true'].notna()
            if os6_mask.sum() > 0:
                os6_results = self._evaluate_binary_predictions(
                    predictions_df.loc[os6_mask, 'os6_true'].values,
                    predictions_df.loc[os6_mask, 'os6_prediction'].values,
                    predictions_df.loc[os6_mask, 'os6_prediction_binary'].values,
                    'OS6'
                )
                results['os6'] = os6_results
        
        # Evaluate OS24
        if 'os24_true' in predictions_df.columns:
            os24_mask = predictions_df['os24_true'].notna()
            if os24_mask.sum() > 0:
                os24_results = self._evaluate_binary_predictions(
                    predictions_df.loc[os24_mask, 'os24_true'].values,
                    predictions_df.loc[os24_mask, 'os24_prediction'].values,
                    predictions_df.loc[os24_mask, 'os24_prediction_binary'].values,
                    'OS24'
                )
                results['os24'] = os24_results
        
        return results
    
    def _evaluate_binary_predictions(self, y_true: np.ndarray, y_prob: np.ndarray, 
                                   y_pred: np.ndarray, endpoint_name: str) -> Dict:
        """Evaluate binary classification predictions"""
        results = {}
        
        # Basic metrics
        results['auc'] = roc_auc_score(y_true, y_prob)
        results['accuracy'] = accuracy_score(y_true, y_pred)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        results['precision'] = precision
        results['recall'] = recall
        results['f1'] = f1
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = cm.tolist()
        results['specificity'] = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
        results['sensitivity'] = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
        
        # Calibration
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=10, strategy='uniform'
        )
        results['calibration'] = {
            'fraction_of_positives': fraction_of_positives.tolist(),
            'mean_predicted_value': mean_predicted_value.tolist()
        }
        
        # Create detailed plots
        self._plot_detailed_evaluation(y_true, y_prob, y_pred, endpoint_name)
        
        return results
    
    def _plot_detailed_evaluation(self, y_true: np.ndarray, y_prob: np.ndarray, 
                                y_pred: np.ndarray, endpoint_name: str):
        """Create detailed evaluation plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{endpoint_name} Detailed Evaluation', fontsize=16, fontweight='bold')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)
        axes[0, 0].plot(fpr, tpr, linewidth=3, label=f'ROC Curve (AUC = {auc_score:.3f})')
        axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curve')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        axes[0, 1].plot(recall, precision, linewidth=3)
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Curve')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 2],
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        axes[0, 2].set_title('Confusion Matrix')
        axes[0, 2].set_xlabel('Predicted')
        axes[0, 2].set_ylabel('Actual')
        
        # Prediction Distribution
        positive_probs = y_prob[y_true == 1]
        negative_probs = y_prob[y_true == 0]
        axes[1, 0].hist(negative_probs, bins=30, alpha=0.7, label='Negative', density=True)
        axes[1, 0].hist(positive_probs, bins=30, alpha=0.7, label='Positive', density=True)
        axes[1, 0].set_xlabel('Predicted Probability')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Prediction Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Calibration Plot
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=10, strategy='uniform'
        )
        axes[1, 1].plot(mean_predicted_value, fraction_of_positives, 'o-', linewidth=2, markersize=8)
        axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[1, 1].set_xlabel('Mean Predicted Probability')
        axes[1, 1].set_ylabel('Fraction of Positives')
        axes[1, 1].set_title('Calibration Plot')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Threshold Analysis
        thresholds = np.linspace(0, 1, 101)
        accuracies = []
        f1_scores = []
        for threshold in thresholds:
            pred_binary = (y_prob >= threshold).astype(int)
            accuracies.append(accuracy_score(y_true, pred_binary))
            f1_scores.append(precision_recall_fscore_support(y_true, pred_binary, average='binary', zero_division=0)[2])
        
        axes[1, 2].plot(thresholds, accuracies, label='Accuracy', linewidth=2)
        axes[1, 2].plot(thresholds, f1_scores, label='F1 Score', linewidth=2)
        axes[1, 2].axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Default Threshold')
        axes[1, 2].set_xlabel('Threshold')
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].set_title('Threshold Analysis')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{endpoint_name.lower()}_detailed_evaluation.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def evaluate_by_center(self, predictions_df: pd.DataFrame) -> Dict:
        """Evaluate performance by center"""
        center_results = {}
        
        for center in predictions_df['center'].unique():
            center_data = predictions_df[predictions_df['center'] == center]
            center_results[center] = {
                'n_patients': len(center_data),
                'os6_metrics': {},
                'os24_metrics': {}
            }
            
            # OS6 evaluation
            if 'os6_true' in center_data.columns and center_data['os6_true'].notna().sum() > 1:
                os6_mask = center_data['os6_true'].notna()
                if os6_mask.sum() > 0 and len(center_data.loc[os6_mask, 'os6_true'].unique()) > 1:
                    center_results[center]['os6_metrics'] = {
                        'auc': roc_auc_score(center_data.loc[os6_mask, 'os6_true'], 
                                           center_data.loc[os6_mask, 'os6_prediction']),
                        'accuracy': accuracy_score(center_data.loc[os6_mask, 'os6_true'], 
                                                 center_data.loc[os6_mask, 'os6_prediction_binary'])
                    }
            
            # OS24 evaluation
            if 'os24_true' in center_data.columns and center_data['os24_true'].notna().sum() > 1:
                os24_mask = center_data['os24_true'].notna()
                if os24_mask.sum() > 0 and len(center_data.loc[os24_mask, 'os24_true'].unique()) > 1:
                    center_results[center]['os24_metrics'] = {
                        'auc': roc_auc_score(center_data.loc[os24_mask, 'os24_true'], 
                                           center_data.loc[os24_mask, 'os24_prediction']),
                        'accuracy': accuracy_score(center_data.loc[os24_mask, 'os24_true'], 
                                                 center_data.loc[os24_mask, 'os24_prediction_binary'])
                    }
        
        # Create center comparison plot
        self._plot_center_comparison(center_results)
        
        return center_results
    
    def _plot_center_comparison(self, center_results: Dict):
        """Plot performance comparison across centers"""
        centers = list(center_results.keys())
        os6_aucs = [center_results[c]['os6_metrics'].get('auc', np.nan) for c in centers]
        os24_aucs = [center_results[c]['os24_metrics'].get('auc', np.nan) for c in centers]
        n_patients = [center_results[c]['n_patients'] for c in centers]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # AUC comparison
        x = np.arange(len(centers))
        width = 0.35
        
        axes[0].bar(x - width/2, os6_aucs, width, label='OS6 AUC', alpha=0.8)
        axes[0].bar(x + width/2, os24_aucs, width, label='OS24 AUC', alpha=0.8)
        axes[0].set_xlabel('Center')
        axes[0].set_ylabel('AUC')
        axes[0].set_title('AUC by Center')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(centers, rotation=45)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Patient count
        axes[1].bar(centers, n_patients, alpha=0.8, color='green')
        axes[1].set_xlabel('Center')
        axes[1].set_ylabel('Number of Patients')
        axes[1].set_title('Patient Count by Center')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        # Scatter plot: AUC vs patient count
        valid_mask = ~np.isnan(os6_aucs)
        if valid_mask.sum() > 0:
            axes[2].scatter([n_patients[i] for i in range(len(centers)) if valid_mask[i]], 
                          [os6_aucs[i] for i in range(len(centers)) if valid_mask[i]], 
                          label='OS6', alpha=0.7, s=100)
        
        valid_mask = ~np.isnan(os24_aucs)
        if valid_mask.sum() > 0:
            axes[2].scatter([n_patients[i] for i in range(len(centers)) if valid_mask[i]], 
                          [os24_aucs[i] for i in range(len(centers)) if valid_mask[i]], 
                          label='OS24', alpha=0.7, s=100)
        
        axes[2].set_xlabel('Number of Patients')
        axes[2].set_ylabel('AUC')
        axes[2].set_title('AUC vs Patient Count')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'center_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def compare_models(self, model_results: List[Dict], model_names: List[str]):
        """Compare multiple models"""
        comparison_data = []
        
        for i, (results, name) in enumerate(zip(model_results, model_names)):
            for endpoint in ['os6', 'os24']:
                if endpoint in results:
                    comparison_data.append({
                        'model': name,
                        'endpoint': endpoint.upper(),
                        'auc': results[endpoint]['auc'],
                        'accuracy': results[endpoint]['accuracy'],
                        'f1': results[endpoint]['f1'],
                        'precision': results[endpoint]['precision'],
                        'recall': results[endpoint]['recall']
                    })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # AUC comparison
        sns.barplot(data=comparison_df, x='model', y='auc', hue='endpoint', ax=axes[0, 0])
        axes[0, 0].set_title('AUC Comparison')
        axes[0, 0].set_ylim(0, 1)
        
        # Accuracy comparison
        sns.barplot(data=comparison_df, x='model', y='accuracy', hue='endpoint', ax=axes[0, 1])
        axes[0, 1].set_title('Accuracy Comparison')
        axes[0, 1].set_ylim(0, 1)
        
        # F1 Score comparison
        sns.barplot(data=comparison_df, x='model', y='f1', hue='endpoint', ax=axes[1, 0])
        axes[1, 0].set_title('F1 Score Comparison')
        axes[1, 0].set_ylim(0, 1)
        
        # Precision vs Recall
        sns.scatterplot(data=comparison_df, x='recall', y='precision', 
                       hue='endpoint', style='model', s=100, ax=axes[1, 1])
        axes[1, 1].set_title('Precision vs Recall')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'model_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        return comparison_df

def main():
    parser = argparse.ArgumentParser(description='Model Evaluation Script')
    
    parser.add_argument('--predictions_file', type=str, required=True,
                        help='Path to predictions CSV file')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--model_name', type=str, default='model',
                        help='Name of the model being evaluated')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load predictions
    print("Loading predictions...")
    predictions_df = pd.read_csv(args.predictions_file)
    print(f"Loaded predictions for {len(predictions_df)} patients")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.output_dir)
    
    # Evaluate overall performance
    print("Evaluating overall performance...")
    overall_results = evaluator.evaluate_predictions(predictions_df)
    
    # Evaluate by center
    print("Evaluating performance by center...")
    center_results = evaluator.evaluate_by_center(predictions_df)
    
    # Save results
    results = {
        'model_name': args.model_name,
        'overall_results': overall_results,
        'center_results': center_results,
        'summary': {
            'total_patients': len(predictions_df),
            'centers': predictions_df['center'].unique().tolist(),
            'os6_available': predictions_df['os6_true'].notna().sum() if 'os6_true' in predictions_df.columns else 0,
            'os24_available': predictions_df['os24_true'].notna().sum() if 'os24_true' in predictions_df.columns else 0
        }
    }
    
    with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    
    # Print summary
    print(f"\nEvaluation Results for {args.model_name}:")
    if 'os6' in overall_results:
        print(f"OS6 - AUC: {overall_results['os6']['auc']:.3f}, "
              f"Accuracy: {overall_results['os6']['accuracy']:.3f}, "
              f"F1: {overall_results['os6']['f1']:.3f}")
    if 'os24' in overall_results:
        print(f"OS24 - AUC: {overall_results['os24']['auc']:.3f}, "
              f"Accuracy: {overall_results['os24']['accuracy']:.3f}, "
              f"F1: {overall_results['os24']['f1']:.3f}")
    
    print(f"\nResults saved to: {args.output_dir}")

if __name__ == "__main__":
    main()