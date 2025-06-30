import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import json
import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TrainingVisualizer:
    """Visualize training progress and results"""
    
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_training_curves(self, history: Dict, save_name: str = 'training_curves.png'):
        """Plot training and validation loss curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total loss
        epochs = range(1, len(history['train_loss']) + 1)
        train_total = [x['total'] for x in history['train_loss']]
        val_total = [x['total'] for x in history['val_loss']]
        
        axes[0, 0].plot(epochs, train_total, 'b-', label='Training', linewidth=2)
        axes[0, 0].plot(epochs, val_total, 'r-', label='Validation', linewidth=2)
        axes[0, 0].set_title('Total Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Learning rate
        if 'learning_rate' in history:
            axes[0, 1].plot(epochs, history['learning_rate'], 'g-', linewidth=2)
            axes[0, 1].set_title('Learning Rate', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Validation metrics (OS6 AUC)
        if history['val_metrics'] and len(history['val_metrics']) > 0:
            os6_aucs = []
            os24_aucs = []
            
            for metrics in history['val_metrics']:
                if 'os6' in metrics:
                    os6_aucs.append(metrics['os6']['auc'])
                else:
                    os6_aucs.append(np.nan)
                
                if 'os24' in metrics:
                    os24_aucs.append(metrics['os24']['auc'])
                else:
                    os24_aucs.append(np.nan)
            
            if not all(np.isnan(os6_aucs)):
                axes[1, 0].plot(epochs, os6_aucs, 'purple', label='OS6 AUC', linewidth=2)
            if not all(np.isnan(os24_aucs)):
                axes[1, 0].plot(epochs, os24_aucs, 'orange', label='OS24 AUC', linewidth=2)
            
            axes[1, 0].set_title('Validation AUC', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('AUC')
            axes[1, 0].set_ylim(0, 1)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Component losses (if available)
        if 'reconstruction' in history['train_loss'][0]:
            train_recon = [x.get('reconstruction', 0) for x in history['train_loss']]
            train_pred = [x.get('prediction', 0) for x in history['train_loss']]
            
            axes[1, 1].plot(epochs, train_recon, 'cyan', label='Reconstruction', linewidth=2)
            axes[1, 1].plot(epochs, train_pred, 'magenta', label='Prediction', linewidth=2)
            axes[1, 1].set_title('Training Loss Components', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curves(self, y_true: Dict, y_prob: Dict, save_name: str = 'roc_curves.png'):
        """Plot ROC curves for OS6 and OS24"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for i, (endpoint, ax) in enumerate(zip(['os6', 'os24'], axes)):
            if endpoint in y_true and endpoint in y_prob:
                fpr, tpr, _ = roc_curve(y_true[endpoint], y_prob[endpoint])
                roc_auc = auc(fpr, tpr)
                
                ax.plot(fpr, tpr, linewidth=3, 
                       label=f'ROC Curve (AUC = {roc_auc:.3f})')
                ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate', fontsize=12)
                ax.set_ylabel('True Positive Rate', fontsize=12)
                ax.set_title(f'{endpoint.upper()} ROC Curve', fontsize=14, fontweight='bold')
                ax.legend(loc="lower right")
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrices(self, y_true: Dict, y_pred: Dict, save_name: str = 'confusion_matrices.png'):
        """Plot confusion matrices for OS6 and OS24"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for i, (endpoint, ax) in enumerate(zip(['os6', 'os24'], axes)):
            if endpoint in y_true and endpoint in y_pred:
                cm = confusion_matrix(y_true[endpoint], y_pred[endpoint])
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                           xticklabels=['Negative', 'Positive'],
                           yticklabels=['Negative', 'Positive'])
                ax.set_title(f'{endpoint.upper()} Confusion Matrix', fontsize=14, fontweight='bold')
                ax.set_xlabel('Predicted', fontsize=12)
                ax.set_ylabel('Actual', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()

class AttentionVisualizer:
    """Visualize attention weights and patterns"""
    
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_attention_weights(self, model: torch.nn.Module, save_name: str = 'attention_weights.png'):
        """Plot attention weight distribution"""
        model.eval()
        
        # Extract attention weights
        attention_weights = model.attention.attention_weights.data.cpu().numpy()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Heatmap of all attention weights
        im = axes[0, 0].imshow(attention_weights, cmap='viridis', aspect='auto')
        axes[0, 0].set_title('Attention Weights Heatmap', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Patches within Group')
        axes[0, 0].set_ylabel('Attention Group')
        plt.colorbar(im, ax=axes[0, 0])
        
        # Distribution of attention weights
        axes[0, 1].hist(attention_weights.flatten(), bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Attention Weight Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Attention Weight')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Average attention per group
        avg_attention = np.mean(attention_weights, axis=1)
        axes[1, 0].bar(range(len(avg_attention)), avg_attention, alpha=0.7)
        axes[1, 0].set_title('Average Attention per Group', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Attention Group')
        axes[1, 0].set_ylabel('Average Attention Weight')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Attention weight variance per group
        var_attention = np.var(attention_weights, axis=1)
        axes[1, 1].bar(range(len(var_attention)), var_attention, alpha=0.7, color='orange')
        axes[1, 1].set_title('Attention Weight Variance per Group', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Attention Group')
        axes[1, 1].set_ylabel('Attention Weight Variance')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_attended_features_distribution(self, attended_features: np.ndarray, 
                                          save_name: str = 'attended_features_dist.png'):
        """Plot distribution of attended features"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Feature magnitude distribution
        feature_magnitudes = np.linalg.norm(attended_features, axis=-1)
        axes[0, 0].hist(feature_magnitudes.flatten(), bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Attended Feature Magnitudes', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('L2 Norm')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Feature variance across samples
        feature_variance = np.var(attended_features, axis=0)
        avg_variance_per_group = np.mean(feature_variance, axis=1)
        axes[0, 1].bar(range(len(avg_variance_per_group)), avg_variance_per_group, alpha=0.7)
        axes[0, 1].set_title('Feature Variance per Attention Group', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Attention Group')
        axes[0, 1].set_ylabel('Average Feature Variance')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Feature correlation heatmap (first 20 features for readability)
        if attended_features.shape[1] >= 20:
            sample_indices = np.random.choice(attended_features.shape[1], 20, replace=False)
            sample_features = attended_features[:, sample_indices, :].reshape(-1, attended_features.shape[-1])
            correlation_matrix = np.corrcoef(sample_features.T)
            
            im = axes[1, 0].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            axes[1, 0].set_title('Feature Correlation Matrix (Sample)', fontsize=14, fontweight='bold')
            plt.colorbar(im, ax=axes[1, 0])
        
        # Average feature values per group
        avg_features_per_group = np.mean(np.mean(attended_features, axis=-1), axis=0)
        axes[1, 1].bar(range(len(avg_features_per_group)), avg_features_per_group, alpha=0.7, color='green')
        axes[1, 1].set_title('Average Feature Value per Group', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Attention Group')
        axes[1, 1].set_ylabel('Average Feature Value')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()

class FeatureVisualizer:
    """Visualize learned features and embeddings"""
    
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_feature_tsne(self, features: np.ndarray, labels: Dict, 
                          save_name: str = 'feature_tsne.png', perplexity: int = 30):
        """Plot t-SNE visualization of features"""
        # Perform t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        features_2d = tsne.fit_transform(features)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Color by OS6
        if 'os6' in labels:
            scatter = axes[0].scatter(features_2d[:, 0], features_2d[:, 1], 
                                    c=labels['os6'], cmap='viridis', alpha=0.7)
            axes[0].set_title('t-SNE colored by OS6', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('t-SNE 1')
            axes[0].set_ylabel('t-SNE 2')
            plt.colorbar(scatter, ax=axes[0])
        
        # Color by OS24
        if 'os24' in labels:
            scatter = axes[1].scatter(features_2d[:, 0], features_2d[:, 1], 
                                    c=labels['os24'], cmap='plasma', alpha=0.7)
            axes[1].set_title('t-SNE colored by OS24', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('t-SNE 1')
            axes[1].set_ylabel('t-SNE 2')
            plt.colorbar(scatter, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_pca(self, features: np.ndarray, labels: Dict, 
                         save_name: str = 'feature_pca.png'):
        """Plot PCA visualization of features"""
        # Perform PCA
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Color by OS6
        if 'os6' in labels:
            scatter = axes[0].scatter(features_2d[:, 0], features_2d[:, 1], 
                                    c=labels['os6'], cmap='viridis', alpha=0.7)
            axes[0].set_title(f'PCA colored by OS6\n(Explained variance: {pca.explained_variance_ratio_.sum():.3f})', 
                            fontsize=14, fontweight='bold')
            axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
            axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
            plt.colorbar(scatter, ax=axes[0])
        
        # Color by OS24
        if 'os24' in labels:
            scatter = axes[1].scatter(features_2d[:, 0], features_2d[:, 1], 
                                    c=labels['os24'], cmap='plasma', alpha=0.7)
            axes[1].set_title(f'PCA colored by OS24\n(Explained variance: {pca.explained_variance_ratio_.sum():.3f})', 
                            fontsize=14, fontweight='bold')
            axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
            axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
            plt.colorbar(scatter, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()

def load_and_visualize_results(results_dir: str):
    """Load training results and create all visualizations"""
    
    # Load training history
    history_path = os.path.join(results_dir, 'training_history.json')
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        # Create training visualizer
        viz = TrainingVisualizer(results_dir)
        viz.plot_training_curves(history)
        print(f"Training curves saved to {results_dir}")
    
    # Load test results
    test_results_path = os.path.join(results_dir, 'test_results.json')
    if os.path.exists(test_results_path):
        with open(test_results_path, 'r') as f:
            test_results = json.load(f)
        print(f"Test results loaded from {results_dir}")
    
    print("Visualization complete!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
        load_and_visualize_results(results_dir)
    else:
        print("Usage: python visualization.py <results_directory>")