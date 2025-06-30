#!/usr/bin/env python3
"""
Analyze reconstruction quality of autoencoder by comparing against baseline predictions
Modified to use train/test split JSON and compute baselines only on training data
"""

import torch
import numpy as np
import pandas as pd
import argparse
import os
import json
from typing import Dict, List, Tuple
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from utils.models import create_model
from utils.dataloader import MedicalImagingDataset, custom_collate_fn, load_pkl_files
from torch.utils.data import DataLoader
from config import load_train_test_split

class ReconstructionAnalyzer:
    """Analyze reconstruction quality with multiple metrics and baselines computed from training data only"""
    
    def __init__(self, model_path: str, pkl_files: List[str], model_config: Dict, 
                 split_json: Dict, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_config = model_config
        self.split_json = split_json
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Load all data
        self.all_data = load_pkl_files(pkl_files)
        
        # Split data based on JSON
        self.train_data, self.test_data = self._split_data_by_json()
        
        # Statistics for baselines (computed from training data only)
        self.train_stats = None
        
    def _load_model(self, model_path: str):
        """Load trained autoencoder model"""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        model = create_model(
            model_type='autoencoder',
            **self.model_config
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model
    
    def _split_data_by_json(self) -> Tuple[List, List]:
        """Split data based on train/test split JSON"""
        train_ids = set(self.split_json['TRAIN_SET'])
        test_ids = set(self.split_json['TEST_SET'])
        
        # Get patient IDs from actual PKL data
        available_patient_ids = {patient['patient_id'] for patient in self.all_data}
        
        train_data = []
        test_data = []
        
        for patient in self.all_data:
            patient_id = patient['patient_id']
            if patient_id in train_ids:
                train_data.append(patient)
            elif patient_id in test_ids:
                test_data.append(patient)
            # Skip patients not in either set
        
        # Calculate coverage statistics
        json_train_available = len(train_ids & available_patient_ids)
        json_test_available = len(test_ids & available_patient_ids)
        json_train_missing = len(train_ids - available_patient_ids)
        json_test_missing = len(test_ids - available_patient_ids)
        
        print(f"📊 DATA AVAILABILITY REPORT:")
        print(f"   PKL files contain: {len(self.all_data)} patients total")
        print(f"   JSON train set: {len(train_ids)} patients ({json_train_available} available, {json_train_missing} missing)")
        print(f"   JSON test set: {len(test_ids)} patients ({json_test_available} available, {json_test_missing} missing)")
        print(f"   ✅ ACTUAL SPLIT: {len(train_data)} train, {len(test_data)} test patients")
        
        if json_train_missing > 0 or json_test_missing > 0:
            print(f"   ⚠️  Some patients in JSON don't have embeddings in PKL files")
        
        return train_data, test_data
    
    def compute_training_statistics(self):
        """Compute mean, median, mode for each feature dimension from training data ONLY"""
        print("Computing baseline statistics from TRAINING data only...")
        
        if not self.train_data:
            raise ValueError("No training data found! Check your train/test split JSON.")
        
        print(f"Computing statistics from {len(self.train_data)} training patients...")
        
        # Create dataset from training data only
        dataset = MedicalImagingDataset(self.train_data, include_missing_endpoints=True, 
                                      endpoints=self.model_config.get('endpoints', ['os6', 'os24']))
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
        
        all_attended_features = []
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch['features'].to(self.device)
                attended_features = self.model.attention(features)  # Shape: (batch, k, 512)
                all_attended_features.append(attended_features.cpu().numpy())
        
        # Concatenate all features: (total_train_samples, k, 512)
        all_features = np.concatenate(all_attended_features, axis=0)
        
        print(f"✅ Statistics computed from {all_features.shape[0]} training samples")
        
        # Compute statistics for each of the k*512 feature dimensions
        k, feat_dim = all_features.shape[1], all_features.shape[2]
        flattened_features = all_features.reshape(-1, k * feat_dim)  # (total_train_samples, k*512)
        
        # Compute statistics
        mean_features = np.mean(flattened_features, axis=0)  # (k*512,)
        median_features = np.median(flattened_features, axis=0)  # (k*512,)
        
        # Mode computation (most frequent value)
        mode_features = np.zeros_like(mean_features)
        for i in range(flattened_features.shape[1]):
            mode_result = stats.mode(flattened_features[:, i], keepdims=True)
            mode_features[i] = mode_result.mode[0]
        
        # Reshape back to (k, 512) format
        mean_reshaped = mean_features.reshape(k, feat_dim)
        median_reshaped = median_features.reshape(k, feat_dim)
        mode_reshaped = mode_features.reshape(k, feat_dim)
        
        self.train_stats = {
            'mean': mean_reshaped,
            'median': median_reshaped,
            'mode': mode_reshaped,
            'shape': (k, feat_dim),
            'n_train_samples': all_features.shape[0]
        }
        
    def compute_reconstruction_metrics(self, true_features: np.ndarray, pred_features: np.ndarray) -> Dict:
        """Compute reconstruction quality metrics"""
        
        # Flatten for easier computation
        input_flat = true_features.reshape(true_features.shape[0], -1)
        pred_flat = pred_features.reshape(pred_features.shape[0], -1)
        
        metrics = {}
        
        # 1. Cosine Similarity (average across samples)
        similarities = []
        for i in range(input_flat.shape[0]):
            sim = np.dot(input_flat[i], pred_flat[i]) / (np.linalg.norm(input_flat[i]) * np.linalg.norm(pred_flat[i]) + 1e-8)
            similarities.append(sim)
        metrics['cosine_similarity'] = np.mean(similarities)
        metrics['cosine_similarity_std'] = np.std(similarities)
        
        # 2. MSE
        mse = np.mean((input_flat - pred_flat) ** 2)
        metrics['mse'] = mse
        
        # 3. RMSE
        metrics['rmse'] = np.sqrt(mse)
        
        # 4. MAE
        metrics['mae'] = np.mean(np.abs(input_flat - pred_flat))
        
        # 5. Euclidean Distance (average across samples)
        distances = []
        for i in range(input_flat.shape[0]):
            dist = np.linalg.norm(input_flat[i] - pred_flat[i])
            distances.append(dist)
        metrics['euclidean_distance'] = np.mean(distances)
        metrics['euclidean_distance_std'] = np.std(distances)
        
        # 6. Pearson Correlation (average across features)
        correlations = []
        for i in range(input_flat.shape[1]):
            if np.std(input_flat[:, i]) > 1e-8 and np.std(pred_flat[:, i]) > 1e-8:
                corr = np.corrcoef(input_flat[:, i], pred_flat[:, i])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        metrics['pearson_correlation'] = np.mean(correlations) if correlations else 0.0
        metrics['pearson_correlation_std'] = np.std(correlations) if correlations else 0.0
        
        # 7. Explained Variance
        total_variance = np.var(input_flat)
        residual_variance = np.var(input_flat - pred_flat)
        metrics['explained_variance'] = 1 - (residual_variance / total_variance) if total_variance > 0 else 0.0
        
        return metrics
    
    def analyze_reconstruction_quality(self) -> Dict[str, Dict]:
        """Analyze reconstruction quality against different baselines using TEST data only"""
        
        if self.train_stats is None:
            self.compute_training_statistics()
        
        if not self.test_data:
            raise ValueError("No test data found! Check your train/test split JSON.")
        
        print(f"🧪 Analyzing reconstruction quality on {len(self.test_data)} TEST patients...")
        
        # Create dataset from test data only
        dataset = MedicalImagingDataset(self.test_data, include_missing_endpoints=True,
                                      endpoints=self.model_config.get('endpoints', ['os6', 'os24']))
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
        
        all_inputs = []
        all_reconstructions = []
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch['features'].to(self.device)
                
                # Get attended features (input to reconstruction)
                attended_features = self.model.attention(features)
                
                # Get reconstruction
                reconstructed, _ = self.model(features)
                
                all_inputs.append(attended_features.cpu().numpy())
                all_reconstructions.append(reconstructed.cpu().numpy())
        
        # Concatenate all test data
        inputs = np.concatenate(all_inputs, axis=0)  # (test_samples, k, 512)
        reconstructions = np.concatenate(all_reconstructions, axis=0)
        
        print(f"✅ Evaluating reconstruction on {len(inputs)} test samples")
        
        # Create baseline predictions using TRAINING statistics
        batch_size = inputs.shape[0]
        k, feat_dim = self.train_stats['shape']
        
        # Expand training statistics to match test batch size
        mean_predictions = np.tile(self.train_stats['mean'][np.newaxis, :, :], (batch_size, 1, 1))
        median_predictions = np.tile(self.train_stats['median'][np.newaxis, :, :], (batch_size, 1, 1))
        mode_predictions = np.tile(self.train_stats['mode'][np.newaxis, :, :], (batch_size, 1, 1))
        
        # Compute metrics for each approach
        results = {}
        
        # 1. Model reconstruction
        print("Computing metrics for model reconstruction...")
        results['model_reconstruction'] = self.compute_reconstruction_metrics(inputs, reconstructions)
        
        # 2. Mean baseline (computed from training data)
        print("Computing metrics for mean baseline...")
        results['mean_baseline'] = self.compute_reconstruction_metrics(inputs, mean_predictions)
        
        # 3. Median baseline (computed from training data)
        print("Computing metrics for median baseline...")
        results['median_baseline'] = self.compute_reconstruction_metrics(inputs, median_predictions)
        
        # 4. Mode baseline (computed from training data)
        print("Computing metrics for mode baseline...")
        results['mode_baseline'] = self.compute_reconstruction_metrics(inputs, mode_predictions)
        
        return results
    
    def create_comparison_report(self, results: Dict[str, Dict], save_path: str = None) -> pd.DataFrame:
        """Create a comparison report and save to CSV"""
        
        methods = ['model_reconstruction', 'mean_baseline', 'median_baseline', 'mode_baseline']
        method_names = ['Model Reconstruction', 'Mean Baseline', 'Median Baseline', 'Mode Baseline']
        
        report_data = []
        
        for method, name in zip(methods, method_names):
            if method in results:
                row = {'Method': name}
                row.update(results[method])
                report_data.append(row)
        
        df = pd.DataFrame(report_data)
        
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"Comparison report saved to: {save_path}")
        
        return df
    
    def plot_metrics_comparison(self, results: Dict[str, Dict], save_dir: str = None):
        """Plot comparison of different metrics"""
        
        methods = ['model_reconstruction', 'mean_baseline', 'median_baseline', 'mode_baseline']
        method_names = ['Model', 'Mean', 'Median', 'Mode']
        
        # Key metrics to visualize
        key_metrics = ['cosine_similarity', 'mse', 'pearson_correlation', 'explained_variance', 
                      'euclidean_distance', 'mae']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(key_metrics):
            if i >= len(axes):
                break
                
            values = []
            labels = []
            
            for method, name in zip(methods, method_names):
                if method in results and metric in results[method]:
                    values.append(results[method][metric])
                    labels.append(name)
            
            if values:
                bars = axes[i].bar(labels, values, alpha=0.7)
                axes[i].set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
                
                # Highlight best performance
                if metric in ['cosine_similarity', 'pearson_correlation', 'explained_variance']:
                    best_idx = np.argmax(values)  # Higher is better
                else:
                    best_idx = np.argmin(values)  # Lower is better
                
                bars[best_idx].set_color('green')
                bars[best_idx].set_alpha(0.9)
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
            print(f"Metrics comparison plot saved to: {os.path.join(save_dir, 'metrics_comparison.png')}")
        
        plt.close()  # Close to free memory
    
    def analyze_feature_wise_performance(self) -> Dict:
        """Analyze reconstruction quality per feature dimension on test data"""
        
        if self.train_stats is None:
            self.compute_training_statistics()
        
        print("Analyzing feature-wise reconstruction performance on test data...")
        
        dataset = MedicalImagingDataset(self.test_data, include_missing_endpoints=True,
                                      endpoints=self.model_config.get('endpoints', ['os6', 'os24']))
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
        
        all_inputs = []
        all_reconstructions = []
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch['features'].to(self.device)
                attended_features = self.model.attention(features)
                reconstructed, _ = self.model(features)
                
                all_inputs.append(attended_features.cpu().numpy())
                all_reconstructions.append(reconstructed.cpu().numpy())
        
        inputs = np.concatenate(all_inputs, axis=0)
        reconstructions = np.concatenate(all_reconstructions, axis=0)
        
        # Reshape to (samples, features)
        inputs_flat = inputs.reshape(inputs.shape[0], -1)
        recon_flat = reconstructions.reshape(reconstructions.shape[0], -1)
        
        # Compute per-feature metrics
        feature_mse = np.mean((inputs_flat - recon_flat) ** 2, axis=0)
        feature_correlations = []
        
        for i in range(inputs_flat.shape[1]):
            if np.std(inputs_flat[:, i]) > 1e-8 and np.std(recon_flat[:, i]) > 1e-8:
                corr = np.corrcoef(inputs_flat[:, i], recon_flat[:, i])[0, 1]
                if not np.isnan(corr):
                    feature_correlations.append(corr)
                else:
                    feature_correlations.append(0.0)
            else:
                feature_correlations.append(0.0)
        
        feature_correlations = np.array(feature_correlations)
        
        return {
            'mean_mse': np.mean(feature_mse),
            'std_mse': np.std(feature_mse),
            'mean_correlation': np.mean(feature_correlations),
            'std_correlation': np.std(feature_correlations),
            'feature_mse': feature_mse,
            'feature_correlations': feature_correlations
        }

def main():
    parser = argparse.ArgumentParser(description='Analyze autoencoder reconstruction quality')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--pkl_files', nargs='+', required=True,
                       help='Paths to pickle files containing patient data')
    parser.add_argument('--train_test_split_json', type=str, required=True,
                       help='Path to JSON file containing train/test split')
    parser.add_argument('--attention_k', type=int, default=16,
                       help='Number of attention features (k)')
    parser.add_argument('--latent_dim', type=int, default=64,
                       help='Latent dimension for autoencoder')
    parser.add_argument('--encoder_layers', nargs='+', type=int, default=[128, 64],
                       help='Hidden layer dimensions for encoder')
    parser.add_argument('--predictor_layers', nargs='+', type=int, default=[32, 16],
                       help='Hidden layer dimensions for predictor')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--endpoints', nargs='+', default=['os6', 'os24'],
                       help='Endpoints to predict')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--output_dir', type=str, default='./reconstruction_analysis',
                       help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load train/test split
    print("Loading train/test split...")
    split_data = load_train_test_split(args.train_test_split_json)
    print(f"JSON contains: {len(split_data['TRAIN_SET'])} train + {len(split_data['TEST_SET'])} test IDs")
    
    # Model configuration
    model_config = {
        'attention_k': args.attention_k,
        'latent_dim': args.latent_dim,
        'encoder_layers': args.encoder_layers,
        'predictor_layers': args.predictor_layers,
        'dropout_rate': args.dropout_rate,
        'endpoints': args.endpoints
    }
    
    print("Initializing analyzer (this will show actual data availability)...")
    analyzer = ReconstructionAnalyzer(args.model_path, args.pkl_files, model_config, split_data, args.device)
    
    # Analyze reconstruction quality
    print("\n" + "="*60)
    print("RECONSTRUCTION QUALITY ANALYSIS")
    print("="*60)
    print("Note: Baselines computed from TRAINING data, evaluation on TEST data only")
    
    results = analyzer.analyze_reconstruction_quality()
    
    # Create comparison report
    report_path = os.path.join(args.output_dir, 'reconstruction_comparison.csv')
    df = analyzer.create_comparison_report(results, report_path)
    
    print("\nReconstruction Quality Comparison:")
    print(df.to_string(index=False))
    
    # Create visualizations
    analyzer.plot_metrics_comparison(results, args.output_dir)
    
    # Feature-wise analysis
    print("\n" + "="*60)
    print("FEATURE-WISE ANALYSIS (on test data)")
    print("="*60)
    
    feature_analysis = analyzer.analyze_feature_wise_performance()
    
    print(f"Mean MSE per feature: {feature_analysis['mean_mse']:.6f} ± {feature_analysis['std_mse']:.6f}")
    print(f"Mean correlation per feature: {feature_analysis['mean_correlation']:.4f} ± {feature_analysis['std_correlation']:.4f}")
    
    # Save detailed results
    results_path = os.path.join(args.output_dir, 'detailed_results.json')
    with open(results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in {**results, **feature_analysis}.items():
            if isinstance(value, dict):
                json_results[key] = {k: float(v) if isinstance(v, np.number) else v 
                                   for k, v in value.items() if not isinstance(v, np.ndarray)}
            elif isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            else:
                json_results[key] = float(value) if isinstance(value, np.number) else value
        
        json.dump(json_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_path}")
    print(f"Analysis complete! Results saved to: {args.output_dir}")
    
    # Print summary of methodology
    print("\n" + "="*60)
    print("METHODOLOGY SUMMARY")
    print("="*60)
    print("✅ Baselines (mean, median, mode) computed from TRAINING data only")
    print("✅ Model evaluation performed on TEST data only")
    print("✅ No data leakage between train and test sets")
    print("✅ Rigorous evaluation methodology following best practices")

if __name__ == "__main__":
    main()