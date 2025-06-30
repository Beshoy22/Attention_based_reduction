#!/usr/bin/env python3
"""
Reconstruction evaluation utilities for integration into training pipeline
"""

import torch
import numpy as np
import pandas as pd
import os
import json
from typing import Dict, List, Tuple, Optional
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt

from .dataloader import MedicalImagingDataset, custom_collate_fn, load_pkl_files
from torch.utils.data import DataLoader
from .models import create_model

class ReconstructionEvaluator:
    """Evaluate reconstruction quality for trained autoencoder models"""
    
    def __init__(self, model, pkl_files: List[str], model_config: Dict, 
                 split_json: Optional[Dict] = None, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_config = model_config
        self.split_json = split_json
        self.model = model
        
        # Load all data
        self.all_data = load_pkl_files(pkl_files)
        
        # Split data based on JSON if provided
        if split_json:
            self.train_data, self.val_data, self.test_data = self._split_data_by_json()
        else:
            # Use all data for evaluation (no train/test distinction)
            self.train_data = self.all_data
            self.val_data = []
            self.test_data = self.all_data
            
        # Statistics for baselines (computed from training data only)
        self.train_stats = None
        
    def _split_data_by_json(self) -> Tuple[List, List, List]:
        """Split data based on train/test split JSON"""
        train_ids = set(self.split_json['TRAIN_SET'])
        test_ids = set(self.split_json['TEST_SET'])
        
        train_data = []
        test_data = []
        val_data = []  # Will be empty for this evaluation
        
        for patient in self.all_data:
            patient_id = patient['patient_id']
            if patient_id in train_ids:
                train_data.append(patient)
            elif patient_id in test_ids:
                test_data.append(patient)
        
        return train_data, val_data, test_data
        
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
        
        all_features = []
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch['features'].to(self.device)
                
                if self.model.reconstruct_all:
                    # For full reconstruction, use original input as target
                    target_features = features  # Shape: (batch, 176, 512)
                else:
                    # For attended reconstruction, use attended features as target
                    target_features = self.model.attention(features)  # Shape: (batch, k, 512)
                
                all_features.append(target_features.cpu().numpy())
        
        # Concatenate all features: (total_train_samples, num_vectors, 512)
        all_features = np.concatenate(all_features, axis=0)
        
        print(f"✅ Statistics computed from {all_features.shape[0]} training samples")
        
        # Compute statistics for each feature dimension
        num_vectors, feat_dim = all_features.shape[1], all_features.shape[2]
        flattened_features = all_features.reshape(-1, num_vectors * feat_dim)  # (total_train_samples, num_vectors*512)
        
        # Compute statistics
        mean_features = np.mean(flattened_features, axis=0)
        median_features = np.median(flattened_features, axis=0)
        
        # Mode computation (most frequent value)
        mode_features = np.zeros_like(mean_features)
        for i in range(flattened_features.shape[1]):
            mode_result = stats.mode(flattened_features[:, i], keepdims=True)
            mode_features[i] = mode_result.mode[0]
        
        # Reshape back to (num_vectors, 512) format
        mean_reshaped = mean_features.reshape(num_vectors, feat_dim)
        median_reshaped = median_features.reshape(num_vectors, feat_dim)
        mode_reshaped = mode_features.reshape(num_vectors, feat_dim)
        
        self.train_stats = {
            'mean': mean_reshaped,
            'median': median_reshaped,
            'mode': mode_reshaped,
            'shape': (num_vectors, feat_dim),
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
    
    def evaluate_set(self, data: List, set_name: str) -> Dict[str, Dict]:
        """Evaluate reconstruction quality on a specific data set"""
        
        if self.train_stats is None:
            self.compute_training_statistics()
        
        if not data:
            print(f"No {set_name} data found!")
            return {}
        
        print(f"🧪 Analyzing reconstruction quality on {len(data)} {set_name} patients...")
        
        # Create dataset
        dataset = MedicalImagingDataset(data, include_missing_endpoints=True,
                                      endpoints=self.model_config.get('endpoints', ['os6', 'os24']))
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
        
        all_inputs = []
        all_reconstructions = []
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch['features'].to(self.device)
                
                # Get appropriate target features
                if self.model.reconstruct_all:
                    target_features = features  # Original input
                else:
                    target_features = self.model.attention(features)  # Attended features
                
                # Get reconstruction
                reconstructed, _ = self.model(features)
                
                all_inputs.append(target_features.cpu().numpy())
                all_reconstructions.append(reconstructed.cpu().numpy())
        
        # Concatenate all data
        inputs = np.concatenate(all_inputs, axis=0)
        reconstructions = np.concatenate(all_reconstructions, axis=0)
        
        print(f"✅ Evaluating reconstruction on {len(inputs)} {set_name} samples")
        
        # Create baseline predictions using TRAINING statistics
        batch_size = inputs.shape[0]
        num_vectors, feat_dim = self.train_stats['shape']
        
        # Expand training statistics to match batch size
        mean_predictions = np.tile(self.train_stats['mean'][np.newaxis, :, :], (batch_size, 1, 1))
        median_predictions = np.tile(self.train_stats['median'][np.newaxis, :, :], (batch_size, 1, 1))
        mode_predictions = np.tile(self.train_stats['mode'][np.newaxis, :, :], (batch_size, 1, 1))
        
        # Compute metrics for each approach
        results = {}
        
        # 1. Model reconstruction
        print(f"Computing metrics for model reconstruction on {set_name}...")
        results['model_reconstruction'] = self.compute_reconstruction_metrics(inputs, reconstructions)
        
        # 2. Mean baseline
        print(f"Computing metrics for mean baseline on {set_name}...")
        results['mean_baseline'] = self.compute_reconstruction_metrics(inputs, mean_predictions)
        
        # 3. Median baseline
        print(f"Computing metrics for median baseline on {set_name}...")
        results['median_baseline'] = self.compute_reconstruction_metrics(inputs, median_predictions)
        
        # 4. Mode baseline
        print(f"Computing metrics for mode baseline on {set_name}...")
        results['mode_baseline'] = self.compute_reconstruction_metrics(inputs, mode_predictions)
        
        return results
    
    def evaluate_all_sets(self) -> Dict[str, Dict[str, Dict]]:
        """Evaluate reconstruction quality on train, validation, and test sets"""
        all_results = {}
        
        # Evaluate train set
        if self.train_data:
            all_results['train'] = self.evaluate_set(self.train_data, 'train')
        
        # Evaluate validation set (if available)
        if self.val_data:
            all_results['validation'] = self.evaluate_set(self.val_data, 'validation')
        
        # Evaluate test set
        if self.test_data:
            all_results['test'] = self.evaluate_set(self.test_data, 'test')
        
        return all_results
    
    def create_comparison_report(self, all_results: Dict[str, Dict[str, Dict]], save_dir: str = None) -> pd.DataFrame:
        """Create a comprehensive comparison report for all sets"""
        
        methods = ['model_reconstruction', 'mean_baseline', 'median_baseline', 'mode_baseline']
        method_names = ['Model Reconstruction', 'Mean Baseline', 'Median Baseline', 'Mode Baseline']
        
        report_data = []
        
        for set_name, results in all_results.items():
            for method, name in zip(methods, method_names):
                if method in results:
                    row = {'Set': set_name.title(), 'Method': name}
                    row.update(results[method])
                    report_data.append(row)
        
        df = pd.DataFrame(report_data)
        
        if save_dir:
            csv_path = os.path.join(save_dir, 'reconstruction_comparison_all_sets.csv')
            df.to_csv(csv_path, index=False)
            print(f"Comprehensive comparison report saved to: {csv_path}")
        
        return df
    
    def save_detailed_results(self, all_results: Dict[str, Dict[str, Dict]], save_dir: str):
        """Save detailed results to JSON"""
        results_path = os.path.join(save_dir, 'detailed_reconstruction_results.json')
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for set_name, set_results in all_results.items():
            json_results[set_name] = {}
            for method, metrics in set_results.items():
                json_results[set_name][method] = {
                    k: float(v) if isinstance(v, np.number) else v 
                    for k, v in metrics.items() if not isinstance(v, np.ndarray)
                }
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Detailed results saved to: {results_path}")


def evaluate_reconstruction_quality(model, pkl_files: List[str], model_config: Dict, 
                                  split_json: Optional[Dict] = None, save_dir: str = None, 
                                  device: str = 'cuda') -> Dict[str, Dict[str, Dict]]:
    """
    Convenience function to evaluate reconstruction quality and save results
    
    Args:
        model: Trained autoencoder model
        pkl_files: List of .pkl files containing patient data
        model_config: Model configuration dictionary
        split_json: Optional train/test split JSON
        save_dir: Directory to save results
        device: Device to use for evaluation
        
    Returns:
        Dictionary containing results for all sets and methods
    """
    
    evaluator = ReconstructionEvaluator(model, pkl_files, model_config, split_json, device)
    
    # Evaluate all sets
    print("\n" + "="*60)
    print("RECONSTRUCTION QUALITY ANALYSIS")
    print("="*60)
    print("Note: Baselines computed from TRAINING data, evaluation on respective sets")
    
    all_results = evaluator.evaluate_all_sets()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Create comparison report
        df = evaluator.create_comparison_report(all_results, save_dir)
        
        print("\nReconstruction Quality Comparison (All Sets):")
        print(df.to_string(index=False))
        
        # Save detailed results
        evaluator.save_detailed_results(all_results, save_dir)
        
        print(f"\nReconstruction analysis complete! Results saved to: {save_dir}")
    
    return all_results