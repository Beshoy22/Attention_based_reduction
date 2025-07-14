#!/usr/bin/env python3
"""
Grid search script for hyperparameter optimization
"""

import torch
import numpy as np
import itertools
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd

from config import Config, load_train_test_split
from utils.dataloader import create_data_loaders
from utils.models import create_model
from utils.losses import CombinedLoss, EndToEndLoss
from utils.optimizers import OptimizerManager
from utils.train_loop import train_model, evaluate_model

def set_seed(seed: int):
    """Set random seeds"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class GridSearch:
    def __init__(self, base_args, param_grid: Dict, save_dir: str):
        self.base_args = base_args
        self.param_grid = param_grid
        self.save_dir = save_dir
        self.results = []
        os.makedirs(save_dir, exist_ok=True)
    
    def _get_param_combinations(self) -> List[Dict]:
        """Generate all parameter combinations"""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combinations = []
        
        for combo in itertools.product(*values):
            param_dict = dict(zip(keys, combo))
            combinations.append(param_dict)
        
        return combinations
    
    def _compute_score(self, val_metrics: Dict, val_loss: float, endpoints: List[str]) -> float:
        """Compute combined score: average AUC - normalized val_loss"""
        auc_scores = []
        for endpoint in endpoints:
            if endpoint in val_metrics:
                auc_scores.append(val_metrics[endpoint]['auc'])
        
        avg_auc = np.mean(auc_scores) if auc_scores else 0.0
        # Normalize loss (assuming typical range 0-2, invert to make higher better)
        normalized_loss = max(0, 2 - val_loss) / 2
        
        # Weighted combination: 70% AUC, 30% loss
        return 0.7 * avg_auc + 0.3 * normalized_loss
    
    def run_search(self, max_epochs: int = 50) -> pd.DataFrame:
        """Run grid search"""
        combinations = self._get_param_combinations()
        print(f"Starting grid search with {len(combinations)} combinations")
        
        device = torch.device(self.base_args.device if torch.cuda.is_available() else 'cpu')
        split_data = None
        if self.base_args.train_test_split_json:
            split_data = load_train_test_split(self.base_args.train_test_split_json)
        
        for i, params in enumerate(combinations):
            print(f"\n=== Combination {i+1}/{len(combinations)} ===")
            print(f"Parameters: {params}")
            
            set_seed(42)  # Consistent seed for fair comparison
            
            try:
                # Create data loaders
                train_loader, val_loader, test_loader, class_weights = create_data_loaders(
                    pkl_files=self.base_args.pkl_files,
                    split_json=split_data,
                    val_split=self.base_args.val_split,
                    batch_size=params.get('batch_size', self.base_args.batch_size),
                    num_workers=0,
                    model_type='autoencoder',
                    endpoints=self.base_args.endpoints,
                    random_state=42
                )
                
                # Create model
                model = create_model(
                    model_type='autoencoder',
                    endpoints=self.base_args.endpoints,
                    attention_k=params.get('attention_k', self.base_args.attention_k),
                    encoder_layers=params.get('encoder_layers', self.base_args.encoder_layers),
                    latent_dim=params.get('latent_dim', self.base_args.latent_dim),
                    predictor_layers=params.get('predictor_layers', self.base_args.predictor_layers),
                    dropout_rate=params.get('dropout_rate', self.base_args.dropout_rate)
                ).to(device)
                
                # Create loss and optimizer
                if True:
                    criterion = CombinedLoss(
                        class_weights=class_weights,
                        reconstruction_weight=params.get('reconstruction_weight', self.base_args.reconstruction_weight),
                        prediction_weight=params.get('prediction_weight', self.base_args.prediction_weight),
                        endpoints=self.base_args.endpoints,
                        device=device
                    )
                else:
                    criterion = EndToEndLoss(class_weights=class_weights, endpoints=self.base_args.endpoints, device=device)
                
                optimizer_manager = OptimizerManager(
                    model=model,
                    optimizer_config={
                        'optimizer_type': 'adam',
                        'learning_rate': params.get('learning_rate', self.base_args.learning_rate),
                        'weight_decay': params.get('weight_decay', self.base_args.weight_decay)
                    },
                    scheduler_config={'scheduler_type': 'plateau', 'patience': 5},
                    early_stopping_config={'patience': 10, 'min_delta': 1e-4}
                )
                
                # Train model
                combo_dir = os.path.join(self.save_dir, f'combo_{i+1}')
                history = train_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    criterion=criterion,
                    optimizer_manager=optimizer_manager,
                    epochs=max_epochs,
                    device=device,
                    model_type='autoencoder',
                    endpoints=self.base_args.endpoints,
                    save_dir=combo_dir,
                    save_best=True
                )
                
                # Get best validation results
                best_epoch = np.argmin([loss['total'] for loss in history['val_loss']])
                best_val_loss = history['val_loss'][best_epoch]['total']
                best_val_metrics = history['val_metrics'][best_epoch]
                best_val_cosine = history['val_loss'][best_epoch].get('cosine_similarity', 0.0)
                
                # Compute combined score
                score = self._compute_score(best_val_metrics, best_val_loss, self.base_args.endpoints)
                
                # Store results
                result = {
                    'combination_id': i + 1,
                    'params': params,
                    'val_loss': best_val_loss,
                    'val_cosine_sim': best_val_cosine,
                    'val_metrics': best_val_metrics,
                    'score': score,
                    'best_epoch': best_epoch,
                    'model_path': os.path.join(combo_dir, 'best_model.pth'),
                    'train_epochs': len(history['train_loss'])
                }
                
                self.results.append(result)
                
                print(f"Score: {score:.4f}, Val Loss: {best_val_loss:.4f}")
                for endpoint in self.base_args.endpoints:
                    if endpoint in best_val_metrics:
                        print(f"Val {endpoint.upper()} AUC: {best_val_metrics[endpoint]['auc']:.4f}")
                
            except Exception as e:
                print(f"Error in combination {i+1}: {e}")
                continue
        
        # Create results DataFrame
        results_df = self._create_results_dataframe()
        results_df.to_csv(os.path.join(self.save_dir, 'grid_search_results.csv'), index=False)
        
        return results_df
    
    def _create_results_dataframe(self) -> pd.DataFrame:
        """Create DataFrame from results"""
        rows = []
        for result in self.results:
            row = {
                'combination_id': result['combination_id'],
                'score': result['score'],
                'val_loss': result['val_loss'],
                'val_cosine_sim': result['val_cosine_sim'],
                'best_epoch': result['best_epoch'],
                'train_epochs': result['train_epochs']
            }
            
            # Add parameter values
            for param, value in result['params'].items():
                row[param] = str(value) if isinstance(value, list) else value
            
            # Add endpoint metrics
            for endpoint in self.base_args.endpoints:
                if endpoint in result['val_metrics']:
                    row[f'{endpoint}_auc'] = result['val_metrics'][endpoint]['auc']
                    row[f'{endpoint}_accuracy'] = result['val_metrics'][endpoint]['accuracy']
                    row[f'{endpoint}_f1'] = result['val_metrics'][endpoint]['f1']
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        if not df.empty and 'score' in df.columns:
            return df.sort_values('score', ascending=False)
        return df
    
    def get_top_models(self, n: int = 2) -> List[Dict]:
        """Get top n models by score"""
        sorted_results = sorted(self.results, key=lambda x: x['score'], reverse=True)
        return sorted_results[:n]
    
    def test_top_models(self, n: int = 2) -> Dict:
        """Test top n models on test set"""
        top_models = self.get_top_models(n)
        test_results = {}
        
        device = torch.device(self.base_args.device if torch.cuda.is_available() else 'cpu')
        split_data = None
        if self.base_args.train_test_split_json:
            split_data = load_train_test_split(self.base_args.train_test_split_json)
        
        for i, model_info in enumerate(top_models):
            print(f"\n=== Testing Top Model {i+1} ===")
            print(f"Parameters: {model_info['params']}")
            print(f"Validation Score: {model_info['score']:.4f}")
            
            try:
                # Recreate data loaders with same parameters
                params = model_info['params']
                _, _, test_loader, class_weights = create_data_loaders(
                    pkl_files=self.base_args.pkl_files,
                    split_json=split_data,
                    val_split=self.base_args.val_split,
                    batch_size=params.get('batch_size', self.base_args.batch_size),
                    num_workers=0,
                    model_type='autoencoder',
                    endpoints=self.base_args.endpoints,
                    random_state=42
                )
                
                # Recreate and load model
                model = create_model(
                    model_type='autoencoder',
                    attention_k=params.get('attention_k', self.base_args.attention_k),
                    encoder_layers=params.get('encoder_layers', self.base_args.encoder_layers),
                    latent_dim=params.get('latent_dim', self.base_args.latent_dim),
                    predictor_layers=params.get('predictor_layers', self.base_args.predictor_layers),
                    dropout_rate=params.get('dropout_rate', self.base_args.dropout_rate)
                ).to(device)
                
                # Load best weights
                checkpoint = torch.load(model_info['model_path'], weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                
                # Create criterion
                if True:
                    criterion = CombinedLoss(
                        class_weights=class_weights,
                        reconstruction_weight=params.get('reconstruction_weight', self.base_args.reconstruction_weight),
                        prediction_weight=params.get('prediction_weight', self.base_args.prediction_weight),
                        endpoints=self.base_args.endpoints,
                        device=device
                    )
                else:
                    criterion = EndToEndLoss(class_weights=class_weights, device=device)
                
                # Test model
                test_losses, test_metrics = evaluate_model(
                    model=model,
                    dataloader=test_loader,
                    criterion=criterion,
                    device=device,
                    model_type='autoencoder',
                    endpoints=self.base_args.endpoints
                )
                
                test_results[f'model_{i+1}'] = {
                    'params': params,
                    'val_score': model_info['score'],
                    'test_losses': test_losses,
                    'test_metrics': test_metrics
                }
                
                print(f"Test Loss: {test_losses['total']:.4f}")
                if 'cosine_similarity' in test_losses:
                    print(f"Test Cosine Sim: {test_losses['cosine_similarity']:.4f}")
                for endpoint in self.base_args.endpoints:
                    if endpoint in test_metrics:
                        print(f"Test {endpoint.upper()} AUC: {test_metrics[endpoint]['auc']:.4f}")
                
            except Exception as e:
                print(f"Error testing model {i+1}: {e}")
                continue
        
        # Save test results
        with open(os.path.join(self.save_dir, 'top_models_test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        
        return test_results

def main():
    # Parse base arguments
    config = Config()
    args = config.parse_args()
    
    # Define parameter grid
    param_grid = {
        'attention_k': [8, 11, 16, 22],
        'latent_dim': [128, 256, 512],
        'encoder_layers': [[128, 64], [256, 128], [256, 128, 64]],
        'predictor_layers': [[32, 16], [64, 32], [128, 64, 32]],
        'dropout_rate': [0.3, 0.4],
        'learning_rate': [1e-4, 1e-3],
        'batch_size': [32, 64],
        'patience': [50]
    }
    
    # Add model-specific parameters
    if True:
        param_grid.update({
            'reconstruction_weight': [0.01, 0.10],
            'prediction_weight': [1.0]
        })
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"gridsearch_autoencoder_{timestamp}")
    
    # Run grid search
    grid_search = GridSearch(args, param_grid, save_dir)
    
    print(f"Grid search will test {len(grid_search._get_param_combinations())} combinations")
    print(f"Results will be saved to: {save_dir}")
    if args.train_test_split_json:
        print(f"Using train/test split from: {args.train_test_split_json}")
    else:
        print(f"Using automatic split: {1-2*args.val_split:.0%} train, {args.val_split:.0%} val, {args.val_split:.0%} test")
    
    # Run search
    results_df = grid_search.run_search(max_epochs=30)  # Shorter epochs for grid search
    
    print(f"\n=== Grid Search Complete ===")
    print(f"Top 5 configurations:")
    print(results_df.head()[['combination_id', 'score', 'val_loss'] + 
                             [f'{ep}_auc' for ep in args.endpoints]].to_string(index=False))
    
    # Test top 2 models
    print(f"\n=== Testing Top 2 Models ===")
    test_results = grid_search.test_top_models(n=2)
    
    print(f"\nGrid search complete! Results saved to: {save_dir}")

if __name__ == "__main__":
    main()