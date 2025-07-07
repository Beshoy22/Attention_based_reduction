#!/usr/bin/env python3
"""
Main script for training end-to-end model with attention mechanism
"""

import torch
import torch.nn as nn
import random
import numpy as np
import os
from datetime import datetime

from config import Config, load_train_test_split
from utils.dataloader import create_data_loaders
from utils.models import create_model, count_parameters
from utils.losses import EndToEndLoss
from utils.optimizers import OptimizerManager
from utils.train_loop import train_model, evaluate_model

def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # Parse arguments
    config = Config()
    args = config.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load train/test split
    split_data = load_train_test_split(args.train_test_split_json)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader, class_weights = create_data_loaders(
        pkl_files=args.pkl_files,
        split_json=split_data,
        val_split=args.val_split,
        batch_size=args.batch_size,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        model_type='endtoend',
        endpoints=args.endpoints,
        random_state=args.seed,
        validate_targets=args.validate_targets
    )
    
    # Create model
    print("Creating end-to-end model...")
    model = create_model(
        model_type='endtoend',
        endpoints=args.endpoints,
        attention_k=args.attention_k,
        encoder_layers=args.encoder_layers,
        predictor_layers=args.predictor_layers,
        dropout_rate=args.dropout_rate
    ).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Create loss function
    criterion = EndToEndLoss(
        class_weights=class_weights,
        device=device
    )
    
    # Create optimizer manager
    optimizer_config = {
        'optimizer_type': 'adam',
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay
    }
    
    scheduler_config = {
        'scheduler_type': 'plateau',
        'patience': 10,
        'factor': 0.5,
        'min_lr': 1e-6
    }
    
    early_stopping_config = {
        'patience': 20,
        'min_delta': 1e-4,
        'restore_best_weights': True
    }
    
    optimizer_manager = OptimizerManager(
        model=model,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        early_stopping_config=early_stopping_config
    )
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"endtoend_{timestamp}")
    
    # Train model
    print("Starting training...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer_manager=optimizer_manager,
        epochs=args.epochs,
        device=device,
        model_type='endtoend',
        save_dir=save_dir
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    
    # Load best model
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']}")
    
    test_losses, test_metrics = evaluate_model(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        model_type='endtoend'
    )
    
    print("\nTest Results:")
    print(f"Test Loss: {test_losses['total']:.4f}")
    if 'os6' in test_metrics:
        print(f"OS6 - AUC: {test_metrics['os6']['auc']:.4f}, "
              f"Accuracy: {test_metrics['os6']['accuracy']:.4f}, "
              f"F1: {test_metrics['os6']['f1']:.4f}")
    if 'os24' in test_metrics:
        print(f"OS24 - AUC: {test_metrics['os24']['auc']:.4f}, "
              f"Accuracy: {test_metrics['os24']['accuracy']:.4f}, "
              f"F1: {test_metrics['os24']['f1']:.4f}")
    
    # Extract and save reduced features from test set
    print("Extracting reduced features from test set...")
    model.eval()
    test_features = []
    test_patient_ids = []
    test_predictions = {'os6': [], 'os24': []}
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            
            # Filter for complete cases
            valid_indices = []
            for i in range(len(batch['os6'])):
                if batch['os6'][i] is not None and batch['os24'][i] is not None:
                    valid_indices.append(i)
            
            if len(valid_indices) == 0:
                continue
            
            features = features[valid_indices]
            reduced_features, os6_pred, os24_pred = model(features)
            
            test_features.append(reduced_features.cpu().numpy())
            test_patient_ids.extend([batch['patient_id'][i] for i in valid_indices])
            test_predictions['os6'].extend(os6_pred.squeeze().cpu().numpy())
            test_predictions['os24'].extend(os24_pred.squeeze().cpu().numpy())
    
    if test_features:
        # Concatenate all features
        all_features = np.concatenate(test_features, axis=0)
        
        # Save features and predictions
        np.save(os.path.join(save_dir, 'test_reduced_features.npy'), all_features)
        
        features_info = {
            'patient_ids': test_patient_ids,
            'predictions': test_predictions,
            'feature_shape': all_features.shape,
            'feature_description': 'Reduced features from second-to-last layer of end-to-end model'
        }
        
        import json
        with open(os.path.join(save_dir, 'test_features_info.json'), 'w') as f:
            json.dump(features_info, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        
        print(f"Saved reduced features with shape {all_features.shape}")
    
    # Save test results
    import json
    test_results = {
        'test_losses': test_losses,
        'test_metrics': test_metrics,
        'model_config': {
            'attention_k': args.attention_k,
            'encoder_layers': args.encoder_layers,
            'predictor_layers': args.predictor_layers,
            'dropout_rate': args.dropout_rate
        },
        'training_config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay
        }
    }
    
    with open(os.path.join(save_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\nResults saved to: {save_dir}")

if __name__ == "__main__":
    main()