#!/usr/bin/env python3
"""
Main script for training autoencoder model with attention mechanism
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
from utils.losses import CombinedLoss
from utils.optimizers import OptimizerManager
from utils.train_loop import train_model, evaluate_model
from utils.reconstruction_evaluator import evaluate_reconstruction_quality

def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def convert_to_json_serializable(obj):
    """Convert numpy/torch types to native Python types for JSON serialization"""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj

def main():
    # Parse arguments
    config = Config()
    args = config.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load train/test split if provided
    split_data = None
    if args.train_test_split_json:
        split_data = load_train_test_split(args.train_test_split_json)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader, class_weights = create_data_loaders(
        pkl_files=args.pkl_files,
        split_json=split_data,
        val_split=args.val_split,
        batch_size=args.batch_size,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        model_type='autoencoder',
        endpoints=args.endpoints,
        random_state=args.seed
    )
    
    # Create model
    print("Creating autoencoder model...")
    model = create_model(
        model_type='autoencoder',
        endpoints=args.endpoints,
        attention_k=args.attention_k,
        encoder_layers=args.encoder_layers,
        latent_dim=args.latent_dim,
        predictor_layers=args.predictor_layers,
        dropout_rate=args.dropout_rate,
        reconstruct_all=args.reconstruct_all
    ).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Create loss function
    criterion = CombinedLoss(
        class_weights=class_weights,
        reconstruction_weight=args.reconstruction_weight,
        prediction_weight=args.prediction_weight,
        endpoints=args.endpoints,
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
    save_dir = os.path.join(args.save_dir, f"autoencoder_{timestamp}")
    
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
        model_type='autoencoder',
        endpoints=args.endpoints,
        selection_metric=args.selection_metric,
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
        model_type='autoencoder',
        endpoints=args.endpoints
    )
    
    print("\nTest Results:")
    print(f"Test Loss: {test_losses['total']:.4f}")
    if 'cosine_similarity' in test_losses:
        print(f"Test Cosine Sim: {test_losses['cosine_similarity']:.4f}")
    for ep in args.endpoints:
        if ep in test_metrics:
            if ep in ['stage_t', 'stage_n']:
                print(f"{ep.upper()} - Accuracy: {test_metrics[ep]['accuracy']:.4f}, "
                      f"F1: {test_metrics[ep]['f1']:.4f}")
            else:
                print(f"{ep.upper()} - AUC: {test_metrics[ep]['auc']:.4f}, "
                      f"Accuracy: {test_metrics[ep]['accuracy']:.4f}, "
                      f"F1: {test_metrics[ep]['f1']:.4f}")
    
    # Perform comprehensive reconstruction evaluation (only if requested)
    reconstruction_results = None
    if hasattr(args, 'reconstruction_evaluation') and getattr(args, 'reconstruction_evaluation', False):
        print("\n" + "="*60)
        print("COMPREHENSIVE RECONSTRUCTION EVALUATION")
        print("="*60)
        
        # Prepare model config for evaluation
        eval_model_config = {
            'attention_k': args.attention_k,
            'latent_dim': args.latent_dim,
            'encoder_layers': args.encoder_layers,
            'predictor_layers': args.predictor_layers,
            'dropout_rate': args.dropout_rate,
            'endpoints': args.endpoints,
            'reconstruct_all': args.reconstruct_all
        }
        
        # Perform reconstruction evaluation (using the same train/test split if provided)
        reconstruction_results = evaluate_reconstruction_quality(
            model=model,
            pkl_files=args.pkl_files,
            model_config=eval_model_config,
            split_json=split_data,
            save_dir=save_dir,
            device=device
        )
    else:
        print("\n" + "="*60)
        print("SKIPPING RECONSTRUCTION EVALUATION")
        print("Use --reconstruction-evaluation flag to enable detailed analysis")
        print("="*60)
    
    # Save test results
    import json
    test_results = {
        'test_losses': test_losses,
        'test_metrics': test_metrics,
        'reconstruction_evaluation': reconstruction_results,
        'model_config': {
            'attention_k': args.attention_k,
            'encoder_layers': args.encoder_layers,
            'latent_dim': args.latent_dim,
            'predictor_layers': args.predictor_layers,
            'dropout_rate': args.dropout_rate,
            'reconstruct_all': args.reconstruct_all
        },
        'training_config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'reconstruction_weight': args.reconstruction_weight,
            'prediction_weight': args.prediction_weight
        }
    }
    
    with open(os.path.join(save_dir, 'test_results.json'), 'w') as f:
        json.dump(convert_to_json_serializable(test_results), f, indent=2)
    
    print(f"\nResults saved to: {save_dir}")

if __name__ == "__main__":
    main()