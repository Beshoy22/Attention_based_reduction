import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
import os
import json

from .losses import CombinedLoss, EndToEndLoss, prepare_batch_targets
from .optimizers import OptimizerManager

def compute_cosine_similarity(input_features: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Compute average cosine similarity between input and reconstructed features"""
    # Flatten both tensors: (batch_size, k, 512) -> (batch_size, k*512)
    input_flat = input_features.view(input_features.size(0), -1)
    recon_flat = reconstructed.view(reconstructed.size(0), -1)
    
    # Compute cosine similarity for each sample in batch
    cos_sim = F.cosine_similarity(input_flat, recon_flat, dim=1)
    return cos_sim.mean().item()

class MetricsCalculator:
    """Calculate metrics for different endpoint types"""
    
    @staticmethod
    def calculate_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """Calculate binary classification metrics"""
        metrics = {}
        
        if len(np.unique(y_true)) > 1:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        else:
            metrics['auc'] = 0.5
        
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
        
        return metrics
    
    @staticmethod
    def calculate_multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate multiclass classification metrics"""
        metrics = {}
        
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
        
        return metrics

def train_epoch_autoencoder(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: CombinedLoss,
    optimizer_manager: OptimizerManager,
    device: str,
    endpoints: List[str]
) -> Dict[str, float]:
    """
    Train one epoch for autoencoder model
    """
    model.train()
    
    total_losses = {'total': 0.0, 'reconstruction': 0.0, 'prediction': 0.0}
    for ep in endpoints:
        total_losses[ep] = 0.0
    cosine_similarities = []
    num_batches = 0
    
    for batch in dataloader:
        features = batch['features'].to(device)
        targets = prepare_batch_targets(batch, device, endpoints)
        
        # Forward pass
        reconstructed, predictions = model(features)
        attended_features = model.attention(features)
        
        # Debug: Check for NaN in model outputs
        if torch.any(torch.isnan(reconstructed)):
            print("WARNING: NaN detected in reconstructed features")
        
        for ep, pred in predictions.items():
            if torch.any(torch.isnan(pred)):
                print(f"WARNING: NaN detected in predictions for endpoint {ep}")
                print(f"Prediction stats: min={pred.min().item():.6f}, max={pred.max().item():.6f}, mean={pred.mean().item():.6f}")
                # Check model parameters for NaN
                for name, param in model.named_parameters():
                    if torch.any(torch.isnan(param)):
                        print(f"NaN detected in parameter {name}")
        
        # Compute cosine similarity (always use appropriate target for comparison)
        if model.reconstruct_all:
            cos_sim = compute_cosine_similarity(features, reconstructed)
        else:
            cos_sim = compute_cosine_similarity(attended_features, reconstructed)
        cosine_similarities.append(cos_sim)
        
        # Calculate loss
        if model.reconstruct_all:
            loss, loss_dict = criterion(reconstructed, attended_features, predictions, targets, original_input=features)
        else:
            loss, loss_dict = criterion(reconstructed, attended_features, predictions, targets)
        
        # Backward pass
        optimizer_manager.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer_manager.step()
        
        # Accumulate losses
        for key in total_losses.keys():
            if key in loss_dict:
                total_losses[key] += loss_dict[key]
        
        num_batches += 1
    
    # Average losses and cosine similarity
    avg_losses = {key: total / num_batches for key, total in total_losses.items()}
    avg_losses['cosine_similarity'] = np.mean(cosine_similarities)
    
    return avg_losses

def train_epoch_endtoend(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: EndToEndLoss,
    optimizer_manager: OptimizerManager,
    device: str,
    endpoints: List[str]
) -> Dict[str, float]:
    """
    Train one epoch for end-to-end model
    """
    model.train()
    
    total_losses = {'total': 0.0}
    for ep in endpoints:
        total_losses[ep] = 0.0
    num_batches = 0
    
    for batch in dataloader:
        features = batch['features'].to(device)
        
        # Filter for complete cases
        valid_indices = []
        for i in range(len(batch['patient_id'])):
            if all(batch['endpoints'][ep][i] is not None for ep in endpoints):
                valid_indices.append(i)
        
        if len(valid_indices) == 0:
            continue
        
        features = features[valid_indices]
        targets = {}
        for ep in endpoints:
            targets[ep] = torch.stack([batch['endpoints'][ep][i] for i in valid_indices]).to(device)
        
        # Forward pass
        _, predictions = model(features)
        
        # Calculate loss
        loss, loss_dict = criterion(predictions, targets)
        
        # Backward pass
        optimizer_manager.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer_manager.step()
        
        # Accumulate losses
        for key in total_losses.keys():
            if key in loss_dict:
                total_losses[key] += loss_dict[key]
        
        num_batches += 1
    
    avg_losses = {key: total / num_batches if num_batches > 0 else 0.0 
                  for key, total in total_losses.items()}
    
    return avg_losses

def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    model_type: str,
    endpoints: List[str] = ['os6', 'os24'],
    save_predictions: bool = False,
    save_path: str = None
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Evaluate model on validation/test set
    
    Args:
        model: Model to evaluate
        dataloader: Validation/test dataloader
        criterion: Loss function
        device: Device to run on
        model_type: 'autoencoder' or 'endtoend'
    
    Returns:
        Tuple of (losses, metrics)
    """
    model.eval()
    
    model.eval()
    
    total_losses = {'total': 0.0}
    for ep in endpoints:
        total_losses[ep] = 0.0
    if model_type == 'autoencoder':
        total_losses.update({'reconstruction': 0.0, 'prediction': 0.0})
    
    # For metrics calculation and prediction saving
    all_predictions = {ep: [] for ep in endpoints}
    all_targets = {ep: [] for ep in endpoints}
    all_patient_ids = []
    cosine_similarities = []
    
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            
            if model_type == 'endtoend':
                # Filter for complete cases
                valid_indices = []
                for i in range(len(batch['patient_id'])):
                    if all(batch['endpoints'][ep][i] is not None for ep in endpoints):
                        valid_indices.append(i)
                
                if len(valid_indices) == 0:
                    continue
                
                features = features[valid_indices]
                targets = {}
                for ep in endpoints:
                    targets[ep] = torch.stack([batch['endpoints'][ep][i] for i in valid_indices]).to(device)
                
                # Forward pass
                _, predictions = model(features)
                
                # Calculate loss
                loss, loss_dict = criterion(predictions, targets)
                
                # Store predictions for metrics and saving
                for ep in endpoints:
                    all_targets[ep].extend(targets[ep].cpu().numpy())
                    
                    if ep in ['stage_t', 'stage_n']:
                        # Multiclass: get predicted classes
                        pred_classes = torch.argmax(predictions[ep], dim=1)
                        all_predictions[ep].extend(pred_classes.cpu().numpy())
                    else:
                        # Binary: get probabilities
                        all_predictions[ep].extend(predictions[ep].squeeze().cpu().numpy())
                
                # Store patient IDs
                all_patient_ids.extend([batch['patient_id'][i] for i in valid_indices])
                
            else:  # autoencoder
                targets = prepare_batch_targets(batch, device, endpoints)
                
                # Forward pass
                reconstructed, predictions = model(features)
                attended_features = model.attention(features)
                
                # Compute cosine similarity (always use appropriate target for comparison)
                if model.reconstruct_all:
                    cos_sim = compute_cosine_similarity(features, reconstructed)
                else:
                    cos_sim = compute_cosine_similarity(attended_features, reconstructed)
                cosine_similarities.append(cos_sim)
                
                # Calculate loss
                if model.reconstruct_all:
                    loss, loss_dict = criterion(reconstructed, attended_features, predictions, targets, original_input=features)
                else:
                    loss, loss_dict = criterion(reconstructed, attended_features, predictions, targets)
                
                # Store predictions for metrics and saving (only for specified endpoints with targets)
                for ep in endpoints:
                    if targets[ep] is not None:
                        valid_mask = targets[ep] != -1
                        if valid_mask.any():
                            all_targets[ep].extend(targets[ep][valid_mask].cpu().numpy())
                            
                            if ep in ['stage_t', 'stage_n']:
                                # Multiclass: get predicted classes
                                pred_classes = torch.argmax(predictions[ep][valid_mask], dim=1)
                                all_predictions[ep].extend(pred_classes.cpu().numpy())
                            else:
                                # Binary: get probabilities
                                all_predictions[ep].extend(predictions[ep].squeeze()[valid_mask].cpu().numpy())
                            
                            # Store patient IDs for valid predictions
                            valid_patient_ids = [batch['patient_id'][i] for i in range(len(batch['patient_id'])) if valid_mask[i]]
                            all_patient_ids.extend(valid_patient_ids)
            
            # Accumulate losses
            for key in total_losses.keys():
                if key in loss_dict:
                    total_losses[key] += loss_dict[key]
            
            num_batches += 1
            
    # Average losses
    avg_losses = {key: total / num_batches if num_batches > 0 else 0.0 
                  for key, total in total_losses.items()}
    
    # Add cosine similarity for autoencoder
    if model_type == 'autoencoder' and cosine_similarities:
        avg_losses['cosine_similarity'] = np.mean(cosine_similarities)
    
    # Calculate metrics (only for specified endpoints)
    metrics = {}
    for ep in endpoints:
        if len(all_targets[ep]) > 0:
            if ep in ['stage_t', 'stage_n']:
                # Multiclass: predictions are already class indices
                ep_metrics = MetricsCalculator.calculate_multiclass_metrics(
                    np.array(all_targets[ep]), np.array(all_predictions[ep])
                )
            else:
                # Binary: convert probabilities to binary predictions
                ep_pred_binary = (np.array(all_predictions[ep]) > 0.5).astype(int)
                ep_metrics = MetricsCalculator.calculate_binary_metrics(
                    np.array(all_targets[ep]), ep_pred_binary, np.array(all_predictions[ep])
                )
            metrics[ep] = ep_metrics
    
    # Save predictions if requested
    if save_predictions and save_path and all_patient_ids:
        import pandas as pd
        pred_data = []
        
        unique_patients = list(set(all_patient_ids))
        for patient_id in unique_patients:
            patient_indices = [i for i, pid in enumerate(all_patient_ids) if pid == patient_id]
            if patient_indices:
                idx = patient_indices[0]
                record = {'patient_id': patient_id}
                
                for ep in endpoints:
                    if idx < len(all_targets[ep]):
                        record[f'{ep}_true'] = all_targets[ep][idx]
                        
                        if ep in ['stage_t', 'stage_n']:
                            # Multiclass: prediction is class index
                            record[f'{ep}_pred'] = all_predictions[ep][idx]
                        else:
                            # Binary: prediction is probability and binary
                            record[f'{ep}_pred'] = all_predictions[ep][idx]
                            record[f'{ep}_pred_binary'] = int(all_predictions[ep][idx] > 0.5)
                
                pred_data.append(record)
        
        pd.DataFrame(pred_data).to_csv(save_path, index=False)
    
    return avg_losses, metrics

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer_manager: OptimizerManager,
    epochs: int,
    device: str,
    model_type: str,
    endpoints: List[str],
    save_dir: str,
    selection_metric: str = 'loss',
    save_best: bool = True
) -> Dict[str, List]:
    """
    Main training loop
    
    Args:
        model: Model to train
        train_loader: Training dataloader
        val_loader: Validation dataloader
        criterion: Loss function
        optimizer_manager: Optimizer manager
        epochs: Number of epochs
        device: Device to run on
        model_type: 'autoencoder' or 'endtoend'
        save_dir: Directory to save results
        save_best: Whether to save best model
    
    Returns:
        Training history dictionary
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_metrics': [], 'val_metrics': [],
        'learning_rate': []
    }
    
    def compute_combined_auc(metrics: Dict) -> float:
        """Compute average AUC for binary endpoints"""
        aucs = []
        for ep in endpoints:
            if ep in metrics and ep in ['os6', 'os24', 'stage_m']:  # Only binary endpoints have AUC
                aucs.append(metrics[ep].get('auc', 0.0))
        return np.mean(aucs) if aucs else 0.0
    
    best_val_loss = float('inf')
    best_val_auc = 0.0
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Model type: {model_type}")
    print(f"Device: {device}")
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training
        if model_type == 'autoencoder':
            train_losses = train_epoch_autoencoder(
                model, train_loader, criterion, optimizer_manager, device, endpoints
            )
        else:
            train_losses = train_epoch_endtoend(
                model, train_loader, criterion, optimizer_manager, device, endpoints
            )
        
        # Validation (save predictions every 10 epochs)
        save_val_preds = (epoch + 1) % 10 == 0
        val_pred_path = os.path.join(save_dir, f'val_predictions_epoch_{epoch+1}.csv') if save_val_preds else None
        
        val_losses, val_metrics = evaluate_model(
            model, val_loader, criterion, device, model_type, endpoints,
            save_predictions=save_val_preds, save_path=val_pred_path
        )
        
        # Update learning rate and check early stopping
        val_loss = val_losses['total']
        val_auc = compute_combined_auc(val_metrics)
        
        # Use appropriate metric for scheduler and early stopping
        scheduler_metric = val_loss if selection_metric == 'loss' else -val_auc  # Negative AUC for minimization
        early_stop = optimizer_manager.step(scheduler_metric)
        
        # Save best model based on selection metric
        if save_best:
            save_current = False
            if selection_metric == 'loss' and val_loss < best_val_loss:
                best_val_loss = val_loss
                save_current = True
                print(f"  New best validation loss: {val_loss:.4f}")
            elif selection_metric == 'auc' and val_auc > best_val_auc:
                best_val_auc = val_auc
                save_current = True
                print(f"  New best validation AUC: {val_auc:.4f}")
            
            if save_current:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer_manager.state_dict(),
                    'val_loss': val_loss,
                    'val_auc': val_auc,
                    'val_metrics': val_metrics
                }, os.path.join(save_dir, 'best_model.pth'))
        
        # Record history
        history['train_loss'].append(train_losses)
        history['val_loss'].append(val_losses)
        history['val_metrics'].append(val_metrics)
        history['learning_rate'].append(optimizer_manager.get_current_lr())
        
        # Print progress
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.2f}s)")
        print(f"  Train Loss: {train_losses['total']:.4f}")
        if model_type == 'autoencoder' and 'cosine_similarity' in train_losses:
            print(f"  Train Cosine Sim: {train_losses['cosine_similarity']:.4f}")
        print(f"  Val Loss: {val_losses['total']:.4f}")
        if model_type == 'autoencoder' and 'cosine_similarity' in val_losses:
            print(f"  Val Cosine Sim: {val_losses['cosine_similarity']:.4f}")
        for ep in endpoints:
            if ep in val_metrics:
                if ep in ['stage_t', 'stage_n']:
                    print(f"  Val {ep.upper()} Acc: {val_metrics[ep]['accuracy']:.4f}")
                else:
                    print(f"  Val {ep.upper()} AUC: {val_metrics[ep]['auc']:.4f}")
        print(f"  LR: {optimizer_manager.get_current_lr():.2e}")
        print()
        
        # Early stopping
        if early_stop:
            print("Early stopping triggered!")
            break
    
    # Save final model and history
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer_manager.state_dict(),
        'history': history
    }, os.path.join(save_dir, 'final_model.pth'))
    
    # Save history as JSON
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        # Convert tensors to lists for JSON serialization
        json_history = {}
        for key, value in history.items():
            if key in ['train_loss', 'val_loss']:
                json_history[key] = value
            else:
                json_history[key] = [float(v) if isinstance(v, (int, float)) else v for v in value]
        json.dump(json_history, f, indent=2)
    
    print("Training completed!")
    if selection_metric == 'loss':
        print(f"Best validation loss: {best_val_loss:.4f}")
    else:
        print(f"Best validation AUC: {best_val_auc:.4f}")
    
    return history