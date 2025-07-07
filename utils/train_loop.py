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
    
    @staticmethod
    def calculate_survival_metrics(y_time: np.ndarray, y_event: np.ndarray, y_pred_hazard: np.ndarray) -> Dict[str, float]:
        """
        Calculate survival analysis metrics including C-Index
        
        Args:
            y_time: Survival times
            y_event: Event indicators (1 if event occurred, 0 if censored)
            y_pred_hazard: Predicted log hazard ratios
            
        Returns:
            Dictionary of survival metrics
        """
        metrics = {}
        
        # Calculate Concordance Index (C-Index)
        try:
            c_index = MetricsCalculator._calculate_concordance_index(y_time, y_event, y_pred_hazard)
            metrics['c_index'] = c_index
        except Exception as e:
            warnings.warn(f"Failed to calculate C-Index: {e}")
            metrics['c_index'] = 0.5
        
        # Calculate percentage of events
        metrics['event_rate'] = np.mean(y_event)
        
        # Calculate median survival time for events
        event_times = y_time[y_event == 1]
        if len(event_times) > 0:
            metrics['median_event_time'] = np.median(event_times)
        else:
            metrics['median_event_time'] = np.nan
        
        return metrics
    
    @staticmethod
    def _calculate_concordance_index(time: np.ndarray, event: np.ndarray, risk_score: np.ndarray) -> float:
        """
        Calculate Harrell's Concordance Index (C-Index) for survival analysis
        
        The C-Index measures the proportion of all pairs of patients where the patient
        with higher risk score has shorter survival time.
        
        Args:
            time: Survival times
            event: Event indicators (1 if event occurred, 0 if censored)
            risk_score: Predicted risk scores (higher score = higher risk)
            
        Returns:
            C-Index value between 0 and 1 (0.5 = random, 1.0 = perfect)
        """
        # Convert to numpy arrays and handle NaN values
        time = np.asarray(time)
        event = np.asarray(event)
        risk_score = np.asarray(risk_score)
        
        # Remove any NaN or invalid values
        valid_mask = (~np.isnan(time)) & (~np.isnan(event)) & (~np.isnan(risk_score))
        if not valid_mask.any():
            warnings.warn("No valid data for C-Index calculation")
            return 0.5
        
        time = time[valid_mask]
        event = event[valid_mask] 
        risk_score = risk_score[valid_mask]
        
        if len(time) < 2:
            warnings.warn("Not enough samples for C-Index calculation")
            return 0.5
        
        # Count concordant, discordant, and tied pairs
        concordant = 0
        discordant = 0
        tied = 0
        comparable = 0
        
        n = len(time)
        for i in range(n):
            for j in range(i + 1, n):
                # Only consider pairs where we can determine the ordering
                if event[i] == 1 or event[j] == 1:
                    comparable += 1
                    
                    # Determine which patient has worse outcome (shorter survival)
                    if event[i] == 1 and event[j] == 1:
                        # Both have events - compare times directly
                        if time[i] < time[j]:
                            worse_idx, better_idx = i, j
                        elif time[i] > time[j]:
                            worse_idx, better_idx = j, i
                        else:
                            # Tied times
                            tied += 1
                            continue
                    elif event[i] == 1:
                        # Patient i has event, j is censored
                        if time[i] <= time[j]:
                            worse_idx, better_idx = i, j
                        else:
                            # Can't determine ordering (event after censoring)
                            comparable -= 1
                            continue
                    else:
                        # Patient j has event, i is censored
                        if time[j] <= time[i]:
                            worse_idx, better_idx = j, i
                        else:
                            # Can't determine ordering (event after censoring)
                            comparable -= 1
                            continue
                    
                    # Check if risk scores are concordant with outcomes
                    if risk_score[worse_idx] > risk_score[better_idx]:
                        concordant += 1
                    elif risk_score[worse_idx] < risk_score[better_idx]:
                        discordant += 1
                    else:
                        tied += 1
        
        if comparable == 0:
            warnings.warn("No comparable pairs for C-Index calculation")
            return 0.5
        
        # C-Index = (concordant + 0.5 * tied) / (concordant + discordant + tied)
        c_index = (concordant + 0.5 * tied) / comparable
        
        return c_index

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
            if ep == 'survival':
                # Handle survival data for endtoend training
                survival_data = [batch['endpoints'][ep][i] for i in valid_indices]
                if all(data is not None and isinstance(data, dict) and 'time' in data and 'event' in data for data in survival_data):
                    times = torch.stack([data['time'] for data in survival_data]).to(device)
                    events = torch.stack([data['event'] for data in survival_data]).to(device)
                    targets[ep] = {'time': times, 'event': events}
                else:
                    # Skip this batch if survival data is invalid
                    continue
            else:
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

def train_epoch_survival(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: 'CoxPHLoss',
    optimizer_manager: 'OptimizerManager',
    device: str
) -> Dict[str, float]:
    """
    Train one epoch for survival model
    """
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        if batch is None:  # Skip invalid batches from collate_fn
            continue
            
        features = batch['features'].to(device)
        
        # Extract survival data
        survival_data = batch['endpoints']['survival']
        if survival_data is None:
            continue
            
        valid_indices = []
        times = []
        events = []
        
        for i, data in enumerate(survival_data):
            if (data is not None and isinstance(data, dict) and 
                'time' in data and 'event' in data):
                valid_indices.append(i)
                times.append(data['time'])
                events.append(data['event'])
        
        if len(valid_indices) == 0:
            continue
        
        # Filter to valid samples
        features = features[valid_indices]
        times = torch.stack(times).to(device)
        events = torch.stack(events).to(device)
        
        # Forward pass
        log_hazards = model(features).squeeze()
        
        # Calculate loss
        loss = criterion(log_hazards, times, events)
        
        # Backward pass
        optimizer_manager.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer_manager.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return {'total': avg_loss, 'survival': avg_loss}

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
                    if ep == 'survival':
                        # Handle survival data for endtoend model
                        survival_data = [batch['endpoints'][ep][i] for i in valid_indices]
                        if all(data is not None and isinstance(data, dict) and 'time' in data and 'event' in data for data in survival_data):
                            times = torch.stack([data['time'] for data in survival_data]).to(device)
                            events = torch.stack([data['event'] for data in survival_data]).to(device)
                            targets[ep] = {'time': times, 'event': events}
                        else:
                            # Skip this batch if survival data is invalid
                            continue
                    else:
                        targets[ep] = torch.stack([batch['endpoints'][ep][i] for i in valid_indices]).to(device)
                
                # Forward pass
                _, predictions = model(features)
                
                # Calculate loss
                loss, loss_dict = criterion(predictions, targets)
                
                # Store predictions for metrics and saving
                for ep in endpoints:
                    if ep == 'survival':
                        # Handle survival data - targets[ep] is dict with 'time' and 'event'
                        if isinstance(targets[ep], dict) and 'time' in targets[ep] and 'event' in targets[ep]:
                            times = targets[ep]['time'].cpu().numpy()
                            events = targets[ep]['event'].cpu().numpy()
                            hazard_pred = predictions[ep].squeeze().cpu().numpy()
                            
                            # Store all survival data together
                            if 'survival_time' not in all_targets:
                                all_targets['survival_time'] = []
                                all_targets['survival_event'] = []
                                all_predictions['survival_hazard'] = []
                            
                            all_targets['survival_time'].extend(times)
                            all_targets['survival_event'].extend(events)
                            all_predictions['survival_hazard'].extend(hazard_pred)
                    else:
                        # Handle traditional endpoints
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
                        if ep == 'survival':
                            # Handle survival data - targets[ep] is dict with 'time' and 'event'
                            if isinstance(targets[ep], dict) and 'time' in targets[ep] and 'event' in targets[ep]:
                                times = targets[ep]['time'].cpu().numpy()
                                events = targets[ep]['event'].cpu().numpy()
                                hazard_pred = predictions[ep].squeeze().cpu().numpy()
                                
                                # Store all survival data together
                                if 'survival_time' not in all_targets:
                                    all_targets['survival_time'] = []
                                    all_targets['survival_event'] = []
                                    all_predictions['survival_hazard'] = []
                                
                                all_targets['survival_time'].extend(times)
                                all_targets['survival_event'].extend(events)
                                all_predictions['survival_hazard'].extend(hazard_pred)
                                
                                # Store patient IDs for survival predictions
                                all_patient_ids.extend(batch['patient_id'][:len(times)])
                        else:
                            # Handle traditional endpoints
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
        if ep == 'survival':
            # Handle survival metrics
            if ('survival_time' in all_targets and 'survival_event' in all_targets and 
                'survival_hazard' in all_predictions and len(all_targets['survival_time']) > 0):
                ep_metrics = MetricsCalculator.calculate_survival_metrics(
                    np.array(all_targets['survival_time']),
                    np.array(all_targets['survival_event']),
                    np.array(all_predictions['survival_hazard'])
                )
                metrics['survival'] = ep_metrics
        else:
            # Handle traditional endpoints
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
                    if ep == 'survival':
                        # Handle survival predictions
                        if (idx < len(all_targets.get('survival_time', [])) and 
                            idx < len(all_targets.get('survival_event', [])) and
                            idx < len(all_predictions.get('survival_hazard', []))):
                            record['survival_time_true'] = all_targets['survival_time'][idx]
                            record['survival_event_true'] = all_targets['survival_event'][idx]
                            record['survival_hazard_pred'] = all_predictions['survival_hazard'][idx]
                    else:
                        # Handle traditional endpoints
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
        model_type: 'autoencoder', 'endtoend', or 'survival'
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
    
    def get_validation_metric(metrics: Dict, metric_type: str) -> float:
        """Get validation metric for model selection"""
        if metric_type == 'c_index':
            return metrics.get('survival', {}).get('c_index', 0.5)
        elif metric_type == 'auc':
            return compute_combined_auc(metrics)
        else:  # loss
            return 0.0  # Will use val_loss instead
    
    best_val_loss = float('inf')
    best_val_auc = 0.0
    best_val_c_index = 0.0
    
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
        elif model_type == 'survival':
            train_losses = train_epoch_survival(
                model, train_loader, criterion, optimizer_manager, device
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
        val_c_index = get_validation_metric(val_metrics, 'c_index')
        
        # Use appropriate metric for scheduler and early stopping
        if selection_metric == 'loss':
            scheduler_metric = val_loss
        elif selection_metric == 'c_index':
            scheduler_metric = -val_c_index  # Negative C-Index for minimization
        else:  # auc
            scheduler_metric = -val_auc  # Negative AUC for minimization
        
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
            elif selection_metric == 'c_index' and val_c_index > best_val_c_index:
                best_val_c_index = val_c_index
                save_current = True
                print(f"  New best validation C-Index: {val_c_index:.4f}")
            
            if save_current:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer_manager.state_dict(),
                    'val_loss': val_loss,
                    'val_auc': val_auc,
                    'val_c_index': val_c_index,
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
                if ep == 'survival':
                    print(f"  Val {ep.upper()} C-Index: {val_metrics[ep]['c_index']:.4f}")
                elif ep in ['stage_t', 'stage_n']:
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
    def convert_to_json_serializable(obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(v) for v in obj]
        else:
            return obj
    
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json_history = convert_to_json_serializable(history)
        json.dump(json_history, f, indent=2)
    
    print("Training completed!")
    if selection_metric == 'loss':
        print(f"Best validation loss: {best_val_loss:.4f}")
    elif selection_metric == 'c_index':
        print(f"Best validation C-Index: {best_val_c_index:.4f}")
    else:
        print(f"Best validation AUC: {best_val_auc:.4f}")
    
    return history