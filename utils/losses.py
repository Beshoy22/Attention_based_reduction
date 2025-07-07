import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import warnings

class BalancedBCELoss(nn.Module):
    """Balanced Binary Cross Entropy Loss"""
    def __init__(self, pos_weight: torch.Tensor):
        super(BalancedBCELoss, self).__init__()
        self.pos_weight = pos_weight
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Handle NaN/Inf predictions by replacing with reasonable values
        if torch.any(torch.isnan(predictions)) or torch.any(torch.isinf(predictions)):
            print(f"WARNING: Found NaN or Inf in predictions, replacing with 0.5")
            predictions = torch.where(torch.isnan(predictions) | torch.isinf(predictions), 
                                    torch.tensor(0.5, dtype=predictions.dtype, device=predictions.device), 
                                    predictions)
        
        # Clamp predictions to safe range to prevent numerical issues
        predictions = torch.clamp(predictions, 1e-7, 1.0 - 1e-7)
        
        return F.binary_cross_entropy(predictions, targets, weight=self.pos_weight, reduction='mean')

class CoxPHLoss(nn.Module):
    """
    Cox Proportional Hazards Loss for survival analysis
    
    Implements the partial likelihood loss for the Cox proportional hazards model.
    This loss handles right-censored survival data by computing the partial likelihood
    based on the ordering of event times.
    """
    def __init__(self, regularization_weight: float = 1e-4):
        super(CoxPHLoss, self).__init__()
        self.regularization_weight = regularization_weight
    
    def forward(self, hazard_pred: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        """
        Compute Cox Proportional Hazards partial likelihood loss
        
        Args:
            hazard_pred: Predicted log hazard ratios (N,)
            times: Survival times (N,)
            events: Event indicators (1 if event occurred, 0 if censored) (N,)
            
        Returns:
            Loss value (scalar)
        """
        # Ensure inputs are 1D
        hazard_pred = hazard_pred.squeeze()
        times = times.squeeze() 
        events = events.squeeze()
        
        # Validate inputs
        if hazard_pred.dim() != 1 or times.dim() != 1 or events.dim() != 1:
            raise ValueError("All inputs must be 1D tensors")
        
        if len(hazard_pred) != len(times) or len(times) != len(events):
            raise ValueError("All inputs must have the same length")
        
        # Filter out any invalid values
        valid_mask = (~torch.isnan(hazard_pred)) & (~torch.isnan(times)) & (~torch.isnan(events))
        if not valid_mask.any():
            warnings.warn("No valid survival data in batch, returning zero loss")
            return torch.tensor(0.0, device=hazard_pred.device, requires_grad=True)
        
        hazard_pred = hazard_pred[valid_mask]
        times = times[valid_mask]
        events = events[valid_mask]
        
        # Only consider patients with events for the partial likelihood
        event_mask = events == 1
        if not event_mask.any():
            warnings.warn("No events in batch, returning zero loss")
            return torch.tensor(0.0, device=hazard_pred.device, requires_grad=True)
        
        # Get event times and corresponding hazard predictions
        event_times = times[event_mask]
        event_hazards = hazard_pred[event_mask]
        
        # Compute partial likelihood
        total_loss = 0.0
        n_events = 0
        
        for i, (event_time, event_hazard) in enumerate(zip(event_times, event_hazards)):
            # Find all patients at risk at this event time (time >= event_time)
            at_risk_mask = times >= event_time
            if not at_risk_mask.any():
                continue
                
            at_risk_hazards = hazard_pred[at_risk_mask]
            
            # Handle tied event times by using the average log-likelihood
            tied_mask = (times == event_time) & (events == 1)
            n_tied = tied_mask.sum().item()
            
            if n_tied > 1:
                # Breslow method for tied times
                log_risk_sum = torch.logsumexp(at_risk_hazards, dim=0)
                tied_hazards = hazard_pred[tied_mask]
                tied_hazard_sum = tied_hazards.sum()
                
                # Partial likelihood contribution for tied events
                partial_loss = tied_hazard_sum - n_tied * log_risk_sum
            else:
                # Standard partial likelihood
                log_risk_sum = torch.logsumexp(at_risk_hazards, dim=0)
                partial_loss = event_hazard - log_risk_sum
            
            total_loss += partial_loss
            n_events += 1
        
        if n_events == 0:
            warnings.warn("No valid events processed, returning zero loss")
            return torch.tensor(0.0, device=hazard_pred.device, requires_grad=True)
        
        # Average over number of events and negate (since we maximize log-likelihood)
        cox_loss = -total_loss / n_events
        
        # Add L2 regularization to prevent overfitting
        regularization = self.regularization_weight * torch.norm(hazard_pred, p=2)
        
        return cox_loss + regularization

class CombinedLoss(nn.Module):
    """
    Combined loss for autoencoder model with mixed endpoint types
    """
    def __init__(
        self, 
        class_weights: Dict[str, torch.Tensor],
        reconstruction_weight: float = 1.0,
        prediction_weight: float = 1.0,
        endpoints: List[str] = ['os6', 'os24'],
        device: str = 'cuda'
    ):
        super(CombinedLoss, self).__init__()
        
        self.reconstruction_weight = reconstruction_weight
        self.prediction_weight = prediction_weight
        self.endpoints = endpoints
        self.device = device
        
        # Reconstruction loss
        self.reconstruction_loss = nn.MSELoss()
        
        # Different losses for different endpoint types
        self.endpoint_losses = nn.ModuleDict()
        for ep in endpoints:
            if ep == 'survival':
                # Cox Proportional Hazards loss for survival analysis
                self.endpoint_losses[ep] = CoxPHLoss()
            elif ep in ['stage_t', 'stage_n']:
                # Multiclass classification
                weight = class_weights.get(ep, None)
                if weight is not None:
                    weight = weight.to(device)
                self.endpoint_losses[ep] = nn.CrossEntropyLoss(weight=weight)
            else:
                # Binary classification
                if ep in class_weights:
                    pos_weight = class_weights[ep][1] if len(class_weights[ep]) > 1 else class_weights[ep][0]
                    self.endpoint_losses[ep] = BalancedBCELoss(pos_weight.to(device))
                else:
                    self.endpoint_losses[ep] = nn.BCELoss()
    
    def forward(
        self, 
        reconstructed: torch.Tensor,
        attended_features: torch.Tensor,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, Optional[torch.Tensor]],
        original_input: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        losses = {}
        
        # Reconstruction loss - choose target based on reconstruction mode
        if original_input is not None:
            # Full input reconstruction mode
            recon_target = original_input
        else:
            # Attended features reconstruction mode (default)
            recon_target = attended_features
        
        recon_loss = self.reconstruction_loss(reconstructed, recon_target)
        losses['reconstruction'] = recon_loss.item()
        total_loss = self.reconstruction_weight * recon_loss
        
        # Prediction losses for each endpoint
        prediction_loss = 0.0
        for ep in self.endpoints:
            if ep in targets and targets[ep] is not None:
                if ep == 'survival':
                    # Handle survival data - targets[ep] is dict with 'time' and 'event'
                    if isinstance(targets[ep], dict) and 'time' in targets[ep] and 'event' in targets[ep]:
                        times = targets[ep]['time']
                        events = targets[ep]['event']
                        hazard_pred = predictions[ep].squeeze()
                        
                        ep_loss = self.endpoint_losses[ep](hazard_pred, times, events)
                        losses[ep] = ep_loss.item()
                        prediction_loss += ep_loss
                else:
                    # Handle traditional endpoints
                    valid_mask = targets[ep] != -1
                    if valid_mask.any():
                        valid_pred = predictions[ep][valid_mask]
                        valid_target = targets[ep][valid_mask]
                        
                        if ep in ['stage_t', 'stage_n']:
                            # Multiclass: targets are class indices, no unsqueeze needed
                            ep_loss = self.endpoint_losses[ep](valid_pred, valid_target)
                        else:
                            # Binary: targets need unsqueeze for BCE
                            ep_loss = self.endpoint_losses[ep](valid_pred, valid_target.unsqueeze(-1))
                        
                        losses[ep] = ep_loss.item()
                        prediction_loss += ep_loss
        
        if prediction_loss > 0:
            losses['prediction'] = prediction_loss.item()
            total_loss += self.prediction_weight * prediction_loss
        
        losses['total'] = total_loss.item()
        return total_loss, losses

class EndToEndLoss(nn.Module):
    """Loss for end-to-end model with mixed endpoint types"""
    def __init__(
        self, 
        class_weights: Dict[str, torch.Tensor],
        endpoints: List[str] = ['os6', 'os24'],
        device: str = 'cuda'
    ):
        super(EndToEndLoss, self).__init__()
        self.endpoints = endpoints
        self.device = device
        
        # Different losses for different endpoint types
        self.endpoint_losses = nn.ModuleDict()
        for ep in endpoints:
            if ep == 'survival':
                # Cox Proportional Hazards loss for survival analysis
                self.endpoint_losses[ep] = CoxPHLoss()
            elif ep in ['stage_t', 'stage_n']:
                # Multiclass classification
                weight = class_weights.get(ep, None)
                if weight is not None:
                    weight = weight.to(device)
                self.endpoint_losses[ep] = nn.CrossEntropyLoss(weight=weight)
            else:
                # Binary classification
                if ep in class_weights:
                    pos_weight = class_weights[ep][1] if len(class_weights[ep]) > 1 else class_weights[ep][0]
                    self.endpoint_losses[ep] = BalancedBCELoss(pos_weight.to(device))
                else:
                    self.endpoint_losses[ep] = nn.BCELoss()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        losses = {}
        total_loss = 0.0
        
        for ep in self.endpoints:
            if ep in predictions and ep in targets:
                if ep == 'survival':
                    # Handle survival data - targets[ep] is dict with 'time' and 'event'
                    if isinstance(targets[ep], dict) and 'time' in targets[ep] and 'event' in targets[ep]:
                        times = targets[ep]['time']
                        events = targets[ep]['event']
                        hazard_pred = predictions[ep].squeeze()
                        
                        ep_loss = self.endpoint_losses[ep](hazard_pred, times, events)
                        losses[ep] = ep_loss.item()
                        total_loss += ep_loss
                elif ep in ['stage_t', 'stage_n']:
                    # Multiclass: targets are class indices
                    ep_loss = self.endpoint_losses[ep](predictions[ep], targets[ep])
                    losses[ep] = ep_loss.item()
                    total_loss += ep_loss
                else:
                    # Binary: targets need unsqueeze
                    ep_loss = self.endpoint_losses[ep](predictions[ep], targets[ep].unsqueeze(-1))
                    losses[ep] = ep_loss.item()
                    total_loss += ep_loss
        
        losses['total'] = total_loss.item()
        return total_loss, losses

def prepare_batch_targets(batch: Dict, device: str, endpoints: List[str]) -> Dict[str, Optional[torch.Tensor]]:
    """Prepare batch targets with proper data types"""
    batch_size = len(batch['patient_id'])
    targets = {}
    
    for ep in endpoints:
        if ep == 'survival':
            # Handle survival data - collect time and event tensors
            times = []
            events = []
            ep_valid = False
            
            for i in range(batch_size):
                if batch['endpoints'][ep][i] is not None:
                    survival_data = batch['endpoints'][ep][i]
                    if isinstance(survival_data, dict) and 'time' in survival_data and 'event' in survival_data:
                        times.append(survival_data['time'])
                        events.append(survival_data['event'])
                        ep_valid = True
                    else:
                        # Invalid survival data structure
                        times.append(torch.tensor(-1.0, dtype=torch.float32))
                        events.append(torch.tensor(-1.0, dtype=torch.float32))
                else:
                    # Missing survival data
                    times.append(torch.tensor(-1.0, dtype=torch.float32))
                    events.append(torch.tensor(-1.0, dtype=torch.float32))
            
            if ep_valid:
                targets[ep] = {
                    'time': torch.stack(times).to(device),
                    'event': torch.stack(events).to(device)
                }
            else:
                targets[ep] = None
        else:
            # Handle traditional endpoints
            ep_targets = []
            ep_valid = False
            for i in range(batch_size):
                if batch['endpoints'][ep][i] is not None:
                    ep_targets.append(batch['endpoints'][ep][i])
                    ep_valid = True
                else:
                    # Use appropriate missing value marker with same dtype as valid values
                    if ep in ['stage_t', 'stage_n']:
                        ep_targets.append(torch.tensor(-1, dtype=torch.long))
                    else:
                        ep_targets.append(torch.tensor(-1.0, dtype=torch.float32))
            
            if ep_valid:
                # Ensure all tensors have consistent dtype before stacking
                if ep in ['stage_t', 'stage_n']:
                    # Convert all to long for multiclass
                    ep_targets = [t.long() if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.long) for t in ep_targets]
                else:
                    # Convert all to float for binary
                    ep_targets = [t.float() if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float32) for t in ep_targets]
                
                targets[ep] = torch.stack(ep_targets).to(device)
            else:
                targets[ep] = None
    
    return targets