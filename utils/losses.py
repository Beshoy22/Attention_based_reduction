import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

class BalancedBCELoss(nn.Module):
    """Balanced Binary Cross Entropy Loss"""
    def __init__(self, pos_weight: torch.Tensor):
        super(BalancedBCELoss, self).__init__()
        self.pos_weight = pos_weight
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy(predictions, targets, weight=self.pos_weight, reduction='mean')

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
            if ep in ['stage_t', 'stage_n']:
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
            if ep in ['stage_t', 'stage_n']:
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
                if ep in ['stage_t', 'stage_n']:
                    # Multiclass: targets are class indices
                    ep_loss = self.endpoint_losses[ep](predictions[ep], targets[ep])
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