import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class AttentionMechanism(nn.Module):
    """
    Simple attention mechanism that collapses 176 vectors to k vectors
    using learnable softmax weights and sectioned grouping
    """
    def __init__(self, input_dim: int = 512, k: int = 32):
        super(AttentionMechanism, self).__init__()
        self.input_dim = input_dim
        self.k = k
        self.num_patches = 176
        self.patches_per_group = self.num_patches // k
        
        # Learnable attention weights for each group
        self.attention_weights = nn.Parameter(torch.randn(k, self.patches_per_group))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, 176, 512)
        
        Returns:
            Output tensor of shape (batch_size, k, 512)
        """
        batch_size = x.size(0)
        
        # Reshape input to group patches
        # x shape: (batch_size, 176, 512) -> (batch_size, k, patches_per_group, 512)
        x_grouped = x.view(batch_size, self.k, self.patches_per_group, self.input_dim)
        
        # Apply softmax to attention weights for each group
        attention_probs = F.softmax(self.attention_weights, dim=1)  # Shape: (k, patches_per_group)
        
        # Apply attention weights
        # attention_probs: (k, patches_per_group) -> (1, k, patches_per_group, 1)
        attention_probs = attention_probs.unsqueeze(0).unsqueeze(-1)
        
        # Weighted sum within each group
        # x_grouped: (batch_size, k, patches_per_group, 512)
        # attention_probs: (1, k, patches_per_group, 1)
        attended_features = (x_grouped * attention_probs).sum(dim=2)  # Shape: (batch_size, k, 512)
        
        return attended_features

class MLPBlock(nn.Module):
    """MLP block with batch normalization and dropout"""
    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float = 0.3):
        super(MLPBlock, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

class Encoder(nn.Module):
    """Encoder network with configurable layers"""
    def __init__(self, input_dim: int, layer_dims: List[int], dropout_rate: float = 0.3):
        super(Encoder, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        for dim in layer_dims[:-1]:
            layers.append(MLPBlock(current_dim, dim, dropout_rate))
            current_dim = dim
        
        # Final layer without dropout and activation for latent representation
        layers.append(nn.Linear(current_dim, layer_dims[-1]))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class Decoder(nn.Module):
    """Decoder network (symmetric to encoder)"""
    def __init__(self, latent_dim: int, layer_dims: List[int], output_dim: int, dropout_rate: float = 0.3):
        super(Decoder, self).__init__()
        
        # Reverse the layer dimensions for symmetric architecture
        reversed_dims = [latent_dim] + layer_dims[::-1] + [output_dim]
        
        layers = []
        for i in range(len(reversed_dims) - 2):
            layers.append(MLPBlock(reversed_dims[i], reversed_dims[i+1], dropout_rate))
        
        # Final reconstruction layer
        layers.append(nn.Linear(reversed_dims[-2], reversed_dims[-1]))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class Predictor(nn.Module):
    """Predictor network for endpoints with different output types"""
    def __init__(self, input_dim: int, layer_dims: List[int], endpoints: List[str], dropout_rate: float = 0.3):
        super(Predictor, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        for dim in layer_dims:
            layers.append(MLPBlock(current_dim, dim, dropout_rate))
            current_dim = dim
        
        self.feature_network = nn.Sequential(*layers)
        
        # Output heads for each endpoint with appropriate dimensions
        self.endpoint_heads = nn.ModuleDict()
        endpoint_dims = {
            'os6': 1, 'os24': 1,  # Binary
            'stage_t': 5,  # 0,1,2,3,4
            'stage_n': 4,  # 0,1,2,3  
            'stage_m': 1,  # Binary
            'survival': 1  # Hazard prediction (log hazard ratio)
        }
        
        for ep in endpoints:
            self.endpoint_heads[ep] = nn.Linear(current_dim, endpoint_dims[ep])
    
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.feature_network(x)
        predictions = {}
        for ep, head in self.endpoint_heads.items():
            pred = head(features)
            # Apply appropriate activation functions
            if ep in ['os6', 'os24', 'stage_m']:
                # Binary endpoints: apply sigmoid
                predictions[ep] = torch.sigmoid(pred)
            elif ep == 'survival':
                # Survival endpoint: no activation (log hazard ratio)
                predictions[ep] = pred
            else:  # stage_t, stage_n - multiclass, no sigmoid
                predictions[ep] = pred
        return predictions

class AutoEncoderModel(nn.Module):
    """
    Autoencoder model with attention mechanism and prediction heads
    """
    def __init__(
        self,
        attention_k: int = 32,
        encoder_layers: List[int] = [256, 128],
        latent_dim: int = 64,
        predictor_layers: List[int] = [64, 32],
        endpoints: List[str] = ['os6', 'os24'],
        dropout_rate: float = 0.3,
        reconstruct_all: bool = False
    ):
        super(AutoEncoderModel, self).__init__()
        
        self.attention_k = attention_k
        self.latent_dim = latent_dim
        self.reconstruct_all = reconstruct_all
        
        # Attention mechanism
        self.attention = AttentionMechanism(input_dim=512, k=attention_k)
        
        # Flatten attention output for encoder input
        encoder_input_dim = attention_k * 512
        
        # Encoder
        self.encoder = Encoder(encoder_input_dim, encoder_layers + [latent_dim], dropout_rate)
        
        # Decoder - output size depends on reconstruction mode
        if reconstruct_all:
            # Reconstruct entire input: 176 * 512
            decoder_output_dim = 176 * 512
        else:
            # Reconstruct only attended features: k * 512
            decoder_output_dim = encoder_input_dim
        
        self.decoder = Decoder(latent_dim, encoder_layers, decoder_output_dim, dropout_rate)
        
        # Predictor
        self.predictor = Predictor(latent_dim, predictor_layers, endpoints, dropout_rate)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Apply attention mechanism
        attended_features = self.attention(x)  # Shape: (batch_size, k, 512)
        
        # Flatten for encoder
        batch_size = attended_features.size(0)
        flattened = attended_features.view(batch_size, -1)  # Shape: (batch_size, k*512)
        
        # Encode to latent space
        latent = self.encoder(flattened)  # Shape: (batch_size, latent_dim)
        
        # Decode for reconstruction
        reconstructed_flat = self.decoder(latent)  # Shape: (batch_size, output_dim)
        
        if self.reconstruct_all:
            # Reconstruct entire input: (batch_size, 176, 512)
            reconstructed = reconstructed_flat.view(batch_size, 176, 512)
        else:
            # Reconstruct only attended features: (batch_size, k, 512)
            reconstructed = reconstructed_flat.view(batch_size, self.attention_k, 512)
        
        # Predict endpoints
        predictions = self.predictor(latent)
        
        return reconstructed, predictions

class EndToEndModel(nn.Module):
    """
    End-to-end model without reconstruction
    """
    def __init__(
        self,
        attention_k: int = 32,
        encoder_layers: List[int] = [256, 128, 64],
        predictor_layers: List[int] = [64, 32],
        endpoints: List[str] = ['os6', 'os24'],
        dropout_rate: float = 0.3
    ):
        super(EndToEndModel, self).__init__()
        
        self.attention_k = attention_k
        
        # Attention mechanism
        self.attention = AttentionMechanism(input_dim=512, k=attention_k)
        
        # Feature encoder
        encoder_input_dim = attention_k * 512
        self.encoder = Encoder(encoder_input_dim, encoder_layers, dropout_rate)
        
        # Predictor (using second-to-last layer as reduced features)
        self.predictor = Predictor(encoder_layers[-1], predictor_layers, endpoints, dropout_rate)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Apply attention mechanism
        attended_features = self.attention(x)  # Shape: (batch_size, k, 512)
        
        # Flatten for encoder
        batch_size = attended_features.size(0)
        flattened = attended_features.view(batch_size, -1)  # Shape: (batch_size, k*512)
        
        # Encode to reduced features
        features = self.encoder(flattened)  # Shape: (batch_size, encoder_layers[-1])
        
        # Predict endpoints
        predictions = self.predictor(features)
        
        return features, predictions

class SurvivalModel(nn.Module):
    """
    Survival analysis model with attention mechanism
    Specialized for survival analysis with Cox Proportional Hazards
    """
    def __init__(
        self,
        attention_k: int = 32,
        encoder_layers: List[int] = [256, 128, 64],
        dropout_rate: float = 0.3,
        section_size: int = 22  # 176 / 8 sections
    ):
        super(SurvivalModel, self).__init__()
        
        self.attention_k = attention_k
        self.section_size = section_size
        
        # Attention mechanism to reduce from 176 to k vectors
        self.attention = AttentionMechanism(
            input_dim=512, 
            k=attention_k, 
            section_size=section_size
        )
        
        # Encoder for feature reduction
        input_dim = attention_k * 512
        self.encoder = Encoder(input_dim, encoder_layers, dropout_rate)
        
        # Survival prediction head - single output for log hazard ratio
        encoder_output_dim = encoder_layers[-1]
        self.hazard_head = nn.Sequential(
            nn.Linear(encoder_output_dim, encoder_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(encoder_output_dim // 2, 1)  # Single hazard output
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for survival prediction
        
        Args:
            x: Input tensor of shape (batch_size, 176, 512)
            
        Returns:
            Log hazard ratios of shape (batch_size, 1)
        """
        # Apply attention mechanism
        attended = self.attention(x)  # Shape: (batch_size, k, 512)
        
        # Flatten for encoder
        batch_size = attended.size(0)
        flattened = attended.view(batch_size, -1)  # Shape: (batch_size, k * 512)
        
        # Encode to reduced features
        features = self.encoder(flattened)  # Shape: (batch_size, encoder_layers[-1])
        
        # Predict log hazard ratio
        log_hazard = self.hazard_head(features)  # Shape: (batch_size, 1)
        
        return log_hazard

def create_model(model_type: str, endpoints: List[str] = ['os6', 'os24'], **kwargs) -> nn.Module:
    """Factory function to create models"""
    if model_type == 'autoencoder':
        return AutoEncoderModel(endpoints=endpoints, **kwargs)
    elif model_type == 'endtoend':
        return EndToEndModel(endpoints=endpoints, **kwargs)
    elif model_type == 'survival':
        return SurvivalModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)