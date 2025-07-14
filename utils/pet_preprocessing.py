import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
import warnings

try:
    from PIL import Image
except ImportError:
    print("Warning: PIL not available, some features may not work")
    Image = None

try:
    import cv2
except ImportError:
    print("Warning: OpenCV not available, using numpy for resizing")
    cv2 = None

class PETPreprocessor:
    """
    Preprocessor for PET images with dynamic resizing based on training set statistics
    """
    
    def __init__(self, target_height: int = 300, aspect_ratio: Optional[float] = None):
        """
        Initialize PET preprocessor
        
        Args:
            target_height: Target height for resizing (default: 300)
            aspect_ratio: Target aspect ratio (width/height). If None, will be computed from training data
        """
        self.target_height = target_height
        self.aspect_ratio = aspect_ratio
        self.target_width = None
        
        if aspect_ratio is not None:
            self.target_width = int(target_height * aspect_ratio)
    
    def compute_aspect_ratio_from_data(self, data: List[Dict]) -> float:
        """
        Compute average aspect ratio from training data
        
        Args:
            data: List of patient dictionaries containing PET paths
            
        Returns:
            Average aspect ratio (width/height)
        """
        ratios = []
        
        for patient in data:
            if 'coronal_png_path' in patient and 'sagittal_png_path' in patient:
                # Load coronal image to get dimensions
                try:
                    coronal_img = Image.open(patient['coronal_png_path'])
                    width, height = coronal_img.size
                    ratios.append(width / height)
                except Exception as e:
                    warnings.warn(f"Could not load PET image for patient {patient.get('patient_id', 'unknown')}: {e}")
                    continue
                    
                # Load sagittal image to get dimensions
                try:
                    sagittal_img = Image.open(patient['sagittal_png_path'])
                    width, height = sagittal_img.size
                    ratios.append(width / height)
                except Exception as e:
                    warnings.warn(f"Could not load PET image for patient {patient.get('patient_id', 'unknown')}: {e}")
                    continue
        
        if not ratios:
            warnings.warn("No valid PET images found for aspect ratio computation, using default 1.0")
            return 1.0
        
        avg_ratio = np.mean(ratios)
        print(f"Computed average aspect ratio from {len(ratios)} PET images: {avg_ratio:.3f}")
        return avg_ratio
    
    def set_target_dimensions(self, aspect_ratio: float):
        """
        Set target dimensions based on computed aspect ratio
        
        Args:
            aspect_ratio: Target aspect ratio (width/height)
        """
        self.aspect_ratio = aspect_ratio
        self.target_width = int(self.target_height * aspect_ratio)
        print(f"Set target PET dimensions: {self.target_width} x {self.target_height}")
    
    def preprocess_pet_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess a single PET image
        
        Args:
            image_path: Path to PET image
            
        Returns:
            Preprocessed tensor of shape (1, target_height, target_width)
        """
        try:
            # Load image
            img = Image.open(image_path).convert('L')  # Convert to grayscale
            img_array = np.array(img, dtype=np.float32)
            
            # Clip pixel values to [0, 30]
            img_array = np.clip(img_array, 0, 30)
            
            # Normalize to [0, 1]
            img_array = img_array / 30.0
            
            # Resize to target dimensions
            if self.target_width is None:
                raise ValueError("Target dimensions not set. Call set_target_dimensions() first.")
            
            if cv2 is not None:
                img_resized = cv2.resize(img_array, (self.target_width, self.target_height), 
                                       interpolation=cv2.INTER_LINEAR)
            else:
                # Fallback to numpy/PIL if cv2 not available
                if Image is not None:
                    img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
                    img_pil_resized = img_pil.resize((self.target_width, self.target_height), Image.LANCZOS)
                    img_resized = np.array(img_pil_resized, dtype=np.float32) / 255.0
                else:
                    # Very basic resize using numpy (not recommended for production)
                    from scipy.ndimage import zoom
                    h_factor = self.target_height / img_array.shape[0]
                    w_factor = self.target_width / img_array.shape[1]
                    img_resized = zoom(img_array, (h_factor, w_factor), order=1)
            
            # Convert to tensor and add channel dimension
            tensor = torch.from_numpy(img_resized).unsqueeze(0)  # Shape: (1, H, W)
            
            return tensor
            
        except Exception as e:
            warnings.warn(f"Error processing PET image {image_path}: {e}")
            # Return zero tensor if processing fails
            if self.target_width is None:
                raise ValueError("Target dimensions not set. Call set_target_dimensions() first.")
            return torch.zeros(1, self.target_height, self.target_width)
    
    def preprocess_pet_pair(self, coronal_path: str, sagittal_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess coronal and sagittal PET images
        
        Args:
            coronal_path: Path to coronal PET image
            sagittal_path: Path to sagittal PET image
            
        Returns:
            Tuple of preprocessed tensors (coronal, sagittal)
        """
        coronal = self.preprocess_pet_image(coronal_path)
        sagittal = self.preprocess_pet_image(sagittal_path)
        
        return coronal, sagittal

class PETEncoder(nn.Module):
    """
    PET encoder with convolutional layers, batch normalization, and ReLU activations
    """
    
    def __init__(self, input_height: int = 300, input_width: int = 300, output_dim: int = 512):
        """
        Initialize PET encoder
        
        Args:
            input_height: Height of input PET images
            input_width: Width of input PET images
            output_dim: Output feature dimension (default: 512 to match CT features)
        """
        super(PETEncoder, self).__init__()
        
        self.input_height = input_height
        self.input_width = input_width
        self.output_dim = output_dim
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # First conv layer
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/2, W/2
            
            # Second conv layer
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/4, W/4
            
            # Third conv layer
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8))  # Fixed 8x8 output
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through PET encoder
        
        Args:
            x: Input tensor of shape (batch_size, 1, height, width)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class MultiModalPETFusion(nn.Module):
    """
    Multi-modal fusion for PET and CT features with missing modality handling
    """
    
    def __init__(self, 
                 fusion_mode: str = 'multiply',  # 'multiply' or 'concatenate'
                 ct_feature_dim: int = 512,
                 pet_feature_dim: int = 512,
                 attention_k: int = 32):
        """
        Initialize multi-modal fusion module
        
        Args:
            fusion_mode: 'multiply' or 'concatenate'
            ct_feature_dim: Dimension of CT features
            pet_feature_dim: Dimension of PET features
            attention_k: Number of attention vectors
        """
        super(MultiModalPETFusion, self).__init__()
        
        self.fusion_mode = fusion_mode
        self.ct_feature_dim = ct_feature_dim
        self.pet_feature_dim = pet_feature_dim
        self.attention_k = attention_k
        
        # Ensure PET features match CT feature dimension for multiplication
        if fusion_mode == 'multiply' and pet_feature_dim != ct_feature_dim:
            self.pet_projection = nn.Linear(pet_feature_dim, ct_feature_dim)
        else:
            self.pet_projection = nn.Identity()
        
        # For concatenation mode, we add 2 additional vectors to the k vectors
        if fusion_mode == 'concatenate':
            self.output_k = attention_k + 2
        else:
            self.output_k = attention_k
    
    def forward(self, 
                ct_features: torch.Tensor, 
                pet_coronal: Optional[torch.Tensor] = None,
                pet_sagittal: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for multi-modal fusion
        
        Args:
            ct_features: CT attention features of shape (batch_size, k, 512)
            pet_coronal: PET coronal features of shape (batch_size, pet_feature_dim)
            pet_sagittal: PET sagittal features of shape (batch_size, pet_feature_dim)
            
        Returns:
            Fused features of shape (batch_size, output_k, 512)
        """
        batch_size = ct_features.size(0)
        
        if self.fusion_mode == 'multiply':
            # Multiplication fusion
            if pet_coronal is not None and pet_sagittal is not None:
                # Project PET features to match CT dimension if needed
                pet_cor_proj = self.pet_projection(pet_coronal)  # (batch_size, 512)
                pet_sag_proj = self.pet_projection(pet_sagittal)  # (batch_size, 512)
                
                # Average the two PET views
                pet_features = (pet_cor_proj + pet_sag_proj) / 2  # (batch_size, 512)
                
                # Expand to match CT features shape and multiply
                pet_features_expanded = pet_features.unsqueeze(1).expand(-1, self.attention_k, -1)
                fused_features = ct_features * pet_features_expanded
                
            else:
                # No PET data available, return CT features unchanged
                fused_features = ct_features
            
            return fused_features
        
        elif self.fusion_mode == 'concatenate':
            # Concatenation fusion
            if pet_coronal is not None and pet_sagittal is not None:
                # Project PET features to match CT dimension if needed
                pet_cor_proj = self.pet_projection(pet_coronal)  # (batch_size, 512)
                pet_sag_proj = self.pet_projection(pet_sagittal)  # (batch_size, 512)
                
                # Add PET features as additional vectors
                pet_cor_vector = pet_cor_proj.unsqueeze(1)  # (batch_size, 1, 512)
                pet_sag_vector = pet_sag_proj.unsqueeze(1)  # (batch_size, 1, 512)
                
                # Concatenate along the vector dimension
                fused_features = torch.cat([ct_features, pet_cor_vector, pet_sag_vector], dim=1)
                
            else:
                # No PET data available, pad with zeros
                zero_padding = torch.zeros(batch_size, 2, self.ct_feature_dim, 
                                         device=ct_features.device, dtype=ct_features.dtype)
                fused_features = torch.cat([ct_features, zero_padding], dim=1)
            
            return fused_features
        
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")

def compute_pet_aspect_ratio(data: List[Dict]) -> float:
    """
    Utility function to compute aspect ratio from patient data
    
    Args:
        data: List of patient dictionaries
        
    Returns:
        Average aspect ratio
    """
    preprocessor = PETPreprocessor()
    return preprocessor.compute_aspect_ratio_from_data(data)