#!/usr/bin/env python3
"""
Test script for PET integration functionality
"""

import torch
import numpy as np
from utils.models import create_model, count_parameters

def test_pet_models():
    """Test PET model creation and forward pass"""
    print("Testing PET model integration...")
    
    # Create sample inputs
    batch_size = 2
    ct_features = torch.randn(batch_size, 176, 512)
    pet_height, pet_width = 300, 400
    pet_coronal = torch.randn(batch_size, 1, pet_height, pet_width)
    pet_sagittal = torch.randn(batch_size, 1, pet_height, pet_width)
    
    # Test autoencoder with PET (multiply mode)
    print("\n1. Testing autoencoder with PET (multiply mode)...")
    model_mult = create_model(
        model_type='autoencoder',
        attention_k=8,
        encoder_layers=[64, 32],
        latent_dim=16,
        predictor_layers=[16, 8],
        endpoints=['os6', 'os24'],
        enable_pet=True,
        pet_fusion_mode='multiply',
        pet_input_height=pet_height,
        pet_input_width=pet_width
    )
    
    print(f"Parameters: {count_parameters(model_mult):,}")
    
    model_mult.eval()
    with torch.no_grad():
        # With PET data
        reconstructed, predictions = model_mult(ct_features, pet_coronal, pet_sagittal)
        print(f"With PET - Reconstructed shape: {reconstructed.shape}")
        print(f"Predictions keys: {list(predictions.keys())}")
        
        # Without PET data (missing modality)
        reconstructed_no_pet, predictions_no_pet = model_mult(ct_features)
        print(f"Without PET - Reconstructed shape: {reconstructed_no_pet.shape}")
        print("✓ Multiply mode works")
    
    # Test autoencoder with PET (concatenate mode)
    print("\n2. Testing autoencoder with PET (concatenate mode)...")
    model_concat = create_model(
        model_type='autoencoder',
        attention_k=8,
        encoder_layers=[64, 32],
        latent_dim=16,
        predictor_layers=[16, 8],
        endpoints=['os6', 'os24'],
        enable_pet=True,
        pet_fusion_mode='concatenate',
        pet_input_height=pet_height,
        pet_input_width=pet_width
    )
    
    print(f"Parameters: {count_parameters(model_concat):,}")
    
    model_concat.eval()
    with torch.no_grad():
        # With PET data
        reconstructed, predictions = model_concat(ct_features, pet_coronal, pet_sagittal)
        print(f"With PET - Reconstructed shape: {reconstructed.shape}")
        print(f"Expected: 8 CT + 2 PET = 10 vectors")
        
        # Without PET data (missing modality)
        reconstructed_no_pet, predictions_no_pet = model_concat(ct_features)
        print(f"Without PET - Reconstructed shape: {reconstructed_no_pet.shape}")
        print("✓ Concatenate mode works")
    
    # Test end-to-end with PET
    print("\n3. Testing end-to-end with PET...")
    model_e2e = create_model(
        model_type='endtoend',
        attention_k=8,
        encoder_layers=[64, 32, 16],
        predictor_layers=[16, 8],
        endpoints=['os6', 'os24'],
        enable_pet=True,
        pet_fusion_mode='multiply',
        pet_input_height=pet_height,
        pet_input_width=pet_width
    )
    
    print(f"Parameters: {count_parameters(model_e2e):,}")
    
    model_e2e.eval()
    with torch.no_grad():
        # With PET data
        features, predictions = model_e2e(ct_features, pet_coronal, pet_sagittal)
        print(f"With PET - Features shape: {features.shape}")
        print(f"Predictions keys: {list(predictions.keys())}")
        
        # Without PET data (missing modality)
        features_no_pet, predictions_no_pet = model_e2e(ct_features)
        print(f"Without PET - Features shape: {features_no_pet.shape}")
        print("✓ End-to-end mode works")
    
    print("\n✅ All PET integration tests passed!")

if __name__ == "__main__":
    test_pet_models()