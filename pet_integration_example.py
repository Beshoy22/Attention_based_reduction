#!/usr/bin/env python3
"""
Example script demonstrating PET integration with the attention-based neural network
"""

import torch
import numpy as np
import pickle
import json
import os
from datetime import datetime
from PIL import Image

def create_sample_pet_data():
    """Create sample PET data for demonstration"""
    
    # Create sample patient data with PET paths
    def create_patient_with_pet(patient_id, center, has_endpoints=True, has_pet=True):
        # Generate 176 random feature vectors of 512 dimensions each
        features = [torch.randn(512) for _ in range(176)]
        
        patient = {
            'filepath': f'/fake/path/{patient_id}.nii',
            'patient_id': patient_id,
            'features': features
        }
        
        if has_endpoints:
            patient['OS_6'] = np.random.choice([0, 1])
            patient['OS_24'] = float(np.random.choice([0, 1]))
        
        if has_pet:
            # Create sample PET images
            pet_dir = f'sample_pet_data/{patient_id}'
            os.makedirs(pet_dir, exist_ok=True)
            
            # Create synthetic PET images (vertically oriented)
            height, width = 400, 300  # Vertically oriented
            
            # Create coronal PET image
            coronal_img = np.random.rand(height, width) * 30  # Values 0-30
            coronal_img = coronal_img.astype(np.uint8)
            coronal_path = os.path.join(pet_dir, 'coronal.png')
            Image.fromarray(coronal_img).save(coronal_path)
            
            # Create sagittal PET image
            sagittal_img = np.random.rand(height, width) * 30  # Values 0-30
            sagittal_img = sagittal_img.astype(np.uint8)
            sagittal_path = os.path.join(pet_dir, 'sagittal.png')
            Image.fromarray(sagittal_img).save(sagittal_path)
            
            # Add paths to patient dictionary
            patient['coronal_png_path'] = coronal_path
            patient['sagittal_png_path'] = sagittal_path
        
        return patient
    
    # Create data for 3 centers
    centers = ['center_A', 'center_B', 'center_C']
    
    for center in centers:
        center_data = []
        
        # Create 50 patients per center
        for i in range(50):
            patient_id = f"{center.upper()}{i:06d}"
            has_endpoints = np.random.random() > 0.1  # 90% have endpoints
            has_pet = np.random.random() > 0.3  # 70% have PET data
            patient = create_patient_with_pet(patient_id, center, has_endpoints, has_pet)
            center_data.append(patient)
        
        # Save center data
        with open(f'{center}_with_pet.pkl', 'wb') as f:
            pickle.dump(center_data, f)
        
        print(f"Created {center}_with_pet.pkl with {len(center_data)} patients")
    
    # Create train/test split JSON
    all_patient_ids = []
    for center in centers:
        for i in range(50):
            all_patient_ids.append(f"{center.upper()}{i:06d}")
    
    # 80/20 split
    np.random.shuffle(all_patient_ids)
    split_idx = int(0.8 * len(all_patient_ids))
    
    split_data = {
        "TRAIN_SET": all_patient_ids[:split_idx],
        "TEST_SET": all_patient_ids[split_idx:]
    }
    
    with open('train_test_split_pet.json', 'w') as f:
        json.dump(split_data, f, indent=2)
    
    print(f"Created train_test_split_pet.json with {len(split_data['TRAIN_SET'])} train and {len(split_data['TEST_SET'])} test patients")

def run_pet_autoencoder_example():
    """Run autoencoder example with PET integration"""
    print("\n=== Running Autoencoder Example with PET Integration ===")
    
    # Multiplication fusion mode
    cmd_multiply = """
python main_autoencoder.py \\
    --pkl_files center_A_with_pet.pkl center_B_with_pet.pkl center_C_with_pet.pkl \\
    --train_test_split_json train_test_split_pet.json \\
    --model_type autoencoder \\
    --attention_k 16 \\
    --latent_dim 64 \\
    --encoder_layers 128 64 \\
    --predictor_layers 32 16 \\
    --epochs 5 \\
    --batch_size 8 \\
    --learning_rate 1e-3 \\
    --dropout_rate 0.2 \\
    --reconstruction_weight 1.0 \\
    --prediction_weight 1.0 \\
    --enable_pet \\
    --pet_fusion_mode multiply \\
    --pet_target_height 300 \\
    --save_dir ./pet_results_multiply
"""
    
    # Concatenation fusion mode
    cmd_concat = """
python main_autoencoder.py \\
    --pkl_files center_A_with_pet.pkl center_B_with_pet.pkl center_C_with_pet.pkl \\
    --train_test_split_json train_test_split_pet.json \\
    --model_type autoencoder \\
    --attention_k 16 \\
    --latent_dim 64 \\
    --encoder_layers 128 64 \\
    --predictor_layers 32 16 \\
    --epochs 5 \\
    --batch_size 8 \\
    --learning_rate 1e-3 \\
    --dropout_rate 0.2 \\
    --reconstruction_weight 1.0 \\
    --prediction_weight 1.0 \\
    --enable_pet \\
    --pet_fusion_mode concatenate \\
    --pet_target_height 300 \\
    --save_dir ./pet_results_concat
"""
    
    print("Command for multiplication fusion:")
    print(cmd_multiply)
    print("\nCommand for concatenation fusion:")
    print(cmd_concat)

def run_pet_endtoend_example():
    """Run end-to-end example with PET integration"""
    print("\n=== Running End-to-End Example with PET Integration ===")
    
    cmd = """
python main_endtoend.py \\
    --pkl_files center_A_with_pet.pkl center_B_with_pet.pkl center_C_with_pet.pkl \\
    --train_test_split_json train_test_split_pet.json \\
    --model_type endtoend \\
    --attention_k 16 \\
    --encoder_layers 128 64 32 \\
    --predictor_layers 32 16 \\
    --epochs 5 \\
    --batch_size 8 \\
    --learning_rate 1e-3 \\
    --dropout_rate 0.2 \\
    --enable_pet \\
    --pet_fusion_mode multiply \\
    --pet_target_height 300 \\
    --save_dir ./pet_results_endtoend
"""
    
    print("Command to run:")
    print(cmd)

def demonstrate_pet_model_usage():
    """Demonstrate direct PET model usage without training"""
    print("\n=== Demonstrating PET Model Usage ===")
    
    from utils.models import create_model, count_parameters
    
    # Create sample input
    batch_size = 4
    sample_ct_input = torch.randn(batch_size, 176, 512)
    
    # Create sample PET input (after preprocessing)
    pet_height, pet_width = 300, 400  # After resizing
    sample_pet_coronal = torch.randn(batch_size, 1, pet_height, pet_width)
    sample_pet_sagittal = torch.randn(batch_size, 1, pet_height, pet_width)
    
    # Test autoencoder model with PET (multiplication fusion)
    print("Creating autoencoder model with PET (multiplication fusion)...")
    autoencoder_mult = create_model(
        model_type='autoencoder',
        attention_k=16,
        encoder_layers=[128, 64],
        latent_dim=32,
        predictor_layers=[32, 16],
        dropout_rate=0.3,
        enable_pet=True,
        pet_fusion_mode='multiply',
        pet_input_height=pet_height,
        pet_input_width=pet_width
    )
    
    print(f"Autoencoder (mult) parameters: {count_parameters(autoencoder_mult):,}")
    
    # Forward pass
    autoencoder_mult.eval()
    with torch.no_grad():
        reconstructed, predictions = autoencoder_mult(sample_ct_input, sample_pet_coronal, sample_pet_sagittal)
        print(f"CT Input shape: {sample_ct_input.shape}")
        print(f"PET Input shapes: {sample_pet_coronal.shape}, {sample_pet_sagittal.shape}")
        print(f"Reconstructed shape: {reconstructed.shape}")
        print(f"Predictions: {list(predictions.keys())}")
        
        # Test without PET data (missing modality)
        reconstructed_no_pet, predictions_no_pet = autoencoder_mult(sample_ct_input)
        print(f"Without PET - Reconstructed shape: {reconstructed_no_pet.shape}")
    
    # Test autoencoder model with PET (concatenation fusion)
    print("\nCreating autoencoder model with PET (concatenation fusion)...")
    autoencoder_concat = create_model(
        model_type='autoencoder',
        attention_k=16,
        encoder_layers=[128, 64],
        latent_dim=32,
        predictor_layers=[32, 16],
        dropout_rate=0.3,
        enable_pet=True,
        pet_fusion_mode='concatenate',
        pet_input_height=pet_height,
        pet_input_width=pet_width
    )
    
    print(f"Autoencoder (concat) parameters: {count_parameters(autoencoder_concat):,}")
    
    # Forward pass
    autoencoder_concat.eval()
    with torch.no_grad():
        reconstructed, predictions = autoencoder_concat(sample_ct_input, sample_pet_coronal, sample_pet_sagittal)
        print(f"With PET - Reconstructed shape: {reconstructed.shape}")
        print(f"Expected shape change: 16 -> 18 vectors (16 CT + 2 PET)")
        
        # Test without PET data (missing modality)
        reconstructed_no_pet, predictions_no_pet = autoencoder_concat(sample_ct_input)
        print(f"Without PET - Reconstructed shape: {reconstructed_no_pet.shape}")
    
    # Test end-to-end model with PET
    print("\nCreating end-to-end model with PET...")
    endtoend_pet = create_model(
        model_type='endtoend',
        attention_k=16,
        encoder_layers=[128, 64, 32],
        predictor_layers=[32, 16],
        dropout_rate=0.3,
        enable_pet=True,
        pet_fusion_mode='multiply',
        pet_input_height=pet_height,
        pet_input_width=pet_width
    )
    
    print(f"End-to-end (PET) parameters: {count_parameters(endtoend_pet):,}")
    
    # Forward pass
    endtoend_pet.eval()
    with torch.no_grad():
        features, predictions = endtoend_pet(sample_ct_input, sample_pet_coronal, sample_pet_sagittal)
        print(f"Features shape: {features.shape}")
        print(f"Predictions: {list(predictions.keys())}")

def main():
    """Main demonstration function"""
    print("=== PET Integration Example ===")
    print("This script demonstrates PET scan integration with the attention-based neural network")
    
    # Set random seed
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create sample data with PET
    print("\n1. Creating sample data with PET...")
    create_sample_pet_data()
    
    # Demonstrate model usage
    demonstrate_pet_model_usage()
    
    # Show example commands
    run_pet_autoencoder_example()
    run_pet_endtoend_example()
    
    print("\n=== Key Features ===")
    print("✓ Dynamic PET image resizing based on training data aspect ratio")
    print("✓ Two fusion modes: multiply (element-wise) and concatenate (add 2 vectors)")
    print("✓ Handles missing PET data gracefully")
    print("✓ Convolutional encoder for PET feature extraction")
    print("✓ Early fusion approach after CT attention mechanism")
    
    print("\n=== Usage Summary ===")
    print("1. Add --enable_pet flag to enable PET integration")
    print("2. Use --pet_fusion_mode to choose 'multiply' or 'concatenate'")
    print("3. Set --pet_target_height for PET image resizing (default: 300)")
    print("4. Ensure your pkl files contain 'coronal_png_path' and 'sagittal_png_path' keys")
    
    print("\n=== Example Complete ===")
    print("Sample data files with PET have been created in the current directory.")
    print("To actually run training, execute the commands shown above.")
    
    # Cleanup option
    cleanup = input("\nRemove sample data files? (y/n): ")
    if cleanup.lower() == 'y':
        import shutil
        files_to_remove = ['center_A_with_pet.pkl', 'center_B_with_pet.pkl', 'center_C_with_pet.pkl', 'train_test_split_pet.json']
        for file in files_to_remove:
            if os.path.exists(file):
                os.remove(file)
                print(f"Removed {file}")
        if os.path.exists('sample_pet_data'):
            shutil.rmtree('sample_pet_data')
            print("Removed sample_pet_data directory")

if __name__ == "__main__":
    main()