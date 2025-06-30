#!/usr/bin/env python3
"""
Example usage script showing how to use the framework
"""

import torch
import numpy as np
import pickle
import json
import os
from datetime import datetime

def create_sample_data():
    """Create sample data for demonstration"""
    
    # Create sample patient data
    def create_patient(patient_id, center, has_endpoints=True):
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
        
        return patient
    
    # Create data for 3 centers
    centers = ['center_A', 'center_B', 'center_C']
    
    for center in centers:
        center_data = []
        
        # Create 50 patients per center (some without endpoints)
        for i in range(50):
            patient_id = f"{center.upper()}{i:06d}"
            has_endpoints = np.random.random() > 0.1  # 90% have endpoints
            patient = create_patient(patient_id, center, has_endpoints)
            center_data.append(patient)
        
        # Save center data
        with open(f'{center}.pkl', 'wb') as f:
            pickle.dump(center_data, f)
        
        print(f"Created {center}.pkl with {len(center_data)} patients")
    
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
    
    with open('train_test_split.json', 'w') as f:
        json.dump(split_data, f, indent=2)
    
    print(f"Created train_test_split.json with {len(split_data['TRAIN_SET'])} train and {len(split_data['TEST_SET'])} test patients")

def run_autoencoder_example():
    """Run autoencoder example"""
    print("\n=== Running Autoencoder Example ===")
    
    cmd = """
python main_autoencoder.py \\
    --pkl_files center_A.pkl center_B.pkl center_C.pkl \\
    --train_test_split_json train_test_split.json \\
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
    --save_dir ./example_results
"""
    
    print("Command to run:")
    print(cmd)
    
    # For demonstration, you would uncomment the following line to actually run it:
    # os.system(cmd.replace('\\\n', ''))

def run_endtoend_example():
    """Run end-to-end example"""
    print("\n=== Running End-to-End Example ===")
    
    cmd = """
python main_endtoend.py \\
    --pkl_files center_A.pkl center_B.pkl center_C.pkl \\
    --train_test_split_json train_test_split.json \\
    --model_type endtoend \\
    --attention_k 16 \\
    --encoder_layers 128 64 32 \\
    --predictor_layers 32 16 \\
    --epochs 5 \\
    --batch_size 8 \\
    --learning_rate 1e-3 \\
    --dropout_rate 0.2 \\
    --save_dir ./example_results
"""
    
    print("Command to run:")
    print(cmd)
    
    # For demonstration, you would uncomment the following line to actually run it:
    # os.system(cmd.replace('\\\n', ''))

def demonstrate_model_usage():
    """Demonstrate direct model usage without training"""
    print("\n=== Demonstrating Model Usage ===")
    
    from utils.models import create_model, count_parameters
    
    # Create sample input
    batch_size = 4
    sample_input = torch.randn(batch_size, 176, 512)
    
    # Test autoencoder model
    print("Creating autoencoder model...")
    autoencoder = create_model(
        model_type='autoencoder',
        attention_k=16,
        encoder_layers=[128, 64],
        latent_dim=32,
        predictor_layers=[32, 16],
        dropout_rate=0.3
    )
    
    print(f"Autoencoder parameters: {count_parameters(autoencoder):,}")
    
    # Forward pass
    autoencoder.eval()
    with torch.no_grad():
        reconstructed, os6_pred, os24_pred = autoencoder(sample_input)
        print(f"Input shape: {sample_input.shape}")
        print(f"Reconstructed shape: {reconstructed.shape}")
        print(f"OS6 predictions shape: {os6_pred.shape}")
        print(f"OS24 predictions shape: {os24_pred.shape}")
    
    # Test end-to-end model
    print("\nCreating end-to-end model...")
    endtoend = create_model(
        model_type='endtoend',
        attention_k=16,
        encoder_layers=[128, 64, 32],
        predictor_layers=[32, 16],
        dropout_rate=0.3
    )
    
    print(f"End-to-end parameters: {count_parameters(endtoend):,}")
    
    # Forward pass
    endtoend.eval()
    with torch.no_grad():
        features, os6_pred, os24_pred = endtoend(sample_input)
        print(f"Input shape: {sample_input.shape}")
        print(f"Reduced features shape: {features.shape}")
        print(f"OS6 predictions shape: {os6_pred.shape}")
        print(f"OS24 predictions shape: {os24_pred.shape}")

def main():
    """Main example function"""
    print("=== Attention-Based Neural Networks Example ===")
    print("This script demonstrates the usage of the framework")
    
    # Set random seed
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create sample data
    print("\n1. Creating sample data...")
    create_sample_data()
    
    # Demonstrate model usage
    demonstrate_model_usage()
    
    # Show example commands
    run_autoencoder_example()
    run_endtoend_example()
    
    print("\n=== Example Complete ===")
    print("To actually run training, uncomment the os.system() calls in the example functions above.")
    print("The sample data files have been created in the current directory.")
    
    # Cleanup option
    cleanup = input("\nRemove sample data files? (y/n): ")
    if cleanup.lower() == 'y':
        for file in ['center_A.pkl', 'center_B.pkl', 'center_C.pkl', 'train_test_split.json']:
            if os.path.exists(file):
                os.remove(file)
                print(f"Removed {file}")

if __name__ == "__main__":
    main()