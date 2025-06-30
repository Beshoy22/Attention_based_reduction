#!/usr/bin/env python3
"""
Test script to verify framework functionality
"""

import torch
import numpy as np
import tempfile
import os
import json
import pickle
import sys
from typing import Dict, List

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported correctly"""
    print("Testing imports...")
    
    try:
        from utils.models import create_model, AutoEncoderModel, EndToEndModel, count_parameters
        from utils.dataloader import create_data_loaders, MedicalImagingDataset
        from utils.losses import CombinedLoss, EndToEndLoss, BalancedBCELoss
        from utils.optimizers import OptimizerManager, create_optimizer, create_scheduler
        from utils.train_loop import train_model, evaluate_model
        from utils.visualization import TrainingVisualizer, AttentionVisualizer, FeatureVisualizer
        from config import Config, load_train_test_split
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def create_test_data(temp_dir: str) -> tuple:
    """Create synthetic test data"""
    print("Creating test data...")
    
    # Create sample patient data
    def create_patient(patient_id: str, center: str, has_endpoints: bool = True):
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
    
    # Create data for 2 centers
    centers = ['center_A', 'center_B']
    pkl_files = []
    
    for center in centers:
        center_data = []
        for i in range(20):  # Small dataset for testing
            patient_id = f"{center.upper()}{i:03d}"
            has_endpoints = np.random.random() > 0.1  # 90% have endpoints
            patient = create_patient(patient_id, center, has_endpoints)
            center_data.append(patient)
        
        pkl_file = os.path.join(temp_dir, f'{center}.pkl')
        with open(pkl_file, 'wb') as f:
            pickle.dump(center_data, f)
        pkl_files.append(pkl_file)
    
    # Create train/test split
    all_patient_ids = []
    for center in centers:
        for i in range(20):
            all_patient_ids.append(f"{center.upper()}{i:03d}")
    
    np.random.shuffle(all_patient_ids)
    split_idx = int(0.8 * len(all_patient_ids))
    
    split_data = {
        "TRAIN_SET": all_patient_ids[:split_idx],
        "TEST_SET": all_patient_ids[split_idx:]
    }
    
    split_file = os.path.join(temp_dir, 'train_test_split.json')
    with open(split_file, 'w') as f:
        json.dump(split_data, f)
    
    print(f"✓ Created test data with {len(all_patient_ids)} patients")
    return pkl_files, split_file

def test_model_creation():
    """Test model creation and basic functionality"""
    print("Testing model creation...")
    
    try:
        from utils.models import create_model, count_parameters
        
        # Test autoencoder
        autoencoder = create_model(
            model_type='autoencoder',
            attention_k=8,
            encoder_layers=[64, 32],
            latent_dim=16,
            predictor_layers=[16, 8],
            dropout_rate=0.2
        )
        
        param_count = count_parameters(autoencoder)
        print(f"✓ Autoencoder created with {param_count:,} parameters")
        
        # Test end-to-end
        endtoend = create_model(
            model_type='endtoend',
            attention_k=8,
            encoder_layers=[64, 32, 16],
            predictor_layers=[16, 8],
            dropout_rate=0.2
        )
        
        param_count = count_parameters(endtoend)
        print(f"✓ End-to-end model created with {param_count:,} parameters")
        
        return True
    except Exception as e:
        print(f"✗ Model creation error: {e}")
        return False

def test_forward_pass():
    """Test forward pass through models"""
    print("Testing forward pass...")
    
    try:
        from utils.models import create_model
        
        # Create test input
        batch_size = 4
        test_input = torch.randn(batch_size, 176, 512)
        
        # Test autoencoder forward pass
        autoencoder = create_model(
            model_type='autoencoder',
            attention_k=8,
            encoder_layers=[64, 32],
            latent_dim=16,
            predictor_layers=[16, 8],
            dropout_rate=0.2
        )
        
        autoencoder.eval()
        with torch.no_grad():
            reconstructed, os6_pred, os24_pred = autoencoder(test_input)
            
        print(f"✓ Autoencoder forward pass - Input: {test_input.shape}, "
              f"Reconstructed: {reconstructed.shape}, "
              f"OS6: {os6_pred.shape}, OS24: {os24_pred.shape}")
        
        # Test end-to-end forward pass
        endtoend = create_model(
            model_type='endtoend',
            attention_k=8,
            encoder_layers=[64, 32, 16],
            predictor_layers=[16, 8],
            dropout_rate=0.2
        )
        
        endtoend.eval()
        with torch.no_grad():
            features, os6_pred, os24_pred = endtoend(test_input)
            
        print(f"✓ End-to-end forward pass - Input: {test_input.shape}, "
              f"Features: {features.shape}, "
              f"OS6: {os6_pred.shape}, OS24: {os24_pred.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Forward pass error: {e}")
        return False

def test_data_loading(pkl_files: List[str], split_file: str):
    """Test data loading functionality"""
    print("Testing data loading...")
    
    try:
        from utils.dataloader import create_data_loaders
        from config import load_train_test_split
        
        split_data = load_train_test_split(split_file)
        
        # Test autoencoder data loading
        train_loader, val_loader, test_loader, class_weights = create_data_loaders(
            pkl_files=pkl_files,
            split_json=split_data,
            val_split=0.3,
            batch_size=4,
            num_workers=0,
            model_type='autoencoder',
            random_state=42
        )
        
        print(f"✓ Autoencoder data loaders created - Train: {len(train_loader.dataset)}, "
              f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
        
        # Test end-to-end data loading
        train_loader, val_loader, test_loader, class_weights = create_data_loaders(
            pkl_files=pkl_files,
            split_json=split_data,
            val_split=0.3,
            batch_size=4,
            num_workers=0,
            model_type='endtoend',
            random_state=42
        )
        
        print(f"✓ End-to-end data loaders created - Train: {len(train_loader.dataset)}, "
              f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
        
        return True
    except Exception as e:
        print(f"✗ Data loading error: {e}")
        return False

def test_loss_functions():
    """Test loss function computations"""
    print("Testing loss functions...")
    
    try:
        from utils.losses import CombinedLoss, EndToEndLoss, prepare_batch_targets
        
        # Create dummy data
        batch_size = 4
        attended_features = torch.randn(batch_size, 8, 512)
        reconstructed = torch.randn(batch_size, 8, 512)
        os6_pred = torch.sigmoid(torch.randn(batch_size, 1))
        os24_pred = torch.sigmoid(torch.randn(batch_size, 1))
        os6_true = torch.randint(0, 2, (batch_size,)).float()
        os24_true = torch.randint(0, 2, (batch_size,)).float()
        
        # Test combined loss
        class_weights = {'os6': torch.tensor([1.0, 1.0]), 'os24': torch.tensor([1.0, 1.0])}
        combined_loss = CombinedLoss(class_weights, device='cpu')
        
        loss, loss_dict = combined_loss(
            reconstructed, attended_features,
            os6_pred, os6_true,
            os24_pred, os24_true
        )
        
        print(f"✓ Combined loss computed: {loss.item():.4f}")
        
        # Test end-to-end loss
        endtoend_loss = EndToEndLoss(class_weights, device='cpu')
        loss, loss_dict = endtoend_loss(os6_pred, os6_true, os24_pred, os24_true)
        
        print(f"✓ End-to-end loss computed: {loss.item():.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Loss function error: {e}")
        return False

def test_optimizer_manager():
    """Test optimizer manager functionality"""
    print("Testing optimizer manager...")
    
    try:
        from utils.models import create_model
        from utils.optimizers import OptimizerManager
        
        model = create_model(
            model_type='autoencoder',
            attention_k=8,
            encoder_layers=[32, 16],
            latent_dim=8,
            predictor_layers=[8, 4],
            dropout_rate=0.2
        )
        
        optimizer_config = {
            'optimizer_type': 'adam',
            'learning_rate': 1e-3,
            'weight_decay': 1e-4
        }
        
        scheduler_config = {
            'scheduler_type': 'plateau',
            'patience': 5,
            'factor': 0.5
        }
        
        early_stopping_config = {
            'patience': 10,
            'min_delta': 1e-4
        }
        
        optimizer_manager = OptimizerManager(
            model=model,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            early_stopping_config=early_stopping_config
        )
        
        print(f"✓ Optimizer manager created with LR: {optimizer_manager.get_current_lr():.2e}")
        
        return True
    except Exception as e:
        print(f"✗ Optimizer manager error: {e}")
        return False

def test_mini_training(pkl_files: List[str], split_file: str, temp_dir: str):
    """Test a mini training loop"""
    print("Testing mini training loop...")
    
    try:
        from utils.dataloader import create_data_loaders
        from utils.models import create_model
        from utils.losses import CombinedLoss
        from utils.optimizers import OptimizerManager
        from utils.train_loop import train_model
        from config import load_train_test_split
        
        # Load data
        split_data = load_train_test_split(split_file)
        train_loader, val_loader, test_loader, class_weights = create_data_loaders(
            pkl_files=pkl_files,
            split_json=split_data,
            val_split=0.3,
            batch_size=4,
            num_workers=0,
            model_type='autoencoder',
            random_state=42
        )
        
        # Create model
        model = create_model(
            model_type='autoencoder',
            attention_k=4,
            encoder_layers=[32, 16],
            latent_dim=8,
            predictor_layers=[8, 4],
            dropout_rate=0.2
        )
        
        # Create loss and optimizer
        criterion = CombinedLoss(class_weights, device='cpu')
        
        optimizer_manager = OptimizerManager(
            model=model,
            optimizer_config={'optimizer_type': 'adam', 'learning_rate': 1e-3, 'weight_decay': 1e-4},
            scheduler_config={'scheduler_type': 'plateau', 'patience': 3},
            early_stopping_config={'patience': 5, 'min_delta': 1e-3}
        )
        
        # Run mini training
        save_dir = os.path.join(temp_dir, 'mini_training')
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer_manager=optimizer_manager,
            epochs=3,  # Very short training
            device='cpu',
            model_type='autoencoder',
            save_dir=save_dir
        )
        
        print(f"✓ Mini training completed - {len(history['train_loss'])} epochs")
        
        return True
    except Exception as e:
        print(f"✗ Mini training error: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("ATTENTION-BASED NEURAL NETWORKS - FRAMEWORK TEST")
    print("=" * 60)
    
    test_results = []
    
    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # Test 1: Imports
        test_results.append(("Imports", test_imports()))
        
        # Test 2: Model Creation
        test_results.append(("Model Creation", test_model_creation()))
        
        # Test 3: Forward Pass
        test_results.append(("Forward Pass", test_forward_pass()))
        
        # Create test data
        pkl_files, split_file = create_test_data(temp_dir)
        
        # Test 4: Data Loading
        test_results.append(("Data Loading", test_data_loading(pkl_files, split_file)))
        
        # Test 5: Loss Functions
        test_results.append(("Loss Functions", test_loss_functions()))
        
        # Test 6: Optimizer Manager
        test_results.append(("Optimizer Manager", test_optimizer_manager()))
        
        # Test 7: Mini Training
        test_results.append(("Mini Training", test_mini_training(pkl_files, split_file, temp_dir)))
    
    # Print results
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<20}: {status}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"TOTAL: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Framework is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)