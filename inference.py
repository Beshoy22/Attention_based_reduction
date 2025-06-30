#!/usr/bin/env python3
"""
Inference script for making predictions on new data using trained models
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import json
import argparse
import os
from typing import Dict, List, Tuple, Optional
import pandas as pd

from utils.models import create_model
from utils.dataloader import MedicalImagingDataset, custom_collate_fn
from torch.utils.data import DataLoader

class ModelInference:
    """Class for model inference and prediction"""
    
    def __init__(self, model_path: str, model_type: str, device: str = 'cuda'):
        """
        Initialize inference class
        
        Args:
            model_path: Path to saved model checkpoint
            model_type: 'autoencoder' or 'endtoend'
            device: Device to run inference on
        """
        self.model_type = model_type
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model checkpoint
        self.checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Extract model configuration from checkpoint or use defaults
        if 'model_config' in self.checkpoint:
            self.model_config = self.checkpoint['model_config']
        else:
            # Default configuration - you may need to adjust these
            self.model_config = {
                'attention_k': 32,
                'encoder_layers': [256, 128],
                'latent_dim': 128,
                'predictor_layers': [64, 32],
                'dropout_rate': 0.3
            }
        
        # Create and load model
        self.model = self._create_and_load_model()
        
    def _create_and_load_model(self) -> nn.Module:
        """Create model and load weights"""
        model = create_model(
            model_type=self.model_type,
            **self.model_config
        ).to(self.device)
        
        # Load weights
        model.load_state_dict(self.checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Loaded {self.model_type} model from checkpoint")
        print(f"Model configuration: {self.model_config}")
        
        return model
    
    def predict_batch(self, features: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Make predictions on a batch of features
        
        Args:
            features: Input features tensor (batch_size, 176, 512)
        
        Returns:
            Dictionary with predictions and features
        """
        self.model.eval()
        
        with torch.no_grad():
            features = features.to(self.device)
            
            if self.model_type == 'autoencoder':
                reconstructed, os6_pred, os24_pred = self.model(features)
                attended_features = self.model.attention(features)
                
                return {
                    'os6_predictions': os6_pred.cpu().numpy(),
                    'os24_predictions': os24_pred.cpu().numpy(),
                    'reconstructed_features': reconstructed.cpu().numpy(),
                    'attended_features': attended_features.cpu().numpy(),
                    'latent_features': self.model.encoder(attended_features.view(features.size(0), -1)).cpu().numpy()
                }
            
            else:  # endtoend
                reduced_features, os6_pred, os24_pred = self.model(features)
                attended_features = self.model.attention(features)
                
                return {
                    'os6_predictions': os6_pred.cpu().numpy(),
                    'os24_predictions': os24_pred.cpu().numpy(),
                    'reduced_features': reduced_features.cpu().numpy(),
                    'attended_features': attended_features.cpu().numpy()
                }
    
    def predict_from_pkl(self, pkl_files: List[str], batch_size: int = 16) -> pd.DataFrame:
        """
        Make predictions on data from pkl files
        
        Args:
            pkl_files: List of pkl file paths
            batch_size: Batch size for inference
        
        Returns:
            DataFrame with predictions
        """
        # Load data
        all_data = []
        for pkl_file in pkl_files:
            with open(pkl_file, 'rb') as f:
                center_data = pickle.load(f)
            
            center_name = os.path.basename(pkl_file).replace('.pkl', '')
            for patient in center_data:
                patient['center'] = center_name
                all_data.append(patient)
        
        # Create dataset and dataloader
        dataset = MedicalImagingDataset(all_data, include_missing_endpoints=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, 
                               collate_fn=custom_collate_fn)
        
        # Collect predictions
        all_predictions = []
        
        for batch in dataloader:
            features = batch['features']
            predictions = self.predict_batch(features)
            
            # Create records for each sample in batch
            batch_size_actual = features.size(0)
            for i in range(batch_size_actual):
                record = {
                    'patient_id': batch['patient_id'][i],
                    'center': batch['center'][i],
                    'os6_prediction': predictions['os6_predictions'][i, 0],
                    'os24_prediction': predictions['os24_predictions'][i, 0],
                    'os6_prediction_binary': int(predictions['os6_predictions'][i, 0] > 0.5),
                    'os24_prediction_binary': int(predictions['os24_predictions'][i, 0] > 0.5)
                }
                
                # Add ground truth if available
                if batch['os6'][i] is not None:
                    record['os6_true'] = batch['os6'][i].item()
                if batch['os24'][i] is not None:
                    record['os24_true'] = batch['os24'][i].item()
                
                all_predictions.append(record)
        
        return pd.DataFrame(all_predictions)
    
    def extract_features(self, pkl_files: List[str], batch_size: int = 16) -> Dict[str, np.ndarray]:
        """
        Extract features from data
        
        Args:
            pkl_files: List of pkl file paths
            batch_size: Batch size for feature extraction
        
        Returns:
            Dictionary with extracted features and metadata
        """
        # Load data
        all_data = []
        for pkl_file in pkl_files:
            with open(pkl_file, 'rb') as f:
                center_data = pickle.load(f)
            
            center_name = os.path.basename(pkl_file).replace('.pkl', '')
            for patient in center_data:
                patient['center'] = center_name
                all_data.append(patient)
        
        # Create dataset and dataloader
        dataset = MedicalImagingDataset(all_data, include_missing_endpoints=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                               collate_fn=custom_collate_fn)
        
        # Extract features
        all_features = {}
        patient_ids = []
        centers = []
        os6_true = []
        os24_true = []
        
        for batch in dataloader:
            features = batch['features']
            predictions = self.predict_batch(features)
            
            # Initialize feature arrays if first batch
            if not all_features:
                for key, value in predictions.items():
                    if key.endswith('_features'):
                        all_features[key] = []
            
            # Collect features
            for key in all_features.keys():
                all_features[key].append(predictions[key])
            
            # Collect metadata
            patient_ids.extend(batch['patient_id'])
            centers.extend(batch['center'])
            
            batch_size_actual = features.size(0)
            for i in range(batch_size_actual):
                os6_true.append(batch['os6'][i].item() if batch['os6'][i] is not None else None)
                os24_true.append(batch['os24'][i].item() if batch['os24'][i] is not None else None)
        
        # Concatenate features
        for key in all_features.keys():
            all_features[key] = np.concatenate(all_features[key], axis=0)
        
        # Add metadata
        all_features['metadata'] = {
            'patient_ids': patient_ids,
            'centers': centers,
            'os6_true': os6_true,
            'os24_true': os24_true
        }
        
        return all_features

def main():
    parser = argparse.ArgumentParser(description='Model Inference Script')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to saved model checkpoint')
    parser.add_argument('--model_type', type=str, choices=['autoencoder', 'endtoend'], required=True,
                        help='Type of model')
    parser.add_argument('--pkl_files', nargs='+', required=True,
                        help='List of .pkl files for inference')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                        help='Directory to save inference results')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run inference on')
    parser.add_argument('--extract_features', action='store_true',
                        help='Extract and save features')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize inference
    print("Initializing model for inference...")
    inference = ModelInference(args.model_path, args.model_type, args.device)
    
    # Make predictions
    print("Making predictions...")
    predictions_df = inference.predict_from_pkl(args.pkl_files, args.batch_size)
    
    # Save predictions
    predictions_path = os.path.join(args.output_dir, 'predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")
    
    # Print summary
    print(f"\nPrediction Summary:")
    print(f"Total patients: {len(predictions_df)}")
    print(f"OS6 positive predictions: {predictions_df['os6_prediction_binary'].sum()}")
    print(f"OS24 positive predictions: {predictions_df['os24_prediction_binary'].sum()}")
    
    if 'os6_true' in predictions_df.columns:
        os6_accuracy = (predictions_df['os6_prediction_binary'] == predictions_df['os6_true']).mean()
        print(f"OS6 accuracy (where ground truth available): {os6_accuracy:.3f}")
    
    if 'os24_true' in predictions_df.columns:
        os24_accuracy = (predictions_df['os24_prediction_binary'] == predictions_df['os24_true']).mean()
        print(f"OS24 accuracy (where ground truth available): {os24_accuracy:.3f}")
    
    # Extract features if requested
    if args.extract_features:
        print("\nExtracting features...")
        features_dict = inference.extract_features(args.pkl_files, args.batch_size)
        
        # Save features
        for key, value in features_dict.items():
            if key != 'metadata':
                np.save(os.path.join(args.output_dir, f'{key}.npy'), value)
                print(f"Saved {key} with shape {value.shape}")
        
        # Save metadata
        with open(os.path.join(args.output_dir, 'features_metadata.json'), 'w') as f:
            json.dump(features_dict['metadata'], f, indent=2)
        
        print("Feature extraction complete!")

if __name__ == "__main__":
    main()