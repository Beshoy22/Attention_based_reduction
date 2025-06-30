#!/usr/bin/env python3
"""
Extract reduced features from trained autoencoder model
"""

import torch
import numpy as np
import pandas as pd
import pickle
import argparse
import os
from typing import List, Dict

from utils.models import create_model
from utils.dataloader import MedicalImagingDataset, custom_collate_fn
from torch.utils.data import DataLoader

def load_all_patients_data(pkl_files: List[str]) -> List[Dict]:
    """Load all patient data from pkl files"""
    all_data = []
    for pkl_file in pkl_files:
        center_name = os.path.basename(pkl_file).replace('.pkl', '')
        with open(pkl_file, 'rb') as f:
            center_data = pickle.load(f)
        
        for patient in center_data:
            patient['center'] = center_name
            all_data.append(patient)
    
    return all_data

def extract_features_and_outcomes(
    model_path: str,
    pkl_files: List[str],
    attention_k: int,
    encoder_layers: List[int],
    latent_dim: int,
    predictor_layers: List[int],
    dropout_rate: float,
    endpoints: List[str],
    device: str = 'cuda'
) -> pd.DataFrame:
    """Extract reduced features and outcomes for all patients"""
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Create model with provided config
    model = create_model(
        model_type='autoencoder',
        endpoints=endpoints,
        attention_k=attention_k,
        encoder_layers=encoder_layers,
        latent_dim=latent_dim,
        predictor_layers=predictor_layers,
        dropout_rate=dropout_rate
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load all patient data
    all_data = load_all_patients_data(pkl_files)
    
    # Create dataset (include all patients regardless of missing endpoints)
    dataset = MedicalImagingDataset(all_data, include_missing_endpoints=True, endpoints=endpoints)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
    
    # Extract features and collect data
    results = []
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            
            # Forward pass through autoencoder to get latent features
            reconstructed, predictions = model(features)
            
            # Get latent features (encoded representation)
            attended_features = model.attention(features)
            batch_size = attended_features.size(0)
            flattened = attended_features.view(batch_size, -1)
            latent_features = model.encoder(flattened)  # This is the reduced feature representation
            
            # Process each patient in batch
            for i in range(batch_size):
                patient_id = batch['patient_id'][i]
                
                # Get reduced features
                features_array = latent_features[i].cpu().numpy()
                
                # Create base record
                record = {'Subject': patient_id}
                
                # Add reduced features as columns
                for j, feat_val in enumerate(features_array):
                    record[f'feature_{j:03d}'] = feat_val
                
                # Find original patient data to get all available outcomes
                original_patient = None
                for patient in all_data:
                    if patient['patient_id'] == patient_id:
                        original_patient = patient
                        break
                
                if original_patient:
                    # Add OS values
                    record['OS_6'] = original_patient.get('OS_6', np.nan)
                    record['OS_24'] = original_patient.get('OS_24', np.nan)
                    
                    # Add TNM values
                    record['STAGE_T'] = original_patient.get('STAGE_DIAGNOSIS_T', np.nan)
                    record['STAGE_N'] = original_patient.get('STAGE_DIAGNOSIS_N', np.nan)
                    record['STAGE_M'] = original_patient.get('STAGE_DIAGNOSIS_M', np.nan)
                    
                    # Add center info
                    record['Center'] = original_patient.get('center', 'unknown')
                else:
                    # Fallback if patient not found
                    record.update({
                        'OS_6': np.nan, 'OS_24': np.nan,
                        'STAGE_T': np.nan, 'STAGE_N': np.nan, 'STAGE_M': np.nan,
                        'Center': 'unknown'
                    })
                
                results.append(record)
    
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description='Extract reduced features from autoencoder')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model (.pth file)')
    parser.add_argument('--pkl_files', nargs='+', required=True, help='List of .pkl files')
    parser.add_argument('--output_csv', type=str, default='reduced_features.csv', help='Output CSV file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    # Model configuration
    parser.add_argument('--attention_k', type=int, default=32, help='Number of attention vectors')
    parser.add_argument('--encoder_layers', nargs='+', type=int, default=[256, 128], help='Encoder layer sizes')
    parser.add_argument('--latent_dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--predictor_layers', nargs='+', type=int, default=[64, 32], help='Predictor layer sizes')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--endpoints', nargs='+', choices=['os6', 'os24', 'stage_t', 'stage_n', 'stage_m'], 
                        default=['os6', 'os24'], help='Endpoints used in training')
    
    args = parser.parse_args()
    
    print(f"Loading model from: {args.model_path}")
    print(f"Processing {len(args.pkl_files)} pkl files...")
    
    # Extract features and outcomes
    df = extract_features_and_outcomes(
        model_path=args.model_path,
        pkl_files=args.pkl_files,
        attention_k=args.attention_k,
        encoder_layers=args.encoder_layers,
        latent_dim=args.latent_dim,
        predictor_layers=args.predictor_layers,
        dropout_rate=args.dropout_rate,
        endpoints=args.endpoints,
        device=args.device
    )
    
    # Save to CSV
    df.to_csv(args.output_csv, index=False)
    
    print(f"Extracted features for {len(df)} patients")
    print(f"Feature dimensions: {len([col for col in df.columns if col.startswith('feature_')])}")
    print(f"Results saved to: {args.output_csv}")
    
    # Print summary of available data
    print("\nData availability summary:")
    outcome_cols = ['OS_6', 'OS_24', 'STAGE_T', 'STAGE_N', 'STAGE_M']
    for col in outcome_cols:
        available = df[col].notna().sum()
        print(f"  {col}: {available}/{len(df)} ({available/len(df)*100:.1f}%)")

if __name__ == "__main__":
    main()