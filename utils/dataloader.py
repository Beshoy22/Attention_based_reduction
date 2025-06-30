import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from typing import List, Dict, Tuple, Optional
import os
from collections import Counter

def custom_collate_fn(batch):
    """
    Custom collate function to handle None values in endpoints
    """
    # Separate the batch elements
    features = torch.stack([item['features'] for item in batch])
    patient_ids = [item['patient_id'] for item in batch]
    centers = [item['center'] for item in batch]
    
    # Handle endpoints which can be None
    endpoints = {}
    if batch:
        for ep_name in batch[0]['endpoints'].keys():
            endpoints[ep_name] = [item['endpoints'][ep_name] for item in batch]
    
    return {
        'features': features,
        'endpoints': endpoints,
        'patient_id': patient_ids,
        'center': centers
    }

class MedicalImagingDataset(Dataset):
    def __init__(self, data: List[Dict], include_missing_endpoints: bool = True, endpoints: List[str] = ['os6', 'os24']):
        """
        Dataset for medical imaging data
        
        Args:
            data: List of patient dictionaries
            include_missing_endpoints: Whether to include patients without endpoints
            endpoints: Which endpoints to consider
        """
        self.endpoints = endpoints
        self.data = []
        
        # Map endpoint names to dict keys
        endpoint_map = {
            'os6': 'OS_6', 'os24': 'OS_24',
            'stage_t': 'STAGE_DIAGNOSIS_T',
            'stage_n': 'STAGE_DIAGNOSIS_N', 
            'stage_m': 'STAGE_DIAGNOSIS_M'
        }
        
        for patient in data:
            if include_missing_endpoints:
                self.data.append(patient)
            else:
                # Require ALL specified endpoints to be present
                has_all = all(endpoint_map[ep] in patient and patient[endpoint_map[ep]] is not None 
                             for ep in endpoints)
                if has_all:
                    self.data.append(patient)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        patient = self.data[idx]
        
        # Convert features to tensor
        features = torch.stack(patient['features'])  # Shape: (176, 512)
        
        # Map endpoint names to dict keys
        endpoint_map = {
            'os6': 'OS_6', 'os24': 'OS_24',
            'stage_t': 'STAGE_DIAGNOSIS_T',
            'stage_n': 'STAGE_DIAGNOSIS_N', 
            'stage_m': 'STAGE_DIAGNOSIS_M'
        }
        
        # Get endpoints if available
        endpoint_values = {}
        for ep in self.endpoints:
            key = endpoint_map[ep]
            value = patient.get(key, None)
            if value is not None:
                # Keep original values for T,N (multiclass), binary for M and survival
                if ep == 'stage_m':
                    value = float(value)  # M is already 0/1
                elif ep in ['stage_t', 'stage_n']:
                    value = int(value)  # Keep as class indices
                else:  # os6, os24
                    value = float(value)
                endpoint_values[ep] = torch.tensor(value, dtype=torch.long if ep in ['stage_t', 'stage_n'] else torch.float32)
            else:
                endpoint_values[ep] = None
        
        return {
            'features': features,
            'endpoints': endpoint_values,
            'patient_id': patient['patient_id'],
            'center': patient.get('center', 'unknown')
        }

def load_pkl_files(pkl_files: List[str]) -> List[Dict]:
    """Load all .pkl files and add center information"""
    all_data = []
    
    for pkl_file in pkl_files:
        center_name = os.path.basename(pkl_file).replace('.pkl', '')
        
        with open(pkl_file, 'rb') as f:
            center_data = pickle.load(f)
        
        # Add center information to each patient
        for patient in center_data:
            patient['center'] = center_name
            all_data.append(patient)
    
    return all_data

def stratified_split_by_center_and_endpoints(
    data: List[Dict], 
    test_size: float, 
    endpoints: List[str] = ['os6', 'os24'],
    random_state: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Stratified split with fallback options when stratification fails
    """
    endpoint_map = {
        'os6': 'OS_6', 'os24': 'OS_24',
        'stage_t': 'STAGE_DIAGNOSIS_T',
        'stage_n': 'STAGE_DIAGNOSIS_N', 
        'stage_m': 'STAGE_DIAGNOSIS_M'
    }
    
    # Priority order for endpoints (OS_6 and OS_24 first)
    priority_endpoints = ['os6', 'os24'] + [ep for ep in endpoints if ep not in ['os6', 'os24']]
    
    # Try stratification with decreasing complexity
    for num_endpoints in range(len(priority_endpoints), 0, -1):
        current_endpoints = priority_endpoints[:num_endpoints]
        current_endpoints = [ep for ep in current_endpoints if ep in endpoints]
        
        if not current_endpoints:
            continue
            
        stratify_labels = []
        valid_data = []
        
        for patient in data:
            center = patient['center']
            
            # Check if patient has ALL current endpoints
            endpoint_values = []
            has_all_endpoints = True
            for ep in current_endpoints:
                key = endpoint_map[ep]
                if key in patient and patient[key] is not None:
                    endpoint_values.append(str(int(patient[key])))
                else:
                    has_all_endpoints = False
                    break
            
            if has_all_endpoints:
                label = f"{center}_{'_'.join(endpoint_values)}"
                stratify_labels.append(label)
                valid_data.append(patient)
        
        if len(valid_data) < 4:  # Need at least 4 samples for stratification
            continue
            
        # Check if all classes have at least 2 members
        label_counts = Counter(stratify_labels)
        if min(label_counts.values()) >= 2:
            try:
                train_data, val_data = train_test_split(
                    valid_data,
                    test_size=test_size,
                    stratify=stratify_labels,
                    random_state=random_state
                )
                
                # Add back patients without required endpoints to training set
                patients_without_endpoints = [p for p in data if p not in valid_data]
                train_data.extend(patients_without_endpoints)
                
                print(f"Stratification successful with endpoints: {current_endpoints}")
                return train_data, val_data
                
            except ValueError:
                continue
    
    # Fallback: random split
    print("Stratification failed, using random split")
    return train_test_split(data, test_size=test_size, random_state=random_state)

def apply_train_test_split(data: List[Dict], split_json: Dict) -> Tuple[List[Dict], List[Dict]]:
    """Apply train/test split based on provided JSON"""
    train_patient_ids = set(split_json['TRAIN_SET'])
    test_patient_ids = set(split_json['TEST_SET'])
    
    train_data = []
    test_data = []
    
    for patient in data:
        patient_id = patient['patient_id']
        if patient_id in train_patient_ids:
            train_data.append(patient)
        elif patient_id in test_patient_ids:
            test_data.append(patient)
        # Ignore patients not in the JSON (as mentioned in requirements)
    
    return train_data, test_data

def create_automatic_train_val_test_split(
    data: List[Dict],
    val_split: float,
    endpoints: List[str] = ['os6', 'os24'],
    random_state: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Create automatic 60-20-20 split with stratification when no JSON provided
    """
    endpoint_map = {
        'os6': 'OS_6', 'os24': 'OS_24',
        'stage_t': 'STAGE_DIAGNOSIS_T',
        'stage_n': 'STAGE_DIAGNOSIS_N', 
        'stage_m': 'STAGE_DIAGNOSIS_M'
    }
    
    # Calculate split sizes: if val_split=0.2, then train=0.6, val=0.2, test=0.2
    test_size = val_split
    val_size = val_split
    train_size = 1.0 - val_size - test_size
    
    print(f"Automatic split: Train {train_size:.0%}, Val {val_size:.0%}, Test {test_size:.0%}")
    
    # Try stratification with decreasing complexity (similar to existing function)
    priority_endpoints = ['os6', 'os24'] + [ep for ep in endpoints if ep not in ['os6', 'os24']]
    
    for num_endpoints in range(len(priority_endpoints), 0, -1):
        current_endpoints = priority_endpoints[:num_endpoints]
        current_endpoints = [ep for ep in current_endpoints if ep in endpoints]
        
        if not current_endpoints:
            continue
            
        stratify_labels = []
        valid_data = []
        
        for patient in data:
            center = patient['center']
            
            # Check if patient has ALL current endpoints
            endpoint_values = []
            has_all_endpoints = True
            for ep in current_endpoints:
                key = endpoint_map[ep]
                if key in patient and patient[key] is not None:
                    endpoint_values.append(str(int(patient[key])))
                else:
                    has_all_endpoints = False
                    break
            
            if has_all_endpoints:
                label = f"{center}_{'_'.join(endpoint_values)}"
                stratify_labels.append(label)
                valid_data.append(patient)
        
        if len(valid_data) < 6:  # Need at least 6 samples for 3-way split
            continue
            
        # Check if all classes have at least 3 members
        label_counts = Counter(stratify_labels)
        if min(label_counts.values()) >= 3:
            try:
                # First split: separate test set
                train_val_data, test_data = train_test_split(
                    valid_data,
                    test_size=test_size,
                    stratify=stratify_labels,
                    random_state=random_state
                )
                
                # Second split: separate train and val from remaining data
                train_val_labels = []
                for patient in train_val_data:
                    center = patient['center']
                    endpoint_values = []
                    for ep in current_endpoints:
                        key = endpoint_map[ep]
                        endpoint_values.append(str(int(patient[key])))
                    train_val_labels.append(f"{center}_{'_'.join(endpoint_values)}")
                
                # Calculate val_size relative to remaining data
                relative_val_size = val_size / (train_size + val_size)
                
                train_data, val_data = train_test_split(
                    train_val_data,
                    test_size=relative_val_size,
                    stratify=train_val_labels,
                    random_state=random_state
                )
                
                # Add back patients without required endpoints to training set
                patients_without_endpoints = [p for p in data if p not in valid_data]
                train_data.extend(patients_without_endpoints)
                
                print(f"Stratification successful with endpoints: {current_endpoints}")
                return train_data, val_data, test_data
                
            except ValueError as e:
                print(f"Stratification failed with {len(current_endpoints)} endpoints: {e}")
                continue
    
    # Fallback: random split
    print("Stratification failed, using random split")
    train_val_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    relative_val_size = val_size / (train_size + val_size)
    train_data, val_data = train_test_split(train_val_data, test_size=relative_val_size, random_state=random_state)
    
    return train_data, val_data, test_data

def compute_class_weights(data: List[Dict], endpoints: List[str]) -> Dict[str, torch.Tensor]:
    """Compute class weights for balanced loss"""
    endpoint_map = {
        'os6': 'OS_6', 'os24': 'OS_24',
        'stage_t': 'STAGE_DIAGNOSIS_T',
        'stage_n': 'STAGE_DIAGNOSIS_N', 
        'stage_m': 'STAGE_DIAGNOSIS_M'
    }
    
    weights = {}
    
    for ep in endpoints:
        key = endpoint_map[ep]
        values = []
        for patient in data:
            if key in patient and patient[key] is not None:
                values.append(int(patient[key]))
        
        if values:
            unique_classes = np.unique(values)
            if ep in ['stage_t', 'stage_n']:
                # Multiclass: compute weight for each class
                ep_weights = compute_class_weight('balanced', classes=unique_classes, y=values)
                weights[ep] = torch.tensor(ep_weights, dtype=torch.float32)
            else:
                # Binary: standard balanced weights
                ep_weights = compute_class_weight('balanced', classes=unique_classes, y=values)
                weights[ep] = torch.tensor(ep_weights, dtype=torch.float32)
    
    return weights

def create_data_loaders(
    pkl_files: List[str],
    split_json: Dict = None,
    val_split: float = 0.2,
    batch_size: int = 16,
    num_workers: int = 0,
    model_type: str = 'autoencoder',
    endpoints: List[str] = ['os6', 'os24'],
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create train, validation, and test data loaders
    
    Args:
        pkl_files: List of .pkl file paths
        split_json: Train/test split JSON (optional)
        val_split: Validation split ratio
        batch_size: Batch size
        num_workers: Number of data loading workers
        model_type: 'autoencoder' or 'endtoend'
        endpoints: List of endpoints to use
        random_state: Random seed
    
    Returns:
        train_loader, val_loader, test_loader, class_weights
    """
    # Load all data
    all_data = load_pkl_files(pkl_files)
    
    # Split data based on whether JSON is provided
    if split_json is not None:
        # Use provided train/test split from JSON
        train_val_data, test_data = apply_train_test_split(all_data, split_json)
        
        # Split train_val into train and validation
        train_data, val_data = stratified_split_by_center_and_endpoints(
            train_val_data, val_split, endpoints, random_state
        )
    else:
        # Create automatic 60-20-20 split (or whatever val_split specifies)
        train_data, val_data, test_data = create_automatic_train_val_test_split(
            all_data, val_split, endpoints, random_state
        )
    
    # Determine whether to include missing endpoints
    include_missing = (model_type == 'autoencoder')
    
    # Create datasets
    train_dataset = MedicalImagingDataset(train_data, include_missing_endpoints=include_missing, endpoints=endpoints)
    val_dataset = MedicalImagingDataset(val_data, include_missing_endpoints=False, endpoints=endpoints)  # Always exclude for validation
    test_dataset = MedicalImagingDataset(test_data, include_missing_endpoints=False, endpoints=endpoints)  # Always exclude for test
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, collate_fn=custom_collate_fn)
    
    # Compute class weights
    class_weights = compute_class_weights(train_data, endpoints)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, class_weights