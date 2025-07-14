import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from typing import List, Dict, Tuple, Optional
import os
import warnings
from collections import Counter
from .pet_preprocessing import PETPreprocessor, compute_pet_aspect_ratio

def check_features_for_nan(features_list: List[torch.Tensor], patient_id: str) -> bool:
    """
    Check if any features contain NaN values
    
    Args:
        features_list: List of feature tensors for a patient
        patient_id: Patient identifier for warning messages
        
    Returns:
        True if no NaN values found, False if NaN values detected
    """
    for i, feature_tensor in enumerate(features_list):
        if torch.isnan(feature_tensor).any():
            warnings.warn(f"WARNING: Patient {patient_id} excluded due to NaN values in feature tensor {i}")
            return False
    return True

def filter_patients_with_nan_features(data: List[Dict]) -> List[Dict]:
    """
    Filter out patients that have NaN values in their feature tensors
    
    Args:
        data: List of patient data dictionaries
        
    Returns:
        Filtered list of patients without NaN features
    """
    filtered_data = []
    excluded_patients = []
    
    for patient in data:
        patient_id = patient.get('patient_id', 'unknown')
        features = patient.get('features', [])
        
        if check_features_for_nan(features, patient_id):
            filtered_data.append(patient)
        else:
            excluded_patients.append(patient_id)
    
    if excluded_patients:
        print(f"EXCLUDED {len(excluded_patients)} patients due to NaN values in features:")
        for patient_id in excluded_patients:
            print(f"  - Patient ID: {patient_id}")
    
    print(f"Feature validation: kept {len(filtered_data)} patients, excluded {len(excluded_patients)} patients with NaN features")
    return filtered_data

def custom_collate_fn(batch):
    """
    Custom collate function to handle None values in endpoints and invalid patients
    """
    # Filter out None items (patients with invalid survival data)
    batch = [item for item in batch if item is not None]
    
    if not batch:
        return None  # Return None if all items in batch are invalid
    
    # Separate the batch elements
    features = torch.stack([item['features'] for item in batch])
    patient_ids = [item['patient_id'] for item in batch]
    centers = [item['center'] for item in batch]
    
    # Handle endpoints which can be None
    endpoints = {}
    if batch:
        for ep_name in batch[0]['endpoints'].keys():
            endpoints[ep_name] = [item['endpoints'][ep_name] for item in batch]
    
    # Handle PET data (can be None)
    pet_coronal = None
    pet_sagittal = None
    
    pet_coronal_list = [item['pet_coronal'] for item in batch]
    pet_sagittal_list = [item['pet_sagittal'] for item in batch]
    
    # Stack PET data if available for all items in batch
    if all(pet is not None for pet in pet_coronal_list):
        pet_coronal = torch.stack(pet_coronal_list)
    if all(pet is not None for pet in pet_sagittal_list):
        pet_sagittal = torch.stack(pet_sagittal_list)
    
    return {
        'features': features,
        'endpoints': endpoints,
        'patient_id': patient_ids,
        'center': centers,
        'pet_coronal': pet_coronal,
        'pet_sagittal': pet_sagittal
    }

class MedicalImagingDataset(Dataset):
    def __init__(self, data: List[Dict], include_missing_endpoints: bool = True, endpoints: List[str] = ['os6', 'os24'], 
                 pet_preprocessor: Optional[PETPreprocessor] = None):
        """
        Dataset for medical imaging data
        
        Args:
            data: List of patient dictionaries
            include_missing_endpoints: Whether to include patients without endpoints
            endpoints: Which endpoints to consider
            pet_preprocessor: PET preprocessor instance for handling PET images
        """
        self.endpoints = endpoints
        self.data = []
        self.pet_preprocessor = pet_preprocessor
        
        # Map endpoint names to dict keys
        endpoint_map = {
            'os6': 'OS_6', 'os24': 'OS_24',
            'stage_t': 'STAGE_DIAGNOSIS_T',
            'stage_n': 'STAGE_DIAGNOSIS_N', 
            'stage_m': 'STAGE_DIAGNOSIS_M',
            'survival': ['OS_MONTHS', 'DEATH_EVENT_OC']  # Survival analysis endpoints
        }
        
        for patient in data:
            if include_missing_endpoints:
                self.data.append(patient)
            else:
                # Require ALL specified endpoints to be present
                has_all = True
                for ep in endpoints:
                    if ep == 'survival':
                        # For survival analysis, need both OS_MONTHS and DEATH_EVENT_OC
                        survival_keys = endpoint_map[ep]
                        if not all(key in patient and patient[key] is not None for key in survival_keys):
                            has_all = False
                            break
                    else:
                        # For other endpoints, check single key
                        if endpoint_map[ep] not in patient or patient[endpoint_map[ep]] is None:
                            has_all = False
                            break
                if has_all:
                    self.data.append(patient)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        patient = self.data[idx]
        
        # Convert features to tensor
        features = torch.stack(patient['features'])  # Shape: (176, 512)
        
        # Additional safety check for NaN values
        if torch.isnan(features).any():
            patient_id = patient.get('patient_id', 'unknown')
            raise ValueError(f"NaN values detected in features for patient {patient_id} during batch creation. This should have been filtered out earlier.")
        
        # Map endpoint names to dict keys
        endpoint_map = {
            'os6': 'OS_6', 'os24': 'OS_24',
            'stage_t': 'STAGE_DIAGNOSIS_T',
            'stage_n': 'STAGE_DIAGNOSIS_N', 
            'stage_m': 'STAGE_DIAGNOSIS_M',
            'survival': ['OS_MONTHS', 'DEATH_EVENT_OC']  # Survival analysis endpoints
        }
        
        # Get endpoints if available
        endpoint_values = {}
        for ep in self.endpoints:
            if ep == 'survival':
                # Handle survival analysis endpoints
                survival_keys = endpoint_map[ep]
                os_months = patient.get(survival_keys[0], None)
                death_event = patient.get(survival_keys[1], None)
                
                if os_months is not None and death_event is not None:
                    # Convert to appropriate tensors
                    os_months = float(os_months)
                    death_event = float(death_event)
                    
                    # Validate survival data - if invalid, this patient should be excluded
                    if os_months < 0:
                        warnings.warn(f"WARNING: Patient {patient['patient_id']} excluded due to invalid survival time {os_months} - must be non-negative")
                        return None  # Signal to exclude this patient
                    if death_event not in [0.0, 1.0]:
                        warnings.warn(f"WARNING: Patient {patient['patient_id']} excluded due to invalid death event {death_event} - must be 0 or 1")
                        return None  # Signal to exclude this patient
                    
                    endpoint_values[ep] = {
                        'time': torch.tensor(os_months, dtype=torch.float32),
                        'event': torch.tensor(death_event, dtype=torch.float32)
                    }
                else:
                    endpoint_values[ep] = None
            else:
                # Handle traditional endpoints
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
        
        # Handle PET data if available
        pet_coronal = None
        pet_sagittal = None
        
        if self.pet_preprocessor is not None:
            # Check if patient has PET data
            if 'coronal_png_path' in patient and 'sagittal_png_path' in patient:
                try:
                    pet_coronal, pet_sagittal = self.pet_preprocessor.preprocess_pet_pair(
                        patient['coronal_png_path'], 
                        patient['sagittal_png_path']
                    )
                except Exception as e:
                    warnings.warn(f"Failed to load PET data for patient {patient['patient_id']}: {e}")
                    pet_coronal = None
                    pet_sagittal = None
        
        return {
            'features': features,
            'endpoints': endpoint_values,
            'patient_id': patient['patient_id'],
            'center': patient.get('center', 'unknown'),
            'pet_coronal': pet_coronal,
            'pet_sagittal': pet_sagittal
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
        'stage_m': 'STAGE_DIAGNOSIS_M',
        'survival': ['OS_MONTHS', 'DEATH_EVENT_OC']
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
                if ep == 'survival':
                    # For survival, check both OS_MONTHS and DEATH_EVENT_OC
                    survival_keys = endpoint_map[ep]
                    if all(key in patient and patient[key] is not None for key in survival_keys):
                        # Use death event as stratification value for survival
                        endpoint_values.append(str(int(patient[survival_keys[1]])))
                    else:
                        has_all_endpoints = False
                        break
                else:
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
        'stage_m': 'STAGE_DIAGNOSIS_M',
        'survival': ['OS_MONTHS', 'DEATH_EVENT_OC']
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
                if ep == 'survival':
                    # For survival, check both OS_MONTHS and DEATH_EVENT_OC
                    survival_keys = endpoint_map[ep]
                    if all(key in patient and patient[key] is not None for key in survival_keys):
                        # Use death event as stratification value for survival
                        endpoint_values.append(str(int(patient[survival_keys[1]])))
                    else:
                        has_all_endpoints = False
                        break
                else:
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
                        if ep == 'survival':
                            # For survival, use death event as stratification value
                            survival_keys = endpoint_map[ep]
                            endpoint_values.append(str(int(patient[survival_keys[1]])))
                        else:
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
        'stage_m': 'STAGE_DIAGNOSIS_M',
        'survival': ['OS_MONTHS', 'DEATH_EVENT_OC']
    }
    
    weights = {}
    
    for ep in endpoints:
        if ep == 'survival':
            # For survival, use death event for class weights
            survival_keys = endpoint_map[ep]
            values = []
            for patient in data:
                if all(key in patient and patient[key] is not None for key in survival_keys):
                    values.append(int(patient[survival_keys[1]]))  # Use death event
            
            if values:
                unique_classes = np.unique(values)
                ep_weights = compute_class_weight('balanced', classes=unique_classes, y=values)
                weights[ep] = torch.tensor(ep_weights, dtype=torch.float32)
        else:
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

def filter_patients_with_all_endpoints(data: List[Dict], endpoints: List[str]) -> List[Dict]:
    """
    Filter patients to keep only those with all required endpoints
    
    Args:
        data: List of patient data dictionaries
        endpoints: List of endpoints that must be present
        
    Returns:
        Filtered list of patients with all required endpoints
    """
    endpoint_map = {
        'os6': 'OS_6', 'os24': 'OS_24',
        'stage_t': 'STAGE_DIAGNOSIS_T',
        'stage_n': 'STAGE_DIAGNOSIS_N', 
        'stage_m': 'STAGE_DIAGNOSIS_M',
        'survival': ['OS_MONTHS', 'DEATH_EVENT_OC']
    }
    
    filtered_data = []
    excluded_count = 0
    
    for patient in data:
        has_all = True
        for ep in endpoints:
            if ep == 'survival':
                # For survival, check both OS_MONTHS and DEATH_EVENT_OC
                survival_keys = endpoint_map[ep]
                if not all(key in patient and patient[key] is not None for key in survival_keys):
                    has_all = False
                    break
            else:
                key = endpoint_map[ep]
                if key not in patient or patient[key] is None:
                    has_all = False
                    break
        
        if has_all:
            filtered_data.append(patient)
        else:
            excluded_count += 1
    
    print(f"Filtered patients: kept {len(filtered_data)}, excluded {excluded_count} patients missing required endpoints")
    return filtered_data

def validate_target_values(data: List[Dict], endpoints: List[str], strict_validation: bool = False) -> None:
    """
    Validate target values are within expected ranges
    
    Args:
        data: List of patient data dictionaries
        endpoints: List of endpoints to validate
        strict_validation: If True, raise error on invalid values. If False, warn only.
    """
    endpoint_map = {
        'os6': 'OS_6', 'os24': 'OS_24',
        'stage_t': 'STAGE_DIAGNOSIS_T',
        'stage_n': 'STAGE_DIAGNOSIS_N', 
        'stage_m': 'STAGE_DIAGNOSIS_M',
        'survival': ['OS_MONTHS', 'DEATH_EVENT_OC']
    }
    
    invalid_patients = []
    
    for patient in data:
        patient_id = patient.get('patient_id', 'unknown')
        
        for ep in endpoints:
            key = endpoint_map[ep]
            if key in patient and patient[key] is not None:
                value = patient[key]
                
                # Check value ranges based on endpoint type
                if ep in ['os6', 'os24', 'stage_m']:  # Binary endpoints
                    if not (0.0 <= float(value) <= 1.0):
                        invalid_patients.append({
                            'patient_id': patient_id,
                            'endpoint': ep,
                            'value': value,
                            'expected_range': '[0.0, 1.0]',
                            'type': 'binary'
                        })
                elif ep in ['stage_t', 'stage_n']:  # Multiclass endpoints
                    if not (0 <= int(value) <= 10):  # Reasonable range for staging
                        invalid_patients.append({
                            'patient_id': patient_id,
                            'endpoint': ep,
                            'value': value,
                            'expected_range': '[0, 10]',
                            'type': 'multiclass'
                        })
    
    if invalid_patients:
        error_msg = f"Found {len(invalid_patients)} invalid target values:\n"
        for inv in invalid_patients:
            error_msg += f"  Patient {inv['patient_id']}: {inv['endpoint']} = {inv['value']} (expected {inv['expected_range']} for {inv['type']})\n"
        
        if strict_validation:
            raise ValueError(error_msg)
        else:
            print(f"WARNING: {error_msg}")

def create_data_loaders(
    pkl_files: List[str],
    split_json: Dict = None,
    val_split: float = 0.2,
    batch_size: int = 16,
    num_workers: int = 0,
    model_type: str = 'autoencoder',
    endpoints: List[str] = ['os6', 'os24'],
    random_state: int = 42,
    validate_targets: bool = False,
    enable_pet: bool = False,
    pet_target_height: int = 300
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict, Optional[PETPreprocessor]]:
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
        validate_targets: If True, validate target values and raise error on invalid values
        enable_pet: If True, enable PET data processing
        pet_target_height: Target height for PET image resizing
    
    Returns:
        train_loader, val_loader, test_loader, class_weights, pet_preprocessor
    """
    # Load all data
    all_data = load_pkl_files(pkl_files)
    
    # Set up PET preprocessing if enabled
    pet_preprocessor = None
    if enable_pet:
        # Compute aspect ratio from training data
        aspect_ratio = compute_pet_aspect_ratio(all_data)
        pet_preprocessor = PETPreprocessor(target_height=pet_target_height, aspect_ratio=aspect_ratio)
        pet_preprocessor.set_target_dimensions(aspect_ratio)
        print(f"PET preprocessing enabled with target dimensions: {pet_preprocessor.target_width}x{pet_preprocessor.target_height}")
    
    # Filter out patients with NaN values in features (before endpoint filtering)
    all_data = filter_patients_with_nan_features(all_data)
    
    # Filter patients to keep only those with all required endpoints
    all_data = filter_patients_with_all_endpoints(all_data, endpoints)
    
    # Validate target values if requested
    if validate_targets:
        validate_target_values(all_data, endpoints, strict_validation=True)
    
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
    
    # Since we've already filtered patients with all endpoints, we can set include_missing_endpoints=False for all datasets
    train_dataset = MedicalImagingDataset(train_data, include_missing_endpoints=False, endpoints=endpoints, pet_preprocessor=pet_preprocessor)
    val_dataset = MedicalImagingDataset(val_data, include_missing_endpoints=False, endpoints=endpoints, pet_preprocessor=pet_preprocessor)
    test_dataset = MedicalImagingDataset(test_data, include_missing_endpoints=False, endpoints=endpoints, pet_preprocessor=pet_preprocessor)
    
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
    
    return train_loader, val_loader, test_loader, class_weights, pet_preprocessor