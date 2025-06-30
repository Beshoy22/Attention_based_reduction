# Attention-based Neural Network for Medical Imaging

A PyTorch-based deep learning framework for medical imaging analysis featuring attention mechanisms, autoencoder architectures, and multi-endpoint prediction capabilities. This framework is specifically designed for processing medical image features and predicting various clinical outcomes.

## Overview

This framework implements attention-based neural networks for medical imaging applications, supporting both autoencoder and end-to-end training approaches. It handles multi-center medical data with various endpoint types including survival outcomes and staging classifications.

## Key Features

### 🧠 Neural Network Architectures

1. **Attention Mechanism** (`utils/models.py:6-47`)
   - Reduces 176 image patches to k representative vectors using learnable attention weights
   - Implements sectioned grouping with softmax-weighted aggregation
   - Configurable attention dimension (default k=32)

2. **Autoencoder Model** (`utils/models.py:143-194`)
   - Combines attention mechanism with encoder-decoder architecture
   - Latent space compression for feature reduction
   - Simultaneous reconstruction and prediction tasks
   - Configurable encoder/decoder layers

3. **End-to-End Model** (`utils/models.py:196-236`)
   - Direct prediction without reconstruction constraint
   - Extracts reduced features from second-to-last layer
   - Optimized for prediction performance

### 📊 Multi-Endpoint Support

The framework supports heterogeneous clinical endpoints:

- **Binary Classification**: 
  - `os6`: 6-month overall survival
  - `os24`: 24-month overall survival  
  - `stage_m`: Metastasis stage (M0/M1)

- **Multi-class Classification**:
  - `stage_t`: Primary tumor stage (T0-T4)
  - `stage_n`: Regional lymph node stage (N0-N3)

### 🗂️ Advanced Data Handling

1. **Multi-Center Support** (`utils/dataloader.py:105-120`)
   - Loads data from multiple medical centers
   - Maintains center information for stratification
   - Handles varying data formats across centers

2. **Intelligent Data Splitting** (`utils/dataloader.py:122-324`)
   - Stratified splitting by center and endpoint combinations
   - Automatic fallback strategies when stratification fails
   - Support for custom train/test splits via JSON configuration
   - Handles missing endpoint values gracefully

3. **Class Balancing** (`utils/dataloader.py:326-355`)
   - Automatic computation of class weights for imbalanced datasets
   - Separate handling for binary and multi-class endpoints
   - Integration with loss functions for balanced training

### 🎯 Sophisticated Loss Functions

1. **Combined Loss** (`utils/losses.py:15-92`)
   - Weighted combination of reconstruction and prediction losses
   - Endpoint-specific loss computation
   - Handles missing endpoint values with masking

2. **Balanced Loss Functions** (`utils/losses.py:6-13`)
   - Class-weighted Binary Cross Entropy for imbalanced data
   - Cross Entropy with class weights for multi-class problems
   - Automatic handling of different endpoint types

### ⚡ Training Infrastructure

1. **Optimizer Management** (`utils/optimizers.py`)
   - Configurable optimizers (Adam, SGD, etc.)
   - Learning rate scheduling (ReduceLROnPlateau, StepLR, etc.)
   - Early stopping with best model restoration

2. **Comprehensive Training Loop** (`utils/train_loop.py`)
   - Integrated training and validation cycles
   - Metric computation for all endpoint types
   - Model checkpointing and best model selection
   - Training history tracking

3. **Evaluation Metrics** 
   - AUC-ROC for binary classification
   - Accuracy and F1-score for all endpoints
   - Cosine similarity for reconstruction quality

### 🛠️ Utility Features

1. **Flexible Configuration** (`config.py`)
   - Command-line argument parsing
   - Comprehensive hyperparameter specification
   - Model type selection (autoencoder vs end-to-end)
   - Training parameter configuration

2. **Visualization Tools** (`utils/visualization.py`)
   - Training curve plotting
   - Loss and metric visualization
   - Model performance analysis

3. **Feature Extraction** (`extract_features.py`)
   - Extract reduced features from trained models
   - Save features for downstream analysis
   - Patient-level feature mapping

## Main Scripts

### Training Scripts

1. **`main_autoencoder.py`** - Training autoencoder models
   ```bash
   python main_autoencoder.py --pkl_files data1.pkl data2.pkl \
                              --model_type autoencoder \
                              --attention_k 32 \
                              --latent_dim 128 \
                              --epochs 100
   ```

2. **`main_endtoend.py`** - Training end-to-end models
   ```bash
   python main_endtoend.py --pkl_files data1.pkl data2.pkl \
                           --model_type endtoend \
                           --attention_k 32 \
                           --epochs 100
   ```

### Analysis Scripts

- **`gridsearch.py`** - Hyperparameter optimization
- **`evaluation.py`** - Model evaluation and metrics computation
- **`inference.py`** - Model inference on new data
- **`analyze_reconstruction.py`** - Reconstruction quality analysis

## Configuration Options

### Model Parameters
- `attention_k`: Number of attention vectors (default: 32)
- `latent_dim`: Autoencoder latent dimension (default: 128)
- `encoder_layers`: Encoder architecture (default: [256, 128])
- `predictor_layers`: Predictor architecture (default: [64, 32])
- `dropout_rate`: Dropout probability (default: 0.3)

### Training Parameters
- `batch_size`: Batch size (default: 16)
- `epochs`: Training epochs (default: 100)
- `learning_rate`: Learning rate (default: 1e-3)
- `weight_decay`: L2 regularization (default: 1e-4)

### Data Parameters
- `endpoints`: Clinical endpoints to predict
- `val_split`: Validation split ratio (default: 0.2)
- `train_test_split_json`: Custom train/test split file

## Data Format

Input data should be pickle files containing lists of patient dictionaries:

```python
{
    'patient_id': str,
    'features': List[torch.Tensor],  # 176 vectors of 512 dimensions
    'OS_6': float,                   # 6-month survival (0/1)
    'OS_24': float,                  # 24-month survival (0/1)
    'STAGE_DIAGNOSIS_T': int,        # T stage (0-4)
    'STAGE_DIAGNOSIS_N': int,        # N stage (0-3)
    'STAGE_DIAGNOSIS_M': float       # M stage (0/1)
}
```

## Output Structure

Results are saved in timestamped directories containing:
- `best_model.pth`: Best model checkpoint
- `training_history.json`: Training metrics and losses
- `test_results.json`: Final evaluation metrics
- `config.json`: Model and training configuration
- `test_reduced_features.npy`: Extracted features (end-to-end mode)

## Requirements

- Python 3.8+
- PyTorch 1.12+
- scikit-learn 1.1+
- NumPy, Pandas, Matplotlib
- See `requirements.txt` for complete dependencies

## Key Advantages

1. **Medical Domain Specific**: Designed for clinical endpoint prediction
2. **Multi-Center Robust**: Handles data from different medical centers
3. **Flexible Architecture**: Support for both reconstruction and direct prediction
4. **Missing Data Handling**: Graceful handling of incomplete clinical data
5. **Comprehensive Evaluation**: Multiple metrics for different endpoint types
6. **Reproducible**: Seed management and deterministic training
7. **Scalable**: Efficient attention mechanism for large feature sets

This framework provides a complete solution for attention-based medical image analysis with robust data handling, flexible model architectures, and comprehensive evaluation capabilities.