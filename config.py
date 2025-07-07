import argparse
import json
from typing import List, Optional

class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Attention-based Neural Network for Medical Imaging')
        self._add_arguments()
    
    def _add_arguments(self):
        # Data arguments
        self.parser.add_argument('--pkl_files', nargs='+', required=True,
                                help='List of .pkl files, each representing a center')
        self.parser.add_argument('--train_test_split_json', type=str, default=None,
                                help='JSON file with TRAIN_SET and TEST_SET patient_ids (optional)')
        self.parser.add_argument('--val_split', type=float, default=0.2,
                                help='Validation split from training data (also used as test split if no JSON provided)')
        self.parser.add_argument('--validate_targets', action='store_true',
                                help='Validate target values are within expected ranges and raise error on invalid values')
        
        # Model arguments
        self.parser.add_argument('--attention_k', type=int, default=32,
                                help='Number of vectors after attention mechanism (k)')
        self.parser.add_argument('--latent_dim', type=int, default=128,
                                help='Latent dimension for autoencoder')
        self.parser.add_argument('--encoder_layers', nargs='+', type=int, 
                                default=[256, 128],
                                help='Number of units in each encoder layer')
        self.parser.add_argument('--predictor_layers', nargs='+', type=int,
                                default=[64, 32],
                                help='Number of units in each predictor layer')
        self.parser.add_argument('--dropout_rate', type=float, default=0.3,
                                help='Dropout rate')
        
        # Training arguments
        self.parser.add_argument('--endpoints', nargs='+', 
                                choices=['os6', 'os24', 'stage_t', 'stage_n', 'stage_m', 'survival'], 
                                default=['os6', 'os24'],
                                help='Which endpoints to train on')
        self.parser.add_argument('--survival_analysis', action='store_true',
                                help='Enable survival analysis mode with Cox Proportional Hazards')
        self.parser.add_argument('--selection_metric', type=str, choices=['loss', 'auc', 'c_index'], 
                                default='loss', help='Metric for best model selection')
        self.parser.add_argument('--batch_size', type=int, default=16,
                                help='Batch size')
        self.parser.add_argument('--epochs', type=int, default=100,
                                help='Number of training epochs')
        self.parser.add_argument('--learning_rate', type=float, default=1e-3,
                                help='Learning rate')
        self.parser.add_argument('--weight_decay', type=float, default=1e-4,
                                help='Weight decay')
        self.parser.add_argument('--reconstruction_weight', type=float, default=1.0,
                                help='Weight for reconstruction loss in autoencoder')
        self.parser.add_argument('--prediction_weight', type=float, default=1.0,
                                help='Weight for prediction loss')
        self.parser.add_argument('--reconstruct_all', action='store_true',
                                help='Reconstruct entire input (176 vectors) instead of only attended features')
        
        # System arguments
        self.parser.add_argument('--device', type=str, default='cuda',
                                help='Device to use (cuda/cpu)')
        self.parser.add_argument('--num_workers', type=int, default=4,
                                help='Number of workers for data loading')
        self.parser.add_argument('--seed', type=int, default=42,
                                help='Random seed')
        self.parser.add_argument('--save_dir', type=str, default='./results',
                                help='Directory to save results')
        
        # Evaluation arguments
        self.parser.add_argument('--reconstruction-evaluation', action='store_true',
                                help='Perform detailed reconstruction evaluation (time-consuming)')
        
        # Model type
        self.parser.add_argument('--model_type', type=str, choices=['autoencoder', 'endtoend', 'survival'],
                                required=True, help='Type of model to train')
    
    def parse_args(self):
        args = self.parser.parse_args()
        
        # Validate survival analysis arguments
        if args.survival_analysis or args.model_type == 'survival':
            if 'survival' not in args.endpoints:
                args.endpoints = ['survival']
            if args.model_type != 'survival':
                print("Warning: --survival-analysis flag set, forcing model_type to 'survival'")
                args.model_type = 'survival'
            if args.selection_metric not in ['loss', 'c_index']:
                print("Warning: For survival analysis, setting selection_metric to 'c_index'")
                args.selection_metric = 'c_index'
        
        return args

def load_train_test_split(json_path: str) -> dict:
    """Load train/test split from JSON file"""
    with open(json_path, 'r') as f:
        split_data = json.load(f)
    return split_data