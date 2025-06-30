import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from typing import Dict, Any

def create_optimizer(
    model: torch.nn.Module,
    optimizer_type: str = 'adam',
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Create optimizer for the model
    
    Args:
        model: PyTorch model
        optimizer_type: Type of optimizer ('adam', 'sgd', 'adamw')
        learning_rate: Learning rate
        weight_decay: Weight decay (L2 regularization)
        **kwargs: Additional optimizer-specific arguments
    
    Returns:
        optimizer: PyTorch optimizer
    """
    
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8)
        )
    elif optimizer_type.lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8)
        )
    elif optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=kwargs.get('momentum', 0.9),
            nesterov=kwargs.get('nesterov', True)
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer

def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = 'plateau',
    **kwargs
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler ('plateau', 'step', 'cosine')
        **kwargs: Additional scheduler-specific arguments
    
    Returns:
        scheduler: PyTorch learning rate scheduler
    """
    
    if scheduler_type.lower() == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get('mode', 'min'),
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 10),
            min_lr=kwargs.get('min_lr', 1e-6)
        )
    elif scheduler_type.lower() == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 30),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_type.lower() == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 100),
            eta_min=kwargs.get('eta_min', 1e-6)
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler

class EarlyStopping:
    """Early stopping utility to stop training when validation loss stops improving"""
    
    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        verbose: bool = True
    ):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as an improvement
            restore_best_weights: Whether to restore model weights from best epoch
            verbose: Whether to print early stopping info
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        """
        Check if training should be stopped
        
        Args:
            val_loss: Current validation loss
            model: PyTorch model
        
        Returns:
            bool: True if training should be stopped
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.verbose:
                print(f"Early stopping triggered after {self.patience} epochs without improvement")
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                if self.verbose:
                    print("Restored best weights")
            return True
        
        return False

class OptimizerManager:
    """Manager class for optimizer, scheduler, and early stopping"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer_config: Dict[str, Any],
        scheduler_config: Dict[str, Any] = None,
        early_stopping_config: Dict[str, Any] = None
    ):
        """
        Args:
            model: PyTorch model
            optimizer_config: Configuration for optimizer
            scheduler_config: Configuration for scheduler (optional)
            early_stopping_config: Configuration for early stopping (optional)
        """
        self.model = model
        
        # Create optimizer
        self.optimizer = create_optimizer(model, **optimizer_config)
        
        # Create scheduler (optional)
        self.scheduler = None
        if scheduler_config:
            self.scheduler = create_scheduler(self.optimizer, **scheduler_config)
        
        # Create early stopping (optional)
        self.early_stopping = None
        if early_stopping_config:
            self.early_stopping = EarlyStopping(**early_stopping_config)
    
    def step(self, loss: float = None) -> bool:
        """
        Perform optimization step and update scheduler
        
        Args:
            loss: Loss value for plateau scheduler and early stopping
        
        Returns:
            bool: True if early stopping is triggered
        """
        self.optimizer.step()
        
        # Update scheduler
        if self.scheduler is not None:
            if isinstance(self.scheduler, ReduceLROnPlateau):
                if loss is not None:
                    self.scheduler.step(loss)
            else:
                self.scheduler.step()
        
        # Check early stopping
        if self.early_stopping is not None and loss is not None:
            return self.early_stopping(loss, self.model)
        
        return False
    
    def zero_grad(self):
        """Zero gradients"""
        self.optimizer.zero_grad()
    
    def get_current_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for saving"""
        state = {
            'optimizer': self.optimizer.state_dict()
        }
        if self.scheduler is not None:
            state['scheduler'] = self.scheduler.state_dict()
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dictionary"""
        self.optimizer.load_state_dict(state_dict['optimizer'])
        if self.scheduler is not None and 'scheduler' in state_dict:
            self.scheduler.load_state_dict(state_dict['scheduler'])