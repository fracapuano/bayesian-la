import os
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import our modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data import get_mnist_dataloaders, downsample
from utils.training import train_models
from models.networks import ResNet3, ResNet5, ResNet10, ResNet20
from utils.bayesian import (
    create_bayesian_models, evaluate_models, evaluate_models_posterior_quality, evaluate_models_smoothness
)


def experiment(config=None):
    """Run the Laplace approximation experiment.
    
    Args:
        config: Configuration dictionary (for wandb)
    """
    
    # Create experiment ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = f"laplace_experiment_{timestamp}"

    # Initialize wandb if not provided
    if config is None:
        config = {
            'dataset': 'mnist',
            'batch_size': 512,
            'epochs': 1,
            'lr': 1e-3,
            'seed': 42,
            'dropout_rate': 0.2,
            'n_samples': 20,
            'hessian_structure': 'diag',
            'subset_of_weights': 'last_layer',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'test_downsampling': 0.05,
            'fit_downsampling': None
        }
        
    # Set random seed for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
    
    # Set device
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    
    # Initialize wandb
    wandb.init(
        project="bayesian-la-experiment",
        name=exp_id,
        config=config
    )
    
    # Get data loaders
    train_loader, valid_loader, test_loader = get_mnist_dataloaders(
        batch_size=config['batch_size']
    )
    test_loader = downsample(test_loader, downsampling=config['test_downsampling'])

    # Define models, optimizers, and schedulers
    models = {}
    models['resnet3_with_skip'] = ResNet3(use_skip=True)
    models['resnet3_without_skip'] = ResNet3(use_skip=False)
    models['resnet5_with_skip'] = ResNet5(use_skip=True)
    models['resnet5_without_skip'] = ResNet5(use_skip=False)
    models['resnet10_with_skip'] = ResNet10(use_skip=True)
    models['resnet10_without_skip'] = ResNet10(use_skip=False)
    models['resnet20_with_skip'] = ResNet20(use_skip=True)
    models['resnet20_without_skip'] = ResNet20(use_skip=False)

    # Define and train models
    models, histories = train_models(models, train_loader, valid_loader, test_loader, config, device, exp_id)
    
    # Define MC Dropout and Laplace Approximation models
    bayesian_models = create_bayesian_models(models, train_loader, config, device, fit_downsampling=config['fit_downsampling'])

    # Evaluate all models : Test Accuracy
    results = {}
    results |= evaluate_models(models, bayesian_models, test_loader, config, device)
    
    # Evaluate all models : Posterior Quality
    results |= evaluate_models_posterior_quality(models, bayesian_models, test_loader, config, device)
    
    # Evaluate all models : Loss Landscape Smoothness
    results |= evaluate_models_smoothness(models, test_loader, device)
    
    wandb.finish()



if __name__ == "__main__":
    experiment() 