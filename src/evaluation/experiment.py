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

from models.networks import ResNet5
from utils.data import get_mnist_dataloaders
from utils.training import train_model, evaluate
from utils.bayesian import (
    MCDropout, LaplaceApproximation, 
    evaluate_posterior_quality, measure_smoothness
)

def run_experiment(config=None):
    """Run the Laplace approximation experiment.
    
    Args:
        config: Configuration dictionary (for wandb)
    """
    # Initialize wandb if not provided
    if config is None:
        config = {
            'dataset': 'mnist',
            'batch_size': 128,
            'epochs': 20,
            'lr': 1e-3,
            'seed': 42,
            'dropout_rate': 0.1,
            'n_samples': 20,
            'hessian_structure': 'diag',
            'subset_of_weights': 'all',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
    # Set random seed for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
    
    # Create experiment ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = f"laplace_experiment_{timestamp}"
    
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
    
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    
    # Define models, optimizers, and schedulers
    models = {}
    optimizers = {}
    schedulers = {}
    
    # Create models with and without skip connections
    models['with_skip'] = ResNet5(use_skip=True)
    models['without_skip'] = ResNet5(use_skip=False)
    
    # Add dropout for MCDropout baseline
    for model_name in models:
        # Add dropout after each layer
        models[model_name].layer1.conv2 = torch.nn.Sequential(
            models[model_name].layer1.conv2,
            torch.nn.Dropout(p=config['dropout_rate'])
        )
        models[model_name].layer2.conv2 = torch.nn.Sequential(
            models[model_name].layer2.conv2, 
            torch.nn.Dropout(p=config['dropout_rate'])
        )
        
        # Move to device
        models[model_name] = models[model_name].to(device)
        
        # Create optimizer and scheduler
        optimizers[model_name] = Adam(models[model_name].parameters(), lr=config['lr'])
        schedulers[model_name] = CosineAnnealingLR(
            optimizers[model_name], T_max=config['epochs']
        )
    
    # Train models
    histories = {}
    for model_name, model in models.items():
        print(f"\n--- Training {model_name} model ---")
        
        # Set wandb group for proper organization
        wandb.config.update({'model_type': model_name}, allow_val_change=True)
        
        # Train the model
        trained_model, history = train_model(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            optimizer=optimizers[model_name],
            device=device,
            epochs=config['epochs'],
            log_wandb=True,
            scheduler=schedulers[model_name]
        )
        
        # Save history
        histories[model_name] = history
        
        # Log final test accuracy
        test_loss, test_acc = evaluate(trained_model, test_loader, device, phase="Test")
        wandb.log({
            f'{model_name}/test_accuracy': test_acc,
            f'{model_name}/test_loss': test_loss
        })
        print(f"{model_name} Test accuracy: {test_acc:.2f}%")
        
        # Save model
        os.makedirs('models', exist_ok=True)
        torch.save(
            trained_model.state_dict(), 
            f"models/{model_name}_{exp_id}.pt"
        )
    
    # --- Posterior Approximation Evaluation ---
    print("\n--- Evaluating Posterior Approximations ---")
    
    # Create Bayesian models
    bayesian_models = {}
    for model_name, model in models.items():
        print(f"\nProcessing {model_name} model:")
        
        # Create MC Dropout model (baseline)
        print("Creating MC Dropout model...")
        bayesian_models[f'{model_name}_mc_dropout'] = MCDropout(
            model=model,
            dropout_rate=config['dropout_rate']
        )
        
        # Create Laplace Approximation model
        print("Creating Laplace Approximation model...")
        bayesian_models[f'{model_name}_laplace'] = LaplaceApproximation(
            model=model,
            likelihood='classification'
        )
        
        # Fit Laplace approximation
        print("Fitting Laplace approximation...")
        bayesian_models[f'{model_name}_laplace'].fit(
            train_loader=train_loader,
            device=device,
            subset_of_weights=config['subset_of_weights'],
            hessian_structure=config['hessian_structure']
        )
    
    # Evaluate all models
    results = {}
    for model_name, model in bayesian_models.items():
        print(f"\nEvaluating {model_name}...")
        
        # Evaluate on test set
        accuracy, entropy = model.evaluate(
            data_loader=test_loader,
            device=device,
            n_samples=config['n_samples']
        )
        
        results[model_name] = {
            'accuracy': accuracy,
            'entropy': entropy
        }
        
        # Log to wandb
        wandb.log({
            f'{model_name}/accuracy': accuracy,
            f'{model_name}/entropy': entropy
        })
        
        print(f"{model_name} - Accuracy: {accuracy:.2f}%, Entropy: {entropy:.4f}")
    
    # --- Posterior Quality Comparison ---
    print("\n--- Evaluating Posterior Quality ---")
    
    # Measure KL divergence between MC Dropout and Laplace
    for model_type in ['with_skip', 'without_skip']:
        true_model = bayesian_models[f'{model_type}_mc_dropout']
        approx_model = bayesian_models[f'{model_type}_laplace']
        
        kl_div, js_div = evaluate_posterior_quality(
            true_model=true_model,
            approx_model=approx_model,
            data_loader=test_loader,
            device=device,
            n_samples=config['n_samples']
        )
        
        results[f'{model_type}_posterior_quality'] = {
            'kl_div': kl_div,
            'js_div': js_div
        }
        
        # Log to wandb
        wandb.log({
            f'{model_type}/kl_divergence': kl_div,
            f'{model_type}/js_divergence': js_div
        })
        
        print(f"{model_type} - KL Div: {kl_div:.4f}, JS Div: {js_div:.4f}")
    
    # --- Loss Landscape Smoothness Analysis ---
    print("\n--- Loss Landscape Smoothness Analysis ---")
    
    smoothness = {}
    for model_name, model in models.items():
        smoothness[model_name] = measure_smoothness(
            model=model,
            data_loader=test_loader,
            device=device,
            n_samples=100
        )
        
        # Log to wandb
        wandb.log({
            f'{model_name}/smoothness': smoothness[model_name]
        })
        
        print(f"{model_name} Smoothness: {smoothness[model_name]:.6f}")
    
    # --- Final Comparison and Visualization ---
    print("\n--- Final Comparison ---")
    
    # Create comparison table
    comparison = {
        'Model': ['With Skip', 'Without Skip'],
        'Test Accuracy': [
            results['with_skip_laplace']['accuracy'],
            results['without_skip_laplace']['accuracy']
        ],
        'Smoothness': [
            smoothness['with_skip'], 
            smoothness['without_skip']
        ],
        'KL Divergence': [
            results['with_skip_posterior_quality']['kl_div'],
            results['without_skip_posterior_quality']['kl_div']
        ],
        'JS Divergence': [
            results['with_skip_posterior_quality']['js_div'],
            results['without_skip_posterior_quality']['js_div']
        ]
    }
    
    # Log comparison table to wandb
    wandb.log({
        'comparison_table': wandb.Table(
            data=[[
                comparison['Model'][i],
                comparison['Test Accuracy'][i],
                comparison['Smoothness'][i],
                comparison['KL Divergence'][i],
                comparison['JS Divergence'][i]
            ] for i in range(len(comparison['Model']))],
            columns=['Model', 'Test Accuracy', 'Smoothness', 'KL Divergence', 'JS Divergence']
        )
    })
    
    # Print comparison
    print("\nComparison Table:")
    print(f"{'Model':<15} | {'Test Acc':<10} | {'Smoothness':<12} | {'KL Div':<10} | {'JS Div':<10}")
    print("-" * 65)
    for i in range(len(comparison['Model'])):
        print(
            f"{comparison['Model'][i]:<15} | "
            f"{comparison['Test Accuracy'][i]:<10.2f} | "
            f"{comparison['Smoothness'][i]:<12.6f} | "
            f"{comparison['KL Divergence'][i]:<10.4f} | "
            f"{comparison['JS Divergence'][i]:<10.4f}"
        )
    
    wandb.finish()

if __name__ == "__main__":
    run_experiment() 