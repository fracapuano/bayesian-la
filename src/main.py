#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script to run the Laplace approximation experiment.

This experiment evaluates the effectiveness of the Laplace approximation in Bayesian neural networks
by analyzing its performance in networks with different curvature characteristics, specifically
comparing networks with and without skip connections.
"""

import os
import sys
import argparse
import torch
from evaluation.experiment import run_experiment

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Experiment to assess the applicability of Laplace approximation in deep learning."
    )
    
    parser.add_argument('--dataset', type=str, default='mnist', 
                        help='Dataset to use (default: mnist)')
    parser.add_argument('--batch-size', type=int, default=128, 
                        help='Batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=20, 
                        help='Number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-3, 
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed (default: 42)')
    parser.add_argument('--dropout-rate', type=float, default=0.1, 
                        help='Dropout rate for MC Dropout (default: 0.1)')
    parser.add_argument('--n-samples', type=int, default=20, 
                        help='Number of Monte Carlo samples (default: 20)')
    parser.add_argument('--hessian-structure', type=str, default='diag', 
                        choices=['diag', 'kron', 'full'], 
                        help='Hessian structure for Laplace approximation (default: diag)')
    parser.add_argument('--subset-of-weights', type=str, default='all', 
                        choices=['all', 'last_layer'], 
                        help='Subset of weights for Laplace approximation (default: all)')
    parser.add_argument('--no-cuda', action='store_true', 
                        help='Disable CUDA even if available')
    
    return parser.parse_args()

def main():
    """Run the main experiment."""
    args = parse_args()
    
    # Determine device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Prepare config
    config = {
        'dataset': args.dataset,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'seed': args.seed,
        'dropout_rate': args.dropout_rate,
        'n_samples': args.n_samples,
        'hessian_structure': args.hessian_structure,
        'subset_of_weights': args.subset_of_weights,
        'device': str(device)
    }
    
    # Run experiment
    run_experiment(config)

if __name__ == "__main__":
    main() 