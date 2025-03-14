# Bayesian-LA: Laplace Approximation for Neural Networks

This repository contains code for experiments studying the impact of the Laplace Approximation for intractable posteriors in Bayesian Machine Learning, particularly focusing on how architectural choices like skip connections affect the quality of the Laplace approximation.

## Experiment Overview

This experiment evaluates the effectiveness of the Laplace approximation in Bayesian neural networks by analyzing its performance in networks with different curvature characteristics. Specifically, we investigate how skip connections influence the smoothness of the posterior landscape and, consequently, the quality of the Laplace approximation.

### Hypothesis

The Laplace approximation assumes a relatively smooth posterior distribution, as it relies on the curvature of the loss function to estimate the Hessian. Skip connections in neural networks are known to significantly impact the loss landscape, producing smoother and more convex surfaces. We expect the Laplace approximation to perform better in networks with skip connections compared to those without.

## Setup

### Requirements

To install the required dependencies:

```bash
pip install -r requirements.txt
```

### Experiment Structure

- `src/models/networks.py`: Defines the neural network architectures (ResNet-5 with and without skip connections)
- `src/utils/`: Contains utilities for data loading, training, and Bayesian inference
- `src/evaluation/experiment.py`: Implements the main experimental workflow
- `src/main.py`: Entry point for running the experiment with command-line arguments

## Running the Experiment

To run the experiment with default parameters:

```bash
python src/main.py
```

### Command-line Arguments

The experiment can be customized with the following command-line arguments:

- `--dataset`: Dataset to use (default: mnist)
- `--batch-size`: Batch size for training (default: 128)
- `--epochs`: Number of epochs to train (default: 20)
- `--lr`: Learning rate (default: 1e-3)
- `--seed`: Random seed (default: 42)
- `--dropout-rate`: Dropout rate for MC Dropout (default: 0.1)
- `--n-samples`: Number of Monte Carlo samples (default: 20)
- `--hessian-structure`: Hessian structure for Laplace approximation (choices: diag, kron, full; default: diag)
- `--subset-of-weights`: Subset of weights for Laplace approximation (choices: all, last_layer; default: all)
- `--no-cuda`: Disable CUDA even if available

Example:

```bash
python src/main.py --epochs 30 --batch-size 256 --hessian-structure kron --subset-of-weights last_layer
```

## Experiment Output

The experiment will:

1. Train two ResNet-5 models (with and without skip connections)
2. Apply the Laplace approximation to both models using the `laplace-torch` package
3. Use MC Dropout as a baseline Bayesian method for comparison
4. Evaluate the quality of the Laplace approximation in both architectures
5. Measure the smoothness of the posterior landscape
6. Log all results and visualizations to Weights & Biases (wandb)

## Key Metrics

- **Test Accuracy**: Classification accuracy on the test set
- **Smoothness**: Measure of the loss landscape curvature
- **KL Divergence**: KL divergence between MC Dropout (baseline) and Laplace approximation
- **JS Divergence**: Jensen-Shannon divergence between the two posterior approximations
- **Predictive Entropy**: Entropy of the predictive distribution (uncertainty measure)

## Weights & Biases Integration

All experiment results are automatically logged to Weights & Biases. To view the results, create a wandb account and run the experiment.

## License

This project is licensed under the terms of the Apache 2.0 license.
