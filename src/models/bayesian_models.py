import torch
import numpy as np
from laplace import Laplace
from tqdm import tqdm
import copy

class MCDropout:
    """Monte Carlo Dropout for approximate Bayesian inference.

    This is a simple baseline Bayesian method that uses dropout at inference time
    to approximate a Bayesian Neural Network.

    Reference:
        Gal, Y., & Ghahramani, Z. (2016). Dropout as a bayesian approximation:
        Representing model uncertainty in deep learning.
    """

    def __init__(self, model, dropout_rate=0.2):
        """Initialize MC Dropout wrapper.

        Args:
            model: PyTorch model
            dropout_rate: Dropout rate to use for uncertainty estimation
        """
        self.model = copy.deepcopy(model)
        self.dropout_rate = dropout_rate

        # Add dropout after each layer
        model.layer1.conv2 = torch.nn.Sequential(
            model.layer1.conv2,
            torch.nn.Dropout(p=self.dropout_rate)
        )
        if getattr(model, "layer2", None) and getattr(model.layer2, "conv2", None):
            model.layer2.conv2 = torch.nn.Sequential(
                model.layer2.conv2,
                torch.nn.Dropout(p=self.dropout_rate)
            )
        if getattr(model, "layer3", None) and getattr(model.layer2, "conv2", None):
            model.layer3.conv2 = torch.nn.Sequential(
                model.layer3.conv2,
                torch.nn.Dropout(p=self.dropout_rate)
            )
        if getattr(model, "layer4", None) and getattr(model.layer2, "conv2", None):
            model.layer4.conv2 = torch.nn.Sequential(
                model.layer4.conv2,
                torch.nn.Dropout(p=self.dropout_rate)
            )
        if getattr(model, "layer5", None) and getattr(model.layer2, "conv2", None):
            model.layer5.conv2 = torch.nn.Sequential(
                model.layer5.conv2,
                torch.nn.Dropout(p=self.dropout_rate)
            )
        if getattr(model, "layer6", None) and getattr(model.layer2, "conv2", None):
            model.layer6.conv2 = torch.nn.Sequential(
                model.layer6.conv2,
                torch.nn.Dropout(p=self.dropout_rate)
            )
        if getattr(model, "layer7", None) and getattr(model.layer2, "conv2", None):
            model.layer7.conv2 = torch.nn.Sequential(
                model.layer7.conv2,
                torch.nn.Dropout(p=self.dropout_rate)
            )
        if getattr(model, "layer8", None) and getattr(model.layer2, "conv2", None):
            model.layer8.conv2 = torch.nn.Sequential(
                model.layer8.conv2,
                torch.nn.Dropout(p=self.dropout_rate)
            )
        if getattr(model, "layer9", None) and getattr(model.layer2, "conv2", None):
            model.layer9.conv2 = torch.nn.Sequential(
                model.layer9.conv2,
                torch.nn.Dropout(p=self.dropout_rate)
            )

        # Enable dropout at inference time
        def _enable_dropout(m):
            if type(m) == torch.nn.Dropout:
                m.train()

        self.model.eval()
        self.model.apply(_enable_dropout)

    def predict(self, x, n_samples=20):
        """Get predictive distribution.

        Args:
            x: Input tensor
            n_samples: Number of MC samples

        Returns:
            mean: Mean of predictions
            variance: Variance of predictions
        """
        self.model.eval()
        samples = []

        with torch.no_grad():
            for _ in range(n_samples):
                output = self.model(x)
                samples.append(torch.softmax(output, dim=1))

        samples = torch.stack(samples)  # [n_samples, batch_size, num_classes]
        mean = samples.mean(dim=0)
        variance = samples.var(dim=0)

        return mean, variance

    def evaluate(self, data_loader, device, n_samples=20):
        """Evaluate model on data loader.

        Args:
            data_loader: Data loader
            device: Device to evaluate on
            n_samples: Number of MC samples

        Returns:
            accuracy, predictive_entropy
        """
        correct = 0
        total = 0
        all_entropies = []

        with tqdm(data_loader, desc="[MCDropout Eval]", leave=False) as pbar:
            for data, target in pbar:
                data, target = data.to(device), target.to(device)

                # Get predictions
                mean, variance = self.predict(data, n_samples=n_samples)

                # Calculate accuracy
                pred = mean.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += data.size(0)

                # Calculate entropy
                entropy = -torch.sum(mean * torch.log(mean + 1e-10), dim=1)
                all_entropies.append(entropy)

                pbar.set_postfix({'acc': 100. * correct / total})

        accuracy = 100. * correct / total
        predictive_entropy = torch.cat(all_entropies).mean().item()

        return accuracy, predictive_entropy


class LaplaceApproximation:
    """Laplace Approximation using laplace-pytorch.

    Reference:
        https://github.com/AlexImmer/Laplace
    """

    def __init__(self, model, likelihood='classification'):
        """Initialize Laplace wrapper.

        Args:
            model: PyTorch model
            likelihood: 'classification' or 'regression'
        """
        self.model = model
        self.likelihood = likelihood
        self.la = None  # Will be initialized after fitting

    def fit(self, train_loader, device, subset_of_weights='last_layer', hessian_structure='diag'):
        """Fit Laplace approximation.

        Args:
            train_loader: Training data loader
            device: Device to fit on
            subset_of_weights: Which weights to consider for the Laplace approximation
            hessian_structure: Structure of the Hessian approximation

        Returns:
            self
        """
        self.la = Laplace(
            model=self.model,
            likelihood=self.likelihood,
            subset_of_weights=subset_of_weights,
            hessian_structure=hessian_structure
        )

        # Fit Laplace approximation
        self.la.fit(train_loader)

        # Optimize prior decision
        print("Optimizing laplace prior precision...")
        self.la.optimize_prior_precision()

        return self

    def predict(self, x, n_samples=20, link_approx='probit'):
        """Get predictive distribution.

        Args:
            x: Input tensor
            n_samples: Number of samples for Monte Carlo approximation
            link_approx: Link approximation method ('mc', 'probit', etc.)

        Returns:
            pred_mean, pred_var
        """
        pred = self.la(
            x=x, pred_type="glm", link_approx=link_approx
        )

        return pred

    def evaluate(self, data_loader, device, n_samples=20, link_approx='mc'):
        """Evaluate model on data loader.

        Args:
            data_loader: Data loader
            device: Device to evaluate on
            n_samples: Number of samples
            link_approx: Link approximation method

        Returns:
            accuracy, predictive_entropy
        """
        correct = 0
        total = 0
        all_entropies = []

        with tqdm(data_loader, desc="[Laplace Eval]", leave=False) as pbar:
            for data, target in pbar:
                data, target = data.to(device), target.to(device)

                # Get predictions
                pred_mean = self.predict(data, n_samples=n_samples, link_approx=link_approx)

                # Calculate accuracy
                pred = pred_mean.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += data.size(0)

                # Calculate entropy (uncertainty)
                entropy = -torch.sum(pred_mean * torch.log(pred_mean + 1e-10), dim=1)
                all_entropies.append(entropy)

                pbar.set_postfix({'acc': 100. * correct / total})

        accuracy = 100. * correct / total
        predictive_entropy = torch.cat(all_entropies).mean().item()

        return accuracy, predictive_entropy


