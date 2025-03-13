import torch
import numpy as np
from laplace import Laplace
from tqdm import tqdm

class MCDropout:
    """Monte Carlo Dropout for approximate Bayesian inference.
    
    This is a simple baseline Bayesian method that uses dropout at inference time
    to approximate a Bayesian Neural Network.
    
    Reference:
        Gal, Y., & Ghahramani, Z. (2016). Dropout as a bayesian approximation:
        Representing model uncertainty in deep learning.
    """
    
    def __init__(self, model, dropout_rate=0.1):
        """Initialize MC Dropout wrapper.
        
        Args:
            model: PyTorch model
            dropout_rate: Dropout rate to use for uncertainty estimation
        """
        self.model = model
        self.dropout_rate = dropout_rate
        
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
    
    def fit(self, train_loader, device, subset_of_weights='all', hessian_structure='diag'):
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
        
        # Convert data to list for Laplace
        x_train, y_train = [], []
        for data, target in tqdm(train_loader, desc="[Preparing Data for LA]"):
            x_train.append(data.to(device))
            y_train.append(target.to(device))
        
        # Fit Laplace approximation
        self.la.fit(x_train, y_train)
        
        return self
    
    def predict(self, x, n_samples=20, link_approx='mc'):
        """Get predictive distribution.
        
        Args:
            x: Input tensor
            n_samples: Number of samples for Monte Carlo approximation
            link_approx: Link approximation method ('mc', 'probit', etc.)
            
        Returns:
            pred_mean, pred_var
        """
        pred_mean, pred_var = self.la.predictive_samples(
            x=x, n_samples=n_samples, link_approx=link_approx
        )
        return pred_mean, pred_var
    
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
                pred_mean, pred_var = self.predict(data, n_samples=n_samples, link_approx=link_approx)
                
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

def evaluate_posterior_quality(true_model, approx_model, data_loader, device, n_samples=20):
    """Compare approximated posterior with true posterior.
    
    Args:
        true_model: True posterior model (usually MC Dropout as reference)
        approx_model: Approximated posterior model (Laplace)
        data_loader: Data loader
        device: Device to evaluate on
        n_samples: Number of samples
        
    Returns:
        kl_divergence, js_divergence
    """
    all_kl_divs = []
    all_js_divs = []
    
    with tqdm(data_loader, desc="[Posterior Quality]", leave=False) as pbar:
        for data, _ in pbar:
            data = data.to(device)
            
            # Get true posterior
            true_mean, _ = true_model.predict(data, n_samples=n_samples)
            
            # Get approximate posterior
            if isinstance(approx_model, LaplaceApproximation):
                approx_mean, _ = approx_model.predict(data, n_samples=n_samples)
            else:
                approx_mean, _ = approx_model.predict(data, n_samples=n_samples)
            
            # Calculate KL divergence
            kl_div = torch.sum(true_mean * (torch.log(true_mean + 1e-10) - torch.log(approx_mean + 1e-10)), dim=1)
            all_kl_divs.append(kl_div)
            
            # Calculate JS divergence
            m = 0.5 * (true_mean + approx_mean)
            js_div = 0.5 * torch.sum(true_mean * (torch.log(true_mean + 1e-10) - torch.log(m + 1e-10)), dim=1) + \
                    0.5 * torch.sum(approx_mean * (torch.log(approx_mean + 1e-10) - torch.log(m + 1e-10)), dim=1)
            all_js_divs.append(js_div)
    
    avg_kl_div = torch.cat(all_kl_divs).mean().item()
    avg_js_div = torch.cat(all_js_divs).mean().item()
    
    return avg_kl_div, avg_js_div

def measure_smoothness(model, data_loader, device, n_samples=100):
    """Measure the smoothness of the posterior landscape.
    
    Args:
        model: Neural network model
        data_loader: Data loader
        device: Device to evaluate on
        n_samples: Number of random perturbations
        
    Returns:
        avg_curvature: Average curvature measure
    """
    model.eval()
    all_curvatures = []
    epsilon = 1e-4  # Small perturbation
    
    # Get a single batch
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        break
    
    with torch.no_grad():
        # Get original prediction
        original_output = model(data)
        
        for _ in range(n_samples):
            # Create a perturbed version of the model
            perturbed_model = type(model)(use_skip=model.use_skip)
            perturbed_model.to(device)
            perturbed_model.load_state_dict(model.state_dict())
            
            # Add small random perturbation to parameters
            for param in perturbed_model.parameters():
                param.data += epsilon * torch.randn_like(param)
            
            # Get perturbed prediction
            perturbed_output = perturbed_model(data)
            
            # Calculate Euclidean distance
            output_diff = (original_output - perturbed_output).pow(2).sum(dim=1).sqrt()
            
            # Normalize by perturbation size
            curvature = output_diff.mean().item() / epsilon
            all_curvatures.append(curvature)
    
    avg_curvature = np.mean(all_curvatures)
    return avg_curvature 