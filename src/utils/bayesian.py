import copy
import gc
import torch
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import wandb
from utils.data import downsample


from models.bayesian_models import LaplaceApproximation, MCDropout


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
                approx_mean = approx_model.predict(data, n_samples=n_samples)
            else:
                approx_mean = approx_model.predict(data, n_samples=n_samples)

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


def compute_posterior_quality(approx_model, data_loader, device, n_samples=20, n_bins=10):
    """
    Measures the quality of the approximate posterior in a multiclass MNIST classification task.

    Computes:
    - Negative Log-Likelihood (NLL)
    - Expected Calibration Error (ECE)

    Args:
        approx_model: The approximate Bayesian neural network.
        data_loader: DataLoader for the dataset.
        device: The device (cuda or cpu) to run computations.
        n_samples: Number of samples for Monte Carlo estimation.
        n_bins: Number of bins for the ECE computation.

    Returns:
        nll (float): Negative Log-Likelihood of the approximate posterior.
        ece (float): Expected Calibration Error.
    """
    
    nll_total = 0
    total_samples = 0
    confidences = []
    accuracies = []

    with tqdm(data_loader, desc="[Posterior Quality]", leave=False) as pbar:
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            total_samples += target.size(0)

            # Get approximate posterior
            if isinstance(approx_model, LaplaceApproximation):
                pred_probs = approx_model.predict(data, n_samples=n_samples)
            elif isinstance(approx_model, MCDropout):
                pred_probs, _ = approx_model.predict(data, n_samples=n_samples)
            else: 
                logits = approx_model(data)
                pred_probs = F.softmax(logits, dim=1)

            # Compute Negative Log-Likelihood (NLL)
            nll_total += F.nll_loss(torch.log(pred_probs), target, reduction='sum').item()

            # Compute Expected Calibration Error (ECE)
            pred_conf, pred_class = pred_probs.max(dim=1)  # Get max confidence & predicted class
            correct = pred_class.eq(target).float()

            confidences.extend(pred_conf.cpu().detach().numpy())
            accuracies.extend(correct.cpu().detach().numpy())

    # Compute final metrics
    nll = nll_total / total_samples
    ece = compute_ece(np.array(confidences), np.array(accuracies), n_bins)

    return nll, ece


def compute_ece(confidences, accuracies, n_bins=10):
    """
    Computes the Expected Calibration Error (ECE).
    
    Args:
        confidences: Array of model confidences.
        accuracies: Array of corresponding accuracy values.
        n_bins: Number of bins for calibration.

    Returns:
        ece (float): Expected Calibration Error.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(conf_in_bin - acc_in_bin) * prop_in_bin

    return ece


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
            perturbed_model = copy.deepcopy(model)
            perturbed_model.to(device)

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

            del perturbed_model
            gc.collect()
            torch.cuda.empty_cache()

    avg_curvature = np.mean(all_curvatures)
    return avg_curvature





def create_bayesian_models(models, train_loader, config, device, fit_downsampling=None):
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

        # Downsample for faster fit
        train_loader = downsample(train_loader, downsampling=fit_downsampling)

        # Fit Laplace approximation
        print("Fitting Laplace approximation...")
        bayesian_models[f'{model_name}_laplace'].fit(
            train_loader=train_loader,
            device=device,
            subset_of_weights=config['subset_of_weights'],
            hessian_structure=config['hessian_structure']
        )

    return bayesian_models


def evaluate_models(models, bayesian_models, test_loader, config, device):
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

    return results


def evaluate_models_posterior_quality(models, bayesian_models, test_loader, config, device):
    # --- Posterior Quality Comparison ---
    print("\n--- Evaluating Posterior Quality ---")
    results = {}

    # Measure KL divergence between MC Dropout and Laplace
    for model_name, model in models.items():
        mc_model = bayesian_models[f'{model_name}_mc_dropout']
        la_model = bayesian_models[f'{model_name}_laplace']

        kl_div, js_div = evaluate_posterior_quality(
            true_model=mc_model,
            approx_model=la_model,
            data_loader=test_loader,
            device=device,
            n_samples=config['n_samples']
        )
        map_nll, map_ece = compute_posterior_quality(model, test_loader, device)
        la_nll, la_ece = compute_posterior_quality(la_model, test_loader, device)
        mc_nll, mc_ece = compute_posterior_quality(mc_model, test_loader, device)

        results[model_name] = {
            'kl_div': kl_div,
            'js_div': js_div,
            'map_nll': map_nll,
            'map_ece': map_ece,
            'la_nll': la_nll,
            'la_ece': la_ece,
            'mc_nll': mc_nll,
            'mc_ece': mc_ece
        }

        # Log to wandb
        wandb.log({
            f'{model_name}/kl_divergence': kl_div,
            f'{model_name}/js_divergence': js_div,
            f'{model_name}/map_nll': map_nll,
            f'{model_name}/map_ece': map_ece,
            f'{model_name}/la_nll': la_nll,
            f'{model_name}/la_ece': la_ece,
            f'{model_name}/mc_nll': mc_nll,
            f'{model_name}/mc_ece': mc_ece
        })

        print(f"{model_name} - MAP NLL: {map_nll:.3f}, MAP ECE: {map_ece:.3f}, LA NLL: {la_nll:.3f}, LA ECE: {la_ece:.3f}, MC NLL: {mc_nll:.3f}, MC ECE: {mc_ece:.3f}, KL Div: {kl_div:.4f}, JS Div: {js_div:.4f}")

    return results


def evaluate_models_smoothness(models, test_loader, device, smoothness_n_samples=100):
    # --- Loss Landscape Smoothness Analysis ---
    print("\n--- Loss Landscape Smoothness Analysis ---")
    results = {}
    
    smoothness = {}
    for model_name, model in models.items():
        smoothness[model_name] = measure_smoothness(
            model=model,
            data_loader=test_loader,
            device=device,
            n_samples=smoothness_n_samples
        )
        
        # Log to wandb
        wandb.log({
            f'{model_name}/smoothness': smoothness[model_name]
        })
        
        print(f"{model_name} Smoothness: {smoothness[model_name]:.6f}")
        results[model_name] = {
            'smoothness': smoothness[model_name]
        }

    return results
