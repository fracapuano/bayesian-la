import torch
import wandb
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import os
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR


def train_epoch(model, loader, optimizer, device, epoch):
    """Train model for one epoch.

    Args:
        model: Neural network model
        loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number

    Returns:
        avg_loss, accuracy
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    with tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False) as pbar:
        for data, target in pbar:
            data, target = data.to(device), target.to(device)

            # Forward pass and loss calculation
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Stats
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)

            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total
            })

    avg_loss = total_loss / total
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def evaluate(model, loader, device, phase="Valid"):
    """Evaluate model on data loader.

    Args:
        model: Neural network model
        loader: Data loader
        device: Device to evaluate on
        phase: Phase name (e.g., 'Valid', 'Test')

    Returns:
        avg_loss, accuracy
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        with tqdm(loader, desc=f"[{phase}]", leave=False) as pbar:
            for data, target in pbar:
                data, target = data.to(device), target.to(device)

                # Forward pass
                output = model(data)
                loss = F.cross_entropy(output, target)

                # Stats
                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += data.size(0)

                # Update progress bar
                pbar.set_postfix({
                    'loss': loss.item(),
                    'acc': 100. * correct / total
                })

    avg_loss = total_loss / total
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def train_model(model, train_loader, valid_loader, optimizer, device,
                epochs=20, log_wandb=True, scheduler=None):
    """Train model with validation and wandb logging.

    Args:
        model: Neural network model
        train_loader: Training data loader
        valid_loader: Validation data loader
        optimizer: Optimizer
        device: Device to train on
        epochs: Number of epochs to train
        log_wandb: Whether to log to wandb
        scheduler: Learning rate scheduler (optional)

    Returns:
        trained model, training history
    """
    history = {
        'train_loss': [], 'train_acc': [],
        'valid_loss': [], 'valid_acc': []
    }

    best_valid_acc = 0
    best_model_state = None

    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # Validate
        valid_loss, valid_acc = evaluate(model, valid_loader, device, phase="Valid")
        history['valid_loss'].append(valid_loss)
        history['valid_acc'].append(valid_acc)

        # Save best model
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_model_state = model.state_dict().copy()

        # Step scheduler if provided
        if scheduler is not None:
            scheduler.step()

        # Log to wandb
        if log_wandb:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'train/accuracy': train_acc,
                'valid/loss': valid_loss,
                'valid/accuracy': valid_acc,
                'lr': optimizer.param_groups[0]['lr']
            })

        # Print epoch summary
        print(f"Epoch {epoch}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
              f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2f}%")
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history 



def train_models(models, train_loader, valid_loader, test_loader, config, device, exp_id):
    
    # Define models, optimizers, and schedulers
    optimizers = {}
    schedulers = {}    

    for model_name in models:

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

    return models, histories