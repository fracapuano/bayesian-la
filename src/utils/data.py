import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_mnist_dataloaders(batch_size=64, valid_split=0.1, num_workers=2, pin_memory=True):
    """Create train, validation, and test dataloaders for MNIST.
    
    Args:
        batch_size: Batch size for training and evaluation
        valid_split: Proportion of training data to use for validation
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        train_loader, valid_loader, test_loader
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Download and create datasets
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    
    # Split training data for validation
    valid_size = int(len(train_dataset) * valid_split)
    train_size = len(train_dataset) - valid_size
    train_dataset, valid_dataset = random_split(
        train_dataset, [train_size, valid_size], 
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    return train_loader, valid_loader, test_loader 