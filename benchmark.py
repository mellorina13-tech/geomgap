"""
Benchmark script for GeomGAP vs Adam on MNIST and CIFAR-10.
Trains a simple MLP and logs loss/accuracy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from optimizer import GeomGAPOptimizer


class SimpleMLP(nn.Module):
    """10-layer MLP as described in the roadmap."""
    def __init__(self, input_dim=784, output_dim=10, hidden_dims=[256, 256, 128, 128, 64, 64, 32, 32, 16]):
        super(SimpleMLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


def get_mnist_dataloaders(batch_size=128):
    """MNIST dataloaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def get_cifar10_dataloaders(batch_size=128):
    """CIFAR-10 dataloaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def run_experiment(dataset='mnist', optimizer_name='geomgap', epochs=5, lr=0.001, batch_size=128):
    """Run a single experiment."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running {optimizer_name} on {dataset} with device {device} (batch_size={batch_size})")
    
    # Data
    if dataset == 'mnist':
        train_loader, test_loader = get_mnist_dataloaders(batch_size=batch_size)
        input_dim = 784
        output_dim = 10
    elif dataset == 'cifar10':
        train_loader, test_loader = get_cifar10_dataloaders(batch_size=batch_size)
        input_dim = 3072  # 32x32x3
        output_dim = 10
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    
    # Model
    model = SimpleMLP(input_dim=input_dim, output_dim=output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    if optimizer_name == 'geomgap':
        optimizer = GeomGAPOptimizer(
            model.parameters(),
            a=lr,           # initial coefficient
            b=lr * 0.01,    # bias
            r=1.005,        # geometric factor
            curvature_factor=0.1,
            grad_threshold=10.0,
            max_grad_norm=1.0,
            weight_decay=1e-4
        )
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        raise ValueError(f"Unknown optimizer {optimizer_name}")
    
    # Training logs
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    
    # Training loop
    for epoch in range(epochs):
        start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        elapsed = time.time() - start
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | "
              f"Time: {elapsed:.2f}s")
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'final_train_acc': train_accs[-1],
        'final_test_acc': test_accs[-1]
    }


def plot_comparison(results, dataset, save_path='comparison.png'):
    """Plot loss and accuracy comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Training loss
    ax = axes[0, 0]
    for opt_name, res in results.items():
        ax.plot(res['train_losses'], label=f'{opt_name}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title(f'{dataset} - Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Test loss
    ax = axes[0, 1]
    for opt_name, res in results.items():
        ax.plot(res['test_losses'], label=f'{opt_name}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Loss')
    ax.set_title(f'{dataset} - Test Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Training accuracy
    ax = axes[1, 0]
    for opt_name, res in results.items():
        ax.plot(res['train_accs'], label=f'{opt_name}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Accuracy (%)')
    ax.set_title(f'{dataset} - Training Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Test accuracy
    ax = axes[1, 1]
    for opt_name, res in results.items():
        ax.plot(res['test_accs'], label=f'{opt_name}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title(f'{dataset} - Test Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    # plt.show()
    print(f"Comparison plot saved to {save_path}")


def main():
    """Run benchmark for MNIST and CIFAR-10."""
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark GeomGAP vs Adam')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs per dataset')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--datasets', nargs='+', default=['mnist', 'cifar10'], choices=['mnist', 'cifar10'], help='Datasets to run')
    args = parser.parse_args()

    datasets = args.datasets
    epochs = args.epochs
    learning_rate = args.lr
    batch_size = args.batch_size

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Benchmark on {dataset.upper()}")
        print('='*60)
        
        results = {}
        for opt_name in ['geomgap', 'adam']:
            print(f"\n--- {opt_name.upper()} ---")
            res = run_experiment(dataset=dataset, optimizer_name=opt_name, epochs=epochs, lr=learning_rate, batch_size=batch_size)
            results[opt_name] = res
        
        # Plot comparison
        plot_comparison(results, dataset, save_path=f'{dataset}_comparison.png')
        
        # Print summary
        print(f"\nSummary for {dataset}:")
        for opt_name, res in results.items():
            print(f"  {opt_name.upper()}: Final Test Accuracy = {res['final_test_acc']:.2f}%")
    
    print("\nBenchmark completed.")


if __name__ == '__main__':
    main()