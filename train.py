"""
Training script for GeomGAP optimizer on MNIST/CIFAR-10.
Example usage:
    python train.py --dataset mnist --epochs 10 --lr 0.001
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from optimizer import GeomGAPOptimizer


class SimpleMLP(nn.Module):
    """10-layer MLP."""
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


def get_dataloaders(dataset='mnist', batch_size=128):
    """Get dataloaders for MNIST or CIFAR-10."""
    if dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        input_dim = 784
        output_dim = 10
    elif dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        input_dim = 3072
        output_dim = 10
    else:
        raise ValueError(f"Dataset {dataset} not supported.")
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader, input_dim, output_dim


def train_epoch(model, loader, optimizer, criterion, device):
    """Train one epoch."""
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


def main():
    parser = argparse.ArgumentParser(description='Train with GeomGAP optimizer')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'], help='Dataset')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (a parameter)')
    parser.add_argument('--r', type=float, default=1.005, help='Geometric factor r')
    parser.add_argument('--b', type=float, default=0.00001, help='Bias b')
    parser.add_argument('--curvature', type=float, default=0.1, help='Curvature factor')
    parser.add_argument('--grad_threshold', type=float, default=10.0, help='Gradient damping threshold')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm for clipping')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {args.dataset} with device {device}")
    print(f"Hyperparameters: lr={args.lr}, r={args.r}, b={args.b}, curvature={args.curvature}")
    
    # Data
    train_loader, test_loader, input_dim, output_dim = get_dataloaders(args.dataset, args.batch_size)
    
    # Model
    model = SimpleMLP(input_dim=input_dim, output_dim=output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = GeomGAPOptimizer(
        model.parameters(),
        a=args.lr,
        b=args.b,
        r=args.r,
        curvature_factor=args.curvature,
        grad_threshold=args.grad_threshold,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay
    )
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Optimizer: {optimizer.__class__.__name__}")
    
    # Training loop
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    
    for epoch in range(args.epochs):
        start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        elapsed = time.time() - start
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | "
              f"Time: {elapsed:.2f}s")
    
    # Final summary
    print(f"\nTraining completed.")
    print(f"Best test accuracy: {max(test_accs):.2f}% at epoch {test_accs.index(max(test_accs)) + 1}")
    
    # Save model checkpoint
    checkpoint_path = f'geomgap_{args.dataset}_model.pth'
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
    }, checkpoint_path)
    print(f"Model saved to {checkpoint_path}")


if __name__ == '__main__':
    main()