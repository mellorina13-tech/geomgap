"""
Comprehensive performance test for GeomGAP vs Adam on MNIST with 10 epochs.

Measures:
- Loss & accuracy per epoch
- Training time per epoch
- Final test accuracy
- Best test accuracy
- Convergence speed (epoch to reach 95% of best accuracy)
- Average epoch time
- Memory usage (optional)
- Generates plots and summary report.
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
import psutil
import gc
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from optimizer import GeomGAPOptimizer


class SimpleMLP(nn.Module):
    """10-layer MLP as used in benchmark."""
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


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch, return loss, accuracy, and time."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start = time.time()
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
    
    epoch_time = time.time() - start
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc, epoch_time


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


def get_memory_usage():
    """Return current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB


def run_optimizer_test(optimizer_name='adam', epochs=10, lr=0.001, batch_size=128, seed=42):
    """
    Run a single optimizer test and return detailed metrics.
    Returns dict with:
        train_losses, test_losses, train_accs, test_accs, epoch_times,
        final_test_acc, best_test_acc, best_epoch,
        convergence_epoch, avg_epoch_time, peak_memory_mb
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Data
    train_loader, test_loader = get_mnist_dataloaders(batch_size=batch_size)
    input_dim = 784
    output_dim = 10
    
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
    epoch_times = []
    memory_before = get_memory_usage()
    
    print(f"  Starting {optimizer_name.upper()}...")
    for epoch in range(epochs):
        # Train
        train_loss, train_acc, epoch_time = train_epoch(model, train_loader, optimizer, criterion, device)
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        epoch_times.append(epoch_time)
        
        print(f"    Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | "
              f"Time: {epoch_time:.2f}s")
    
    memory_after = get_memory_usage()
    peak_memory = max(memory_before, memory_after)
    
    # Compute derived metrics
    final_test_acc = test_accs[-1]
    best_test_acc = max(test_accs)
    best_epoch = test_accs.index(best_test_acc) + 1
    avg_epoch_time = np.mean(epoch_times)
    
    # Convergence speed: epoch when accuracy first reaches 95% of best accuracy
    target_acc = 0.95 * best_test_acc
    convergence_epoch = None
    for i, acc in enumerate(test_accs):
        if acc >= target_acc:
            convergence_epoch = i + 1
            break
    
    results = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'epoch_times': epoch_times,
        'final_test_acc': final_test_acc,
        'best_test_acc': best_test_acc,
        'best_epoch': best_epoch,
        'convergence_epoch': convergence_epoch,
        'avg_epoch_time': avg_epoch_time,
        'peak_memory_mb': peak_memory,
        'total_train_time': sum(epoch_times),
    }
    return results


def plot_comparison(results, save_dir='.'):
    """Plot loss and accuracy comparison for both optimizers."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Training loss
    ax = axes[0, 0]
    for opt_name, res in results.items():
        ax.plot(res['train_losses'], label=f'{opt_name}', marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Test loss
    ax = axes[0, 1]
    for opt_name, res in results.items():
        ax.plot(res['test_losses'], label=f'{opt_name}', marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Loss')
    ax.set_title('Test Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Training accuracy
    ax = axes[0, 2]
    for opt_name, res in results.items():
        ax.plot(res['train_accs'], label=f'{opt_name}', marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Accuracy (%)')
    ax.set_title('Training Accuracy Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Test accuracy
    ax = axes[1, 0]
    for opt_name, res in results.items():
        ax.plot(res['test_accs'], label=f'{opt_name}', marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Test Accuracy Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Epoch times
    ax = axes[1, 1]
    for opt_name, res in results.items():
        ax.plot(res['epoch_times'], label=f'{opt_name}', marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Epoch Time (s)')
    ax.set_title('Epoch Training Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bar chart for final metrics
    ax = axes[1, 2]
    metrics = ['Final Test Acc', 'Best Test Acc', 'Avg Epoch Time']
    x = np.arange(len(metrics))
    width = 0.35
    for idx, opt_name in enumerate(results.keys()):
        values = [
            results[opt_name]['final_test_acc'],
            results[opt_name]['best_test_acc'],
            results[opt_name]['avg_epoch_time']
        ]
        offset = -width/2 if idx == 0 else width/2
        ax.bar(x + offset, values, width, label=opt_name)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Value')
    ax.set_title('Key Metrics Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(save_dir, f'mnist_performance_comparison_{timestamp}.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Comparison plot saved to {plot_path}")
    return plot_path


def generate_summary_report(results, save_dir='.'):
    """Generate a text summary report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(save_dir, f'mnist_performance_report_{timestamp}.txt')
    with open(report_path, 'w') as f:
        f.write("MNIST Performance Test Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Test date: {datetime.now().isoformat()}\n")
        f.write(f"Epochs: {len(next(iter(results.values()))['train_losses'])}\n")
        f.write(f"Batch size: 128\n")
        f.write(f"Learning rate: 0.001\n")
        f.write(f"Seed: 42\n")
        f.write("\n")
        
        for opt_name, res in results.items():
            f.write(f"{opt_name.upper()} Optimizer\n")
            f.write("-" * 30 + "\n")
            f.write(f"Final test accuracy: {res['final_test_acc']:.2f}%\n")
            f.write(f"Best test accuracy: {res['best_test_acc']:.2f}% (epoch {res['best_epoch']})\n")
            f.write(f"Convergence epoch (95% of best): {res['convergence_epoch']}\n")
            f.write(f"Average epoch time: {res['avg_epoch_time']:.2f} s\n")
            f.write(f"Total training time: {res['total_train_time']:.2f} s\n")
            f.write(f"Peak memory usage: {res['peak_memory_mb']:.2f} MB\n")
            f.write("\n")
        
        # Comparison
        f.write("Comparison Summary\n")
        f.write("-" * 30 + "\n")
        adam = results.get('adam')
        geomgap = results.get('geomgap')
        if adam and geomgap:
            f.write(f"Accuracy difference (GeomGAP - Adam): {geomgap['final_test_acc'] - adam['final_test_acc']:.2f}%\n")
            f.write(f"Best accuracy difference: {geomgap['best_test_acc'] - adam['best_test_acc']:.2f}%\n")
            f.write(f"Time ratio (GeomGAP/Adam): {geomgap['avg_epoch_time'] / adam['avg_epoch_time']:.2f}\n")
            f.write(f"Memory ratio (GeomGAP/Adam): {geomgap['peak_memory_mb'] / adam['peak_memory_mb']:.2f}\n")
            if geomgap['final_test_acc'] > adam['final_test_acc']:
                f.write("GeomGAP outperforms Adam in final accuracy.\n")
            else:
                f.write("Adam outperforms GeomGAP in final accuracy.\n")
        f.write("\n")
        
        # Per epoch data
        f.write("Per Epoch Data (first 5 epochs)\n")
        f.write("-" * 30 + "\n")
        for opt_name, res in results.items():
            f.write(f"{opt_name.upper()}:\n")
            f.write("Epoch | Train Loss | Test Loss | Train Acc | Test Acc | Time (s)\n")
            for i in range(min(5, len(res['train_losses']))):
                f.write(f"{i+1:4d} | {res['train_losses'][i]:10.4f} | {res['test_losses'][i]:9.4f} | "
                        f"{res['train_accs'][i]:9.2f}% | {res['test_accs'][i]:8.2f}% | {res['epoch_times'][i]:.2f}\n")
            f.write("\n")
    
    print(f"Summary report saved to {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description='Comprehensive performance test for GeomGAP vs Adam on MNIST')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs (default: 5)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save outputs')
    args = parser.parse_args()
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("Comprehensive Performance Test: GeomGAP vs Adam on MNIST")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, Batch size: {args.batch_size}, Seed: {args.seed}")
    print("=" * 70)
    
    results = {}
    for opt_name in ['adam', 'geomgap']:
        print(f"\nRunning {opt_name.upper()} optimizer...")
        res = run_optimizer_test(
            optimizer_name=opt_name,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            seed=args.seed
        )
        results[opt_name] = res
        print(f"  {opt_name.upper()} completed.")
        print(f"  Final test accuracy: {res['final_test_acc']:.2f}%")
        print(f"  Best test accuracy: {res['best_test_acc']:.2f}% at epoch {res['best_epoch']}")
        print(f"  Average epoch time: {res['avg_epoch_time']:.2f} s")
    
    # Plot comparison
    plot_path = plot_comparison(results, save_dir=args.output_dir)
    
    # Generate summary report
    report_path = generate_summary_report(results, save_dir=args.output_dir)
    
    print("\n" + "=" * 70)
    print("Performance test completed successfully!")
    print(f"Plot saved: {plot_path}")
    print(f"Report saved: {report_path}")
    print("\nSummary:")
    for opt_name, res in results.items():
        print(f"  {opt_name.upper()}: Final Test Acc = {res['final_test_acc']:.2f}%, "
              f"Best = {res['best_test_acc']:.2f}%, "
              f"Avg Epoch Time = {res['avg_epoch_time']:.2f}s")
    print("=" * 70)


if __name__ == '__main__':
    main()