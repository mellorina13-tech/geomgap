# GeomGAP vs Adam Performance Comparison on MNIST

## Test Overview

- **Dataset**: MNIST (handwritten digits)
- **Model**: 10‑layer MLP (256‑256‑128‑128‑64‑64‑32‑32‑16) with ReLU activations and dropout (0.1)
- **Batch size**: 128
- **Learning rate**: 0.001
- **Weight decay**: 1e‑4
- **GeomGAP hyperparameters**: a=0.001, b=1e‑5, r=1.005, curvature_factor=0.1, grad_threshold=10.0, max_grad_norm=1.0
- **Adam hyperparameters**: default PyTorch settings (beta1=0.9, beta2=0.999)

## Existing 1‑Epoch Benchmark Results

The previous benchmark (run for a single epoch) produced the following metrics:

| Optimizer | Train Loss | Train Accuracy | Test Loss | Test Accuracy |
|-----------|------------|----------------|-----------|---------------|
| GeomGAP   | 1.2096     | 58.12%         | 0.7714    | **74.84%**    |
| Adam      | 1.3030     | 49.47%         | 0.4714    | **84.13%**    |

**Observation**: After just one epoch, Adam achieves a higher test accuracy (84.13% vs 74.84%), while GeomGAP shows a lower training loss and higher training accuracy, indicating different optimization dynamics.

## Prepared 10‑Epoch Comprehensive Performance Test

A new, more comprehensive test script has been created to evaluate both optimizers over **10 epochs** with detailed per‑epoch tracking.

### Script Location
- `geomgap/performance_test.py` – full‑featured test with plotting and report generation
- `geomgap/simple_benchmark.py` – lightweight version without plotting

### Metrics Collected
- Training & test loss per epoch
- Training & test accuracy per epoch
- Epoch‑wise training time
- Peak memory usage (optional)
- Final test accuracy
- Best test accuracy and the epoch it occurred
- Convergence speed (epoch to reach 95% of best accuracy)
- Average epoch time & total training time

### How to Run
```bash
cd geomgap
python performance_test.py --epochs 10 --output_dir ./results
```

The script will:
1. Train GeomGAP for 10 epochs, printing progress after each epoch.
2. Train Adam for 10 epochs under identical conditions (same random seed).
3. Generate a multi‑panel comparison plot (`mnist_performance_comparison_<timestamp>.png`).
4. Write a detailed summary report (`mnist_performance_report_<timestamp>.txt`).

## Expected Insights from a 10‑Epoch Run

- **Convergence behavior**: Whether GeomGAP catches up or surpasses Adam after more epochs.
- **Stability**: GeomGAP’s geometric damping should reduce loss oscillations, especially at later stages.
- **Time efficiency**: Comparison of per‑epoch training times (GeomGAP’s extra computations may cause a slight overhead).
- **Memory footprint**: Both optimizers should have similar memory usage.

## Conclusions (Based on 1‑Epoch Data)

1. **Adam** converges faster in the very first epoch, reaching a test accuracy of **84.13%**.
2. **GeomGAP** shows a higher training accuracy (58.12% vs 49.47%) and lower training loss, suggesting it may be fitting the training data more aggressively.
3. The lower test accuracy of GeomGAP after one epoch could be due to overfitting or slower adaptation to the test distribution.
4. GeomGAP’s geometric damping and hyperbolic curvature are designed to prevent gradient explosion in deeper networks and longer training runs; its advantages may become more apparent over multiple epochs or on more complex datasets.

## Recommendations

- Run the 10‑epoch test to obtain a more complete picture of long‑term performance.
- Experiment with different hyperparameters (especially `r` and `curvature_factor`) to tune GeomGAP for MNIST.
- Test on larger datasets (CIFAR‑10, ImageNet) where gradient explosion is more likely and GeomGAP’s protective mechanisms could be more beneficial.

## Files Created

- `performance_test.py` – comprehensive test script
- `simple_benchmark.py` – lightweight benchmark
- `debug_train.py` – diagnostic script (verifies basic functionality)
- `check_data.py` – data loading sanity check
- `mnist_comparison.png` – previous 1‑epoch comparison plot (from earlier benchmark)
- `mnist_10epoch_results.json` – will be generated after running the 10‑epoch test

## Next Steps

1. Execute the 10‑epoch test in a stable environment (the current environment exhibits hanging issues with `torchvision` imports).
2. Analyze the generated plots and reports.
3. Fine‑tune GeomGAP hyperparameters based on the results.
4. Extend the comparison to other optimizers (RMSProp, SGD with momentum) and datasets.

---
*Report generated on 2026‑02‑28*