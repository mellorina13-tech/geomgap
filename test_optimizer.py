import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.insert(0, '.')
from optimizer import GeomGAPOptimizer, GeomGAPSGD


def test_geomgap_initialization():
    """Test optimizer initialization with default parameters."""
    model = nn.Linear(10, 2)
    optimizer = GeomGAPOptimizer(model.parameters())
    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert group['a'] == 0.001
    assert group['b'] == 1e-5
    assert group['r'] == 1.01
    assert group['curvature_factor'] == 0.1
    print("PASS: GeomGAPOptimizer initialization passed.")


def test_geomgap_step():
    """Test a single optimization step."""
    model = nn.Linear(5, 3, bias=False)
    optimizer = GeomGAPOptimizer(model.parameters(), a=0.01, b=0.001, r=1.005)
    loss_fn = nn.MSELoss()
    x = torch.randn(2, 5)
    y = torch.randn(2, 3)
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()
    # Check that parameters changed
    with torch.no_grad():
        params_before = [p.clone() for p in model.parameters()]
    optimizer.step()  # second step
    with torch.no_grad():
        for p_before, p_after in zip(params_before, model.parameters()):
            assert not torch.allclose(p_before, p_after), "Parameters should change after step"
    print("PASS: GeomGAPOptimizer step passed.")


def test_geomgap_learning_rate_calculation():
    """Test GAP learning rate formula."""
    model = nn.Linear(3, 1)
    optimizer = GeomGAPOptimizer(model.parameters(), a=0.01, b=0.0, r=1.02)
    # Simulate step count
    for p in model.parameters():
        optimizer.state[p]['step'] = 10
    lr = optimizer.get_learning_rate()
    expected = 0.01 * (1.02 ** 10)
    assert abs(lr - expected) < 1e-9
    print("PASS: Learning rate calculation passed.")


def test_hyperbolic_curvature():
    """Test hyperbolic curvature scaling."""
    optimizer = GeomGAPOptimizer([torch.tensor([1.0])])  # dummy params
    grad = torch.tensor([3.0, 4.0], dtype=torch.float32)  # norm = 5
    scaled = optimizer._hyperbolic_curvature(grad, curvature_factor=0.1, step=0)
    # denominator = sqrt(1 + 0.1 * 25) = sqrt(3.5) ≈ 1.870828693
    expected = grad / torch.sqrt(torch.tensor(1.0 + 0.1 * 25))
    assert torch.allclose(scaled, expected, rtol=1e-6)
    print("PASS: Hyperbolic curvature scaling passed.")


def test_geometric_damping():
    """Test geometric damping."""
    optimizer = GeomGAPOptimizer([torch.tensor([1.0])])
    damping = optimizer._geometric_damping(grad_norm=5.0, threshold=10.0, step=0, r=1.01)
    assert damping == 1.0  # below threshold
    damping = optimizer._geometric_damping(grad_norm=15.0, threshold=10.0, step=100, r=1.01)
    assert 0.0 < damping < 1.0
    print("PASS: Geometric damping passed.")


def test_safe_geometric_clamp():
    """Test safe geometric clamp."""
    optimizer = GeomGAPOptimizer([torch.tensor([1.0])])
    param = torch.tensor([2.0, 3.0])
    update = torch.tensor([100.0, 200.0])  # large update
    safe = optimizer._safe_geometric_clamp(param, update)
    # Should be scaled down because ratio > 0.5
    param_norm = torch.norm(param).item()
    update_norm = torch.norm(update).item()
    safe_norm = torch.norm(safe).item()
    assert safe_norm <= 0.5 * param_norm + 1e-6
    print("PASS: Safe geometric clamp passed.")


def test_geomgap_sgd():
    """Test GeomGAPSGD variant."""
    model = nn.Linear(7, 4)
    optimizer = GeomGAPSGD(model.parameters(), a=0.01, b=0.001, r=1.005, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    x = torch.randn(3, 7)
    y = torch.randint(0, 4, (3,))
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()
    print("PASS: GeomGAPSGD step passed.")


def test_nan_safety():
    """Ensure optimizer handles NaN gradients safely."""
    model = nn.Linear(2, 2)
    optimizer = GeomGAPOptimizer(model.parameters())
    for p in model.parameters():
        # Create gradient with same shape, insert NaN in some positions
        grad = torch.randn_like(p)
        if grad.dim() == 2:
            grad[0, 0] = float('nan')
        else:
            grad[0] = float('nan')
        p.grad = grad
    try:
        optimizer.step()
        # Should not raise error; NaN gradient should be zeroed
        print("PASS: NaN safety passed.")
    except Exception as e:
        print(f"FAIL: NaN safety failed: {e}")
        raise


if __name__ == '__main__':
    test_geomgap_initialization()
    test_geomgap_step()
    test_geomgap_learning_rate_calculation()
    test_hyperbolic_curvature()
    test_geometric_damping()
    test_safe_geometric_clamp()
    test_geomgap_sgd()
    test_nan_safety()
    print("\nAll tests passed!")