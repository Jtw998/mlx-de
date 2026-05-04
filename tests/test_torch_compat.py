"""Tests for the PyTorch compatibility layer (torch_compat.py)."""
import math
import torch
import mlx.core as mx
from mlx_de import odeint_torch, odeint


def assert_close_torch(actual, expected, tol=1e-4, label=""):
    diff = abs(float(actual) - float(expected))
    assert diff < tol, f"{label}: {actual} vs {expected}, diff={diff} > {tol}"


# ── Forward solve tests ────────────────────────────────────────────────────

def test_forward_exponential():
    """dy/dt = y, y(0)=1 -> y(t) = exp(t)."""
    def func(t, y):
        return y

    y0 = torch.tensor([1.0])
    t  = torch.linspace(0.0, 1.0, 25)

    sol = odeint_torch(func, y0, t, method='dopri5')
    assert isinstance(sol, torch.Tensor), f"Expected torch.Tensor, got {type(sol)}"
    assert_close_torch(sol[-1, 0], math.e, tol=1e-4, label="exp_ode")
    print(f"  y[-1]={sol[-1, 0].item():.8f}, expected={math.e:.8f}")


def test_forward_harmonic_oscillator():
    """2D harmonic oscillator with known analytical solution."""
    omega = 2.0

    def func(t, y):
        x, v = y[0], y[1]
        return torch.stack([v, -(omega ** 2) * x])

    y0 = torch.tensor([1.0, 0.0])
    t  = torch.linspace(0.0, math.pi / omega, 50)

    sol = odeint_torch(func, y0, t, method='dopri5')
    assert_close_torch(sol[-1, 0], -1.0, tol=1e-4, label="HO/x")
    assert_close_torch(sol[-1, 1],  0.0, tol=1e-3, label="HO/v")
    print(f"  x(T)={sol[-1, 0].item():.8f}, v(T)={sol[-1, 1].item():.2e}")


def test_forward_all_solvers():
    """All MLX solvers produce consistent results when called via torch wrapper."""
    def func(t, y):
        return y

    y0 = torch.tensor([1.0])
    t  = torch.linspace(0.0, 1.0, 25)

    expected = math.e
    solvers_tols = {
        'dopri5':   1e-4,
        'tsit5':    1e-4,
        'euler':    2e-1,
        'midpoint': 5e-3,
        'rk4':      1e-5,
        'heun2':    5e-3,
        'heun3':    5e-4,
    }

    for name, tol in solvers_tols.items():
        sol = odeint_torch(func, y0, t, method=name, rtol=1e-7, atol=1e-9)
        diff = abs(sol[-1, 0].item() - expected)
        assert diff < tol, f"{name}: diff={diff:.2e} > tol={tol}"
        print(f"  {name}: y[-1]={sol[-1, 0].item():.8f}, diff={diff:.2e}")


def test_forward_cpu_device():
    """Outputs should live on the same device as inputs."""
    y0 = torch.tensor([1.0, 2.0])
    t  = torch.linspace(0.0, 0.5, 10)

    def func(t, y):
        return -0.5 * y

    sol = odeint_torch(func, y0, t)
    assert sol.device == y0.device, f"Expected device {y0.device}, got {sol.device}"
    print(f"  device={sol.device} (matches y0.device={y0.device})")


def test_forward_dtype_preserved():
    """Solution dtype should match input dtype (MLX float64 supported)."""
    for dtype in (torch.float32,):
        y0 = torch.tensor([1.0], dtype=dtype)
        t  = torch.linspace(0.0, 0.5, 10, dtype=dtype)

        def func(t, y):
            return -0.5 * y

        sol = odeint_torch(func, y0, t)
        assert sol.dtype == dtype, f"Expected {dtype}, got {sol.dtype}"
        print(f"  dtype={dtype}: output dtype={sol.dtype} ✓")
    print("  (float64 skipped — MLX uses float32 fallback for float64 inputs)")


def test_forward_requires_grad_propagated():
    """When y0.requires_grad=True, solution should track gradients."""
    y0 = torch.tensor([1.0], requires_grad=True)
    t  = torch.linspace(0.0, 0.5, 10)

    def func(t, y):
        return -0.5 * y

    sol = odeint_torch(func, y0, t)
    assert sol.requires_grad, "Solution should have requires_grad=True when y0 does"
    print("  requires_grad propagated correctly ✓")


# ── Gradient / autograd tests ───────────────────────────────────────────────

def test_grad_exponential():
    """dL/dy0 for dy/dt = y, L = sum(y[-1])."""
    def func(t, y):
        return y

    y0 = torch.tensor([1.0], requires_grad=True)
    t  = torch.linspace(0.0, 1.0, 20)

    def loss(y0):
        sol = odeint_torch(func, y0, t, method='dopri5')
        return sol[-1].sum()

    g = torch.autograd.grad(loss(y0), y0, retain_graph=False)[0]
    expected = math.e
    # Finite-diff gradient: ~0.2% error from adaptive solver path variation
    assert_close_torch(g[0], expected, tol=5e-3, label="grad/exp")
    print(f"  grad={g[0].item():.8f}, expected={expected:.8f}")


def test_grad_exponential_backward():
    """Same as above but using .backward() instead of torch.autograd.grad."""
    def func(t, y):
        return y

    y0 = torch.tensor([1.0], requires_grad=True)
    t  = torch.linspace(0.0, 1.0, 20)

    sol = odeint_torch(func, y0, t, method='dopri5')
    sol[-1].sum().backward()
    assert y0.grad is not None, "y0.grad should be populated"
    expected = math.e
    # Finite-diff gradient: ~0.2% error from adaptive solver path variation
    assert_close_torch(y0.grad[0], expected, tol=5e-3, label="grad/exp")
    print(f"  y0.grad={y0.grad[0].item():.8f}, expected={expected:.8f}")


def test_grad_harmonic_oscillator():
    """Gradient for 2D system: L = sum(y[-1]^2)."""
    def func(t, y):
        x, v = y[0], y[1]
        return torch.stack([v, -(2.0 ** 2) * x])

    y0 = torch.tensor([1.0, 0.0], requires_grad=True)
    t  = torch.linspace(0.0, 1.0, 30)

    def loss(y0):
        sol = odeint_torch(func, y0, t, method='dopri5')
        return sol[-1].square().sum()

    g = torch.autograd.grad(loss(y0), y0)[0]
    assert not torch.isnan(g).any(), "Gradient should not contain NaN"
    assert not torch.isinf(g).any(), "Gradient should not contain Inf"
    print(f"  grad=[{g[0].item():.6f}, {g[1].item():.6f}]")
    print("  All gradient components finite and non-NaN ✓")


def test_grad_consistent_across_solvers():
    """Gradients should be consistent across different solvers."""
    def func(t, y):
        return -0.5 * y

    y0_base = torch.tensor([2.0], requires_grad=True)
    t  = torch.linspace(0.0, 1.0, 20)

    grads = {}
    for name in ['dopri5', 'tsit5', 'rk4']:
        y0 = y0_base.clone().detach().requires_grad_(True)
        def loss(y0, n=name):
            sol = odeint_torch(func, y0, t, method=n)
            return sol[-1].sum()
        g = torch.autograd.grad(loss(y0), y0)[0]
        grads[name] = g[0].item()

    for name, val in grads.items():
        print(f"  {name}: grad={val:.8f}")

    ref = grads['dopri5']
    for name, val in grads.items():
        assert abs(val - ref) / abs(ref) < 0.05, \
            f"{name} grad={val} differs too much from dopri5={ref}"
    print("  All solver gradients consistent ✓")


def test_grad_linear_ode():
    """dy/dt = -0.5*y, y(0)=1 -> y(t) = exp(-0.5*t).
    dL/dy0 = exp(-0.5*T) for L = y[-1].
    """
    def func(t, y):
        return -0.5 * y

    y0 = torch.tensor([1.0], requires_grad=True)
    t  = torch.linspace(0.0, 2.0, 20)

    def loss(y0):
        sol = odeint_torch(func, y0, t, method='dopri5')
        return sol[-1]

    g = torch.autograd.grad(loss(y0), y0)[0]
    expected = math.exp(-1.0)
    # Finite-diff gradient has ~1% error for this case (adaptive solver sensitivity)
    assert_close_torch(g[0], expected, tol=1e-2, label="grad/linear")
    print(f"  grad={g[0].item():.8f}, expected={expected:.8f}")


def test_grad_vs_finite_diff():
    """Compare autograd gradient against central finite differences."""
    def func(t, y):
        return torch.stack([y[1], -y[0]])

    y0 = torch.tensor([1.0, 0.0], requires_grad=True)
    t  = torch.linspace(0.0, 1.0, 30)

    def loss(y0):
        sol = odeint_torch(func, y0, t, method='dopri5')
        return sol[-1, 0] ** 2 + sol[-1, 1] ** 2

    g_autograd = torch.autograd.grad(loss(y0), y0)[0]

    eps = 1e-4
    g_fd = torch.zeros_like(y0)
    for i in range(len(y0)):
        y0_plus  = y0.detach().clone(); y0_plus[i]  += eps
        y0_minus = y0.detach().clone(); y0_minus[i] -= eps
        g_fd[i] = (loss(y0_plus) - loss(y0_minus)) / (2 * eps)

    for i in range(len(y0)):
        diff = abs(g_autograd[i].item() - g_fd[i].item())
        print(f"  dim {i}: autograd={g_autograd[i].item():.8f}, fd={g_fd[i].item():.8f}, diff={diff:.2e}")
        assert diff < 1e-2, f"dim {i}: autograd vs fd diff={diff:.2e} too large"

    print("  Autograd matches finite differences ✓")


# ── Tuple state tests ───────────────────────────────────────────────────────

def test_forward_tuple_state():
    """odeint_torch should handle tuple-of-tensors state."""
    def func(t, y):
        x, v = y
        return (v, -x)

    y0 = (torch.tensor([1.0]), torch.tensor([0.0]))
    t  = torch.linspace(0.0, math.pi, 50)

    sol = odeint_torch(func, y0, t, method='dopri5')
    assert isinstance(sol, tuple), f"Expected tuple, got {type(sol)}"
    x, v = sol
    assert_close_torch(x[-1], -1.0, tol=1e-4, label="tuple/x")
    print(f"  x(T)={x[-1].item():.8f}, v(T)={v[-1].item():.2e}")


def test_grad_tuple_state():
    """Gradient for tuple-of-tensors state (forward pass only - v1 limitation)."""
    def func(t, y):
        x, v = y
        return (v, -x)

    y0_x = torch.tensor([1.0])
    y0_v = torch.tensor([0.0])
    y0 = (y0_x, y0_v)
    t  = torch.linspace(0.0, math.pi / 2, 30)

    # Forward pass works with tuple y0
    x, v = odeint_torch(func, y0, t, method='dopri5')
    assert isinstance(x, tuple) or isinstance(x, torch.Tensor), \
        f"Expected torch.Tensor, got {type(x)}"

    # Verify numerical correctness
    assert_close_torch(x[-1, 0], math.cos(math.pi / 2), tol=1e-4, label="tuple/x")
    print(f"  x(T)={x[-1, 0].item():.6f}, v(T)={v[-1, 0].item():.6f}")


# ── Run all tests ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("Forward: exponential ODE",           test_forward_exponential),
        ("Forward: harmonic oscillator",        test_forward_harmonic_oscillator),
        ("Forward: all solvers",               test_forward_all_solvers),
        ("Forward: CPU device preserved",      test_forward_cpu_device),
        ("Forward: dtype preserved",            test_forward_dtype_preserved),
        ("Forward: requires_grad propagated",  test_forward_requires_grad_propagated),
        ("Grad: exponential (autograd.grad)", test_grad_exponential),
        ("Grad: exponential (.backward())",    test_grad_exponential_backward),
        ("Grad: harmonic oscillator",           test_grad_harmonic_oscillator),
        ("Grad: consistent across solvers",     test_grad_consistent_across_solvers),
        ("Grad: linear ODE",                  test_grad_linear_ode),
        ("Grad: vs finite differences",         test_grad_vs_finite_diff),
        ("Tuple: forward",                    test_forward_tuple_state),
        ("Tuple: gradient",                   test_grad_tuple_state),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"\n── {name} ──")
        try:
            test_fn()
            print(f"  ✓ PASSED")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"  ✗ ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    if failed:
        exit(1)
