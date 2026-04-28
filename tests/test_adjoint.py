"""Adjoint gradient verification using finite differences."""
import math
import mlx.core as mx
from mlx_de import odeint, odeint_adjoint


def finite_diff_grad(loss_fn, y0, eps=1e-4):
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    y0_plus = y0 + eps
    y0_minus = y0 - eps
    return (loss_fn(y0_plus) - loss_fn(y0_minus)) / (2 * eps)


def test_adjoint_vs_finite_diff_exp():
    """Adjoint gradient for exp ODE should match finite differences."""
    def exp_ode(t, y):
        return y

    t = mx.linspace(0.0, 1.0, 20)
    y0 = mx.array([1.0])

    def loss(y0):
        y = odeint_adjoint(exp_ode, y0, t, method="dopri5")
        return mx.sum(y[-1])

    g_adjoint = mx.grad(loss)(y0)
    g_fd = finite_diff_grad(loss, y0, eps=1e-4)
    diff = abs(g_adjoint[0].item() - g_fd.item())
    print(f"  adjoint={g_adjoint[0].item():.8f}, fd={g_fd.item():.8f}, diff={diff:.2e}")
    assert diff < 1e-2, f"adjoint vs fd diff too large: {diff}"


def test_adjoint_vs_finite_diff_harmonic():
    """Adjoint gradient for harmonic oscillator should match finite differences."""
    class HarmonicOscillator:
        def __init__(self, omega=2.0):
            self.omega = omega

        def __call__(self, t, y):
            x, v = y[0], y[1]
            return mx.array([v, -(self.omega ** 2) * x])

    func = HarmonicOscillator(2.0)
    t = mx.linspace(0.0, 1.0, 30)
    y0 = mx.array([1.0, 0.0])

    def loss(y0):
        y = odeint_adjoint(func, y0, t, method="dopri5")
        return mx.sum(y[-1] ** 2)

    g_adjoint = mx.grad(loss)(y0)

    # Finite diff each component separately
    for i in range(2):
        def loss_i(y0):
            y = odeint_adjoint(func, y0, t, method="dopri5")
            return mx.sum(y[-1] ** 2)

        eps = 1e-4
        y0_plus = mx.array(y0)
        y0_plus[i] = y0_plus[i] + eps
        y0_minus = mx.array(y0)
        y0_minus[i] = y0_minus[i] - eps
        g_fd_i = (loss_i(y0_plus) - loss_i(y0_minus)).item() / (2 * eps)
        diff = abs(g_adjoint[i].item() - g_fd_i)
        print(f"  dim {i}: adjoint={g_adjoint[i].item():.6f}, fd={g_fd_i:.6f}, diff={diff:.2e}")
        assert diff < 1e-1, f"dim {i}: adjoint vs fd diff too large: {diff}"


def test_adjoint_vs_direct_consistency():
    """Adjoint and direct gradients should be close for multiple problem types."""
    def exp_ode(t, y):
        return -0.5 * y

    t = mx.linspace(0.0, 2.0, 20)
    y0 = mx.array([1.0])

    def loss_direct(y0):
        y = odeint(exp_ode, y0, t, method="dopri5")
        return mx.sum(y[-1] ** 2)

    def loss_adjoint(y0):
        y = odeint_adjoint(exp_ode, y0, t, method="dopri5")
        return mx.sum(y[-1] ** 2)

    g_direct = mx.grad(loss_direct)(y0)
    g_adjoint = mx.grad(loss_adjoint)(y0)
    diff = abs(g_direct[0].item() - g_adjoint[0].item())
    print(f"  direct={g_direct[0].item():.8f}, adjoint={g_adjoint[0].item():.8f}, diff={diff:.2e}")
    assert diff < 1e-2, f"direct vs adjoint diff too large: {diff}"


if __name__ == "__main__":
    tests = [
        ("Adjoint vs FD (exp)", test_adjoint_vs_finite_diff_exp),
        ("Adjoint vs FD (harmonic)", test_adjoint_vs_finite_diff_harmonic),
        ("Adjoint vs direct consistency", test_adjoint_vs_direct_consistency),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"\n── {name} ──")
        try:
            test_fn()
            print(f"  PASSED")
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
