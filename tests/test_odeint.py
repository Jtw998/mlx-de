"""Tests for odeint forward solves and gradient computation."""
import math
import mlx.core as mx
from mlx_de import odeint, odeint_adjoint, SOLVERS
from problems import ExponentialODE, HarmonicOscillator, LinearODE, SinusoidalODE


def assert_close(actual, expected, tol=1e-4, label=""):
    diff = abs(float(actual) - float(expected))
    assert diff < tol, f"{label}: {actual} vs {expected}, diff={diff} > {tol}"


# ── Forward solve tests ──────────────────────────────────────────────

def test_exponential_all_solvers():
    """All solvers should approximate exp(1) to within their expected accuracy."""
    func = ExponentialODE()
    t = mx.linspace(0.0, 1.0, 25)
    y0 = mx.array([1.0])
    expected = math.e

    adaptive_tols = {"dopri5": 1e-4, "tsit5": 1e-4}
    fixed_tols = {"euler": 2e-1, "midpoint": 5e-3, "rk4": 1e-5,
                  "heun2": 5e-3, "heun3": 5e-4}

    for name in SOLVERS:
        y = odeint(func, y0, t, method=name, rtol=1e-7, atol=1e-9)
        tol = adaptive_tols.get(name, fixed_tols.get(name, 1e-2))
        assert_close(y[-1, 0], expected, tol=tol, label=f"exp_ode/{name}")
        print(f"  {name}: y[-1]={y[-1, 0].item():.8f}, diff={abs(y[-1, 0].item() - expected):.2e}")


def test_harmonic_oscillator():
    """Dopri5 should solve the harmonic oscillator accurately."""
    omega = 2.0
    func = HarmonicOscillator(omega)
    t = mx.linspace(0.0, math.pi / omega, 50)
    y0 = mx.array([1.0, 0.0])

    y = odeint(func, y0, t, method="dopri5")

    # At t = pi/omega: x should be cos(pi)=-1, v should be ~0
    assert_close(y[-1, 0], -1.0, tol=1e-4, label="HO/x")
    assert_close(y[-1, 1], 0.0, tol=1e-3, label="HO/v")
    print(f"  x(T)={y[-1, 0].item():.8f}, v(T)={y[-1, 1].item():.2e}")


def test_linear_ode():
    """Linear decay should match analytical exp(-0.5*t)."""
    func = LinearODE()
    t = mx.linspace(0.0, 2.0, 30)
    y0 = mx.array([1.0])

    y = odeint(func, y0, t, method="dopri5")
    expected = math.exp(-1.0)
    assert_close(y[-1, 0], expected, tol=1e-4, label="linear_ode")
    print(f"  y[-1]={y[-1, 0].item():.8f}, expected={expected:.8f}")


def test_sinusoidal_ode():
    """Non-autonomous ODE: dy/dt=cos(t)."""
    func = SinusoidalODE()
    t = mx.linspace(0.0, math.pi / 2, 30)
    y0 = mx.array([0.0])

    y = odeint(func, y0, t, method="dopri5")
    assert_close(y[-1, 0], 1.0, tol=1e-4, label="sin_ode")
    print(f"  y[-1]={y[-1, 0].item():.8f}")


# ── Gradient tests ───────────────────────────────────────────────────

def test_odeint_gradient_exponential():
    """mx.grad through odeint should give dL/dy0 = exp(1)."""
    func = ExponentialODE()
    t = mx.linspace(0.0, 1.0, 20)
    y0 = mx.array([1.0])

    def loss(y0):
        y = odeint(func, y0, t, method="dopri5")
        return mx.sum(y[-1])

    g = mx.grad(loss)(y0)
    expected = math.e
    assert_close(g[0], expected, tol=1e-3, label="odeint_grad/exp")
    print(f"  grad={g[0].item():.8f}, expected={expected:.8f}, diff={abs(g[0].item()-expected):.2e}")


def test_adjoint_gradient_exponential():
    """odeint_adjoint should match odeint gradient for exponential ODE."""
    func = ExponentialODE()
    t = mx.linspace(0.0, 1.0, 20)
    y0 = mx.array([1.0])

    def loss_direct(y0):
        y = odeint(func, y0, t, method="dopri5")
        return mx.sum(y[-1])

    def loss_adjoint(y0):
        y = odeint_adjoint(func, y0, t, method="dopri5")
        return mx.sum(y[-1])

    g_direct = mx.grad(loss_direct)(y0)
    g_adjoint = mx.grad(loss_adjoint)(y0)
    diff = abs(g_direct[0].item() - g_adjoint[0].item())
    assert diff < 1e-2, f"direct={g_direct[0].item()}, adjoint={g_adjoint[0].item()}, diff={diff}"
    print(f"  direct={g_direct[0].item():.8f}, adjoint={g_adjoint[0].item():.8f}, diff={diff:.2e}")


def test_adjoint_gradient_harmonic():
    """Adjoint gradient for 2D harmonic oscillator should match direct."""
    func = HarmonicOscillator(2.0)
    t = mx.linspace(0.0, 1.0, 30)
    y0 = mx.array([1.0, 0.0])

    def loss_direct(y0):
        y = odeint(func, y0, t, method="dopri5")
        return mx.sum(y[-1] ** 2)

    def loss_adjoint(y0):
        y = odeint_adjoint(func, y0, t, method="dopri5")
        return mx.sum(y[-1] ** 2)

    g_direct = mx.grad(loss_direct)(y0)
    g_adjoint = mx.grad(loss_adjoint)(y0)
    for i in range(2):
        diff = abs(g_direct[i].item() - g_adjoint[i].item())
        assert diff < 1e-1, f"dim {i}: direct={g_direct[i].item()}, adjoint={g_adjoint[i].item()}"
        print(f"  dim {i}: direct={g_direct[i].item():.6f}, adjoint={g_adjoint[i].item():.6f}, diff={diff:.2e}")


# ── Event handling tests ─────────────────────────────────────────────

def test_event_exponential():
    """Event at y=2 for exp ODE: t_event should be ln(2)~0.693."""
    func = ExponentialODE()
    t = mx.array([0.0, 2.0])
    y0 = mx.array([1.0])

    def event_fn(t, y):
        return y - 2.0

    event_t, solution = odeint(func, y0, t, method="dopri5", event_fn=event_fn)
    assert_close(event_t, math.log(2), tol=1e-3, label="event_t")
    assert_close(solution[-1, 0], 2.0, tol=1e-2, label="event_y")
    print(f"  event_t={event_t.item():.8f}, expected={math.log(2):.8f}")
    print(f"  y_at_event={solution[-1, 0].item():.8f}")


# ── Run all ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("Exponential (all solvers)", test_exponential_all_solvers),
        ("Harmonic oscillator", test_harmonic_oscillator),
        ("Linear ODE", test_linear_ode),
        ("Sinusoidal ODE", test_sinusoidal_ode),
        ("odeint gradient", test_odeint_gradient_exponential),
        ("Adjoint gradient (exp)", test_adjoint_gradient_exponential),
        ("Adjoint gradient (HO)", test_adjoint_gradient_harmonic),
        ("Event handling", test_event_exponential),
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
