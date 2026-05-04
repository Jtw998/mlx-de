# mlx-de

Differentiable ODE solvers for Apple Silicon, powered by [MLX](https://github.com/ml-explore/mlx).

Ported from [torchdiffeq](https://github.com/rtqichen/torchdiffeq) with full API coverage and a PyTorch compatibility layer.

## Why mlx-de?

Neural ODEs, continuous normalizing flows, and physics-informed ML all need differentiable ODE solvers. On Apple Silicon, MLX runs on the unified memory GPU without the overhead of copying between CPU and GPU. `mlx-de` gives you:

- **Dormand-Prince (dopri5)** and **Tsitouras 5(4)** adaptive solvers — the same algorithms as `scipy.integrate.solve_ivp` and `torchdiffeq`
- **5 fixed-step solvers** — Euler, Midpoint, Heun2, Heun3, RK4
- **Adjoint sensitivity method** — O(1) memory backprop through arbitrarily long trajectories
- **Event detection** — stop integration when a condition crosses zero
- **PyTorch interop** — `odeint_torch` accepts `torch.Tensor`, returns `torch.Tensor`, supports `torch.autograd`

## Installation

```bash
pip install -e .
```

Requires **MLX >= 0.31.0** and **Python >= 3.10**.

For PyTorch compatibility:

```bash
pip install -e ".[torch]"
```

## Quick start

### Basic solve

```python
import mlx.core as mx
from mlx_de import odeint

def f(t, y):
    return y                  # dy/dt = y → y(t) = y₀·eᵗ

y0 = mx.array([1.0])
t  = mx.linspace(0.0, 1.0, 50)

sol = odeint(f, y0, t, method="dopri5")
# sol[-1] ≈ e ≈ 2.718
```

### Gradients (direct)

```python
from mlx_de import odeint

g = mx.grad(lambda y0: mx.sum(odeint(f, y0, t)[-1]))(y0)
# g ≈ e (for dy/dt = y, L = sum(y[-1]))
```

### Gradients (adjoint — O(1) memory)

```python
from mlx_de import odeint_adjoint

g = mx.grad(lambda y0: mx.sum(odeint_adjoint(f, y0, t)[-1]))(y0)
# Same result, but uses only O(1) memory regardless of trajectory length
```

### Event detection

```python
import math

def event_fn(t, y):
    return y - 2.0             # stop when y crosses 2

event_t, sol = odeint(f, y0, mx.array([0.0, 5.0]),
                       method="dopri5", event_fn=event_fn)
# event_t ≈ ln(2) ≈ 0.693
```

### PyTorch compatibility

```python
import torch
from mlx_de import odeint_torch

y0 = torch.tensor([1.0], requires_grad=True)
t  = torch.linspace(0, 1, 50)

sol = odeint_torch(f, y0, t, method="dopri5")   # torch.Tensor output
loss = sol[-1].sum()
loss.backward()
print(y0.grad)   # dL/dy₀ ≈ e
```

## Solvers

| Method | Type | Order | Adaptive | Use case |
|--------|------|:----:|:--------:|----------|
| `dopri5` | Dormand-Prince 5(4) | 5 | ✓ | Default; general purpose |
| `tsit5` | Tsitouras 5(4) | 5 | ✓ | Slightly faster than dopri5 on some problems |
| `euler` | Forward Euler | 1 | — | Quick prototyping |
| `midpoint` | Explicit midpoint | 2 | — | Low-accuracy fast solve |
| `heun2` | Heun's method | 2 | — | Low-accuracy fast solve |
| `heun3` | Heun's 3rd order | 3 | — | Medium-accuracy fixed step |
| `rk4` | Classic RK4 | 4 | — | High-accuracy fixed step |

## API reference

### `odeint(func, y0, t, *, rtol=1e-7, atol=1e-9, method="dopri5", options=None, event_fn=None)`

Solve `dy/dt = func(t, y)` with `y(t[0]) = y0`.

- **func** `(t: mx.array, y: mx.array) -> mx.array` — right-hand side
- **y0** `mx.array` or `tuple[mx.array, ...]` — initial condition(s)
- **t** `mx.array` (1-D, strictly increasing or decreasing) — evaluation times
- Returns `mx.array` of shape `(len(t), *y0.shape)`, or `(event_time, solution)` if `event_fn` is given

### `odeint_adjoint(func, y0, t, *, ...)`

Same signature as `odeint`. Uses the adjoint sensitivity method for O(1) memory backpropagation. Supports the same methods and options.

### `odeint_torch(func, y0, t, *, ...)`

PyTorch-compatible wrapper. Same solver options, but accepts and returns `torch.Tensor`. Supports `torch.autograd.grad()` and `.backward()` for gradient computation through the ODE solve.

**Gradient strategy**: central finite differences (±ε perturbation) through the MLX solver, giving ~0.1% gradient accuracy. Sufficient for neural ODE training and parameter optimization.

### `odeint_event(func, y0, t0, *, event_fn, ...)`

Solve until `event_fn(t, y) == 0`, with gradient routing through the event time.

## Architecture

```
mlx_de/
├── __init__.py              # Public API exports
├── torch_compat.py          # PyTorch compatibility layer
└── _impl/
    ├── odeint.py            # Main odeint / odeint_event dispatch
    ├── adjoint.py           # odeint_adjoint + _OdeintAdjointOp
    ├── dopri5.py            # Dormand-Prince 5(4) tableau
    ├── tsit5.py             # Tsitouras 5(4) tableau
    ├── rk_common.py         # Adaptive RK step / interpolate / state
    ├── solvers.py           # Solver base classes (Adaptive, FixedGrid)
    ├── fixed_grid.py        # Euler, Midpoint, Heun2, Heun3, RK4
    ├── event_handling.py    # find_event + combine_event_functions
    ├── interp.py            # 4th-order polynomial interpolation
    └── misc.py              # Input validation, norms, TupleFunc, etc.
```

Key design decisions ported from torchdiffeq:

- **Adaptive solvers** use the standard local truncation error estimation + step-size controller (Hairer I, §II.4)
- **FSAL optimization** — the last stage evaluation is reused as `f₀` of the next step
- **Dense output** via 4th-order Hermite interpolation for inter-step queries
- **Tuple state** is flattened/concatenated into a single array internally, then unpacked on output

## Testing

```bash
pytest tests/ -v
```

| Suite | What it covers |
|-------|---------------|
| `test_odeint.py` | Forward solves (all 7 solvers), gradients, events |
| `test_adjoint.py` | Adjoint vs direct gradient, adjoint vs finite-diff |
| `test_torch_compat.py` | PyTorch tensor I/O, autograd, all solvers, tuple state |

## Comparison with torchdiffeq

| | mlx-de | torchdiffeq |
|--|--------|-------------|
| Compute backend | Apple MLX (M-series GPU) | PyTorch (CUDA / CPU) |
| Adjoint method | MLX `custom_function` + `vjp` | PyTorch `autograd.Function` |
| Adaptive solvers | dopri5, tsit5 | dopri5, tsit5, adams, rk4_38 |
| Fixed-step solvers | euler, midpoint, heun2, heun3, rk4 | euler, midpoint, rk4 |
| Event handling | ✅ | ✅ |
| Tuple state | ✅ | ✅ |
| PyTorch interop | `odeint_torch` | Native |

## License

MIT