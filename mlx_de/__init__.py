from ._impl.odeint import odeint as _mlx_odeint, odeint_event, SOLVERS
from ._impl.adjoint import odeint_adjoint

# PyTorch compatibility layer
from .torch_compat import odeint_torch

# Keep the original MLX odeint accessible, and allow torch_compat to be opt-in
odeint = _mlx_odeint

__all__ = [
    "odeint",
    "odeint_torch",
    "odeint_event",
    "odeint_adjoint",
    "SOLVERS",
]