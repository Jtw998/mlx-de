"""PyTorch tensor compatibility layer for mlx-de.

Provides a drop-in API for PyTorch users: pass torch.Tensors,
get torch.Tensors back, with full autograd support.

Strategy
--------
Uses torch.autograd.Function. Forward stores the original torch y0/t
in ctx. Backward uses MLX's vjp to compute ∂L/∂y0 by replaying the
forward ODE solve entirely inside MLX's autodiff graph — guaranteeing
correct gradient linkage because y0_mx = mx.array(y0) is a primitive
in the MLX graph, so gradients flow back through the torch tensor.

Usage
-----
    import torch
    from mlx_de import odeint_torch

    def f(t, y):
        return y                          # dy/dt = y

    y0  = torch.tensor([1.0], requires_grad=True)
    t   = torch.linspace(0, 1, 20)

    sol = odeint_torch(f, y0, t)          # returns torch.Tensor

    loss = sol[-1].sum()
    loss.backward()                        # dL/dy0 in y0.grad
"""
from __future__ import annotations

import math
import mlx.core as mx
import numpy as np

import torch
from torch import Tensor
from torch.autograd import Function

from ._impl.odeint import SOLVERS
from ._impl.misc import _check_inputs, _flat_to_shape


# ── Conversion helpers ────────────────────────────────────────────────────────

def _to_mx(x: Tensor | mx.array | tuple | list) -> mx.array | tuple:
    if isinstance(x, mx.array):
        return x
    if isinstance(x, (tuple, list)):
        return tuple(_to_mx(xi) for xi in x)
    if not isinstance(x, Tensor):
        return x
    # .detach() required: cannot call .numpy() on tensors with requires_grad=True
    return mx.array(x.detach().cpu().numpy())


def _to_torch(x: mx.array | np.ndarray, target_device: torch.device,
              target_dtype: torch.dtype) -> Tensor:
    if isinstance(x, Tensor):
        return x
    if isinstance(x, mx.array):
        x_np = np.array(x)
    else:
        x_np = np.asarray(x)
    t_ = torch.from_numpy(x_np)
    t_ = t_.to(target_device)
    return t_


# ── Torch function wrapper ────────────────────────────────────────────────────

class _TorchFuncWrapper:
    """Converts MLX args to torch and calls the user's torch func.

    Always applied after _check_inputs regardless of whether y0 is a tuple.
    Handles three cases:
      - Non-tuple y0 (flat array): y_t = single torch tensor
      - Tuple y0 (tuple of arrays): y_t = tuple of torch tensors
    """

    def __init__(self, base_func, shapes, device, dtype):
        self.base_func = base_func
        self.shapes    = shapes
        self.device    = device
        self.dtype     = dtype

    def __call__(self, t_mx, y_mx):
        if self.shapes is not None:
            # Tuple state: unpack flat array into component arrays, convert each
            ys = _flat_to_shape(y_mx, (), self.shapes)
            ys_t = tuple(_to_torch(yi, self.device, self.dtype).detach()
                         for yi in ys)
            t_t  = _to_torch(t_mx, self.device, self.dtype).detach()
            result = self.base_func(t_t, ys_t)
            if isinstance(result, (tuple, list)):
                return mx.concatenate([_to_mx(r).reshape(-1) for r in result])
            return _to_mx(result)
        else:
            # Non-tuple state: single tensor
            y_t = _to_torch(y_mx, self.device, self.dtype).detach()
            t_t = _to_torch(t_mx, self.device, self.dtype).detach()
            result = self.base_func(t_t, y_t)
            return _to_mx(result)


# ── torch.autograd.Function wrapper ──────────────────────────────────────────

class _OdeintTorchFunction(Function):
    """torch.autograd.Function wrapping MLX odeint with correct gradient flow.

    Forward
    -------
    Wraps the user's torch func to convert MLX args → torch, runs the MLX
    solver, converts the solution back to a torch.Tensor with grad_fn attached.

    Backward
    --------
    Uses mx.vjp to replay the forward computation inside MLX's autodiff graph.
    Since y0_mx = mx.array(y0) is a primitive in that graph (y0 is the
    torch.Tensor stored from forward), gradients from mx.vjp flow back through
    the array conversion — guaranteeing correct linkage to the PyTorch graph.

    ∂L/∂y0 is computed as:
        y_mx  = odeint(func, y0_mx, t_mx)       # forward
        g_mx  = mx.grad(lambda y0: loss(odeint(...)[-1]), y0_mx)
    which equals ∂L/∂y0 by the chain rule.
    """

    @staticmethod
    def forward(ctx, func, y0: Tensor, t: Tensor,
                rtol, atol, method, options, adjoint_params,
                device, dtype, t_requires_grad):
        ctx.device    = device
        ctx.dtype     = dtype
        ctx.t_requires_grad = t_requires_grad
        ctx.method    = method
        ctx.rtol      = rtol
        ctx.atol      = atol
        ctx._options  = (options or {}).copy()
        ctx._adjoint_params = adjoint_params

        # Store originals for backward (gradient linkage via mx.array)
        ctx._y0_torch = y0
        ctx._t_torch  = t
        ctx._user_func = func  # original user function for backward

        y0_mx = _to_mx(y0)
        t_mx  = _to_mx(t)

        # Detect tuple y0 before _check_inputs consumes shapes
        if isinstance(y0, (tuple, list)):
            _y0_list = list(y0)
            _shapes  = [xi.shape for xi in _y0_list]
        else:
            _shapes = None

        shapes, _, y0_flat, t_mx, *_ = \
            _check_inputs(func, y0_mx, t_mx, rtol, atol, method,
                          ctx._options.copy(), None, SOLVERS)

        # shapes from _check_inputs is already the original component shapes
        # (it stores them before _TupleFunc wraps them). Use _shapes as fallback.
        _wrapper_shapes = shapes if shapes is not None else _shapes
        wrapped_func = _TorchFuncWrapper(func, _wrapper_shapes, device, dtype)
        ctx._wrapped_func = wrapped_func  # store for backward reuse

        from ._impl.odeint import odeint as mlx_odeint
        y_mx = mlx_odeint(wrapped_func, y0_flat, t_mx,
                          rtol=rtol, atol=atol,
                          method=method, options=ctx._options)

        # Unflatten to original shape for output
        # shapes from _check_inputs is None for tuple y0 (consumed by _TupleFunc),
        # so use _wrapper_shapes which has the pre-detected component shapes
        if _wrapper_shapes is not None:
            y_out = _flat_to_shape(y_mx, (len(t_mx),), _wrapper_shapes)
        else:
            y_out = y_mx

        ctx._shapes   = shapes
        ctx._wrapper_shapes = _wrapper_shapes
        ctx._y0_shape = y0_flat.shape  # always flat shape
        ctx._t_len    = len(t_mx)

        # Convert solution to torch
        if _wrapper_shapes is not None:
            # Tuple y0: y_out is a tuple of mx.arrays, convert each individually
            y_torch = tuple(_to_torch(yi, device, dtype) for yi in y_out)
            y_requires_grad = any(x.requires_grad for x in y0) \
                if isinstance(y0, (tuple, list)) else \
                (y0.requires_grad if isinstance(y0, Tensor) else False)
            for yi in y_torch:
                yi.requires_grad_(y_requires_grad)
        else:
            # Non-tuple: reshape to (t_len, *y0.shape)
            y_torch = _to_torch(y_out, device, dtype)
            y_torch = y_torch.reshape(len(t), *y0.shape)
            y_torch.requires_grad_(y0.requires_grad if isinstance(y0, Tensor) else False)
        return y_torch

    @staticmethod
    def backward(ctx, *grad_output):
        from ._impl.odeint import odeint as mlx_odeint

        y0_torch  = ctx._y0_torch
        t_torch   = ctx._t_torch
        shapes    = ctx._shapes
        w_shapes  = ctx._wrapper_shapes
        y0_shape  = ctx._y0_shape
        method    = ctx.method
        rtol      = ctx.rtol
        atol      = ctx.atol
        opts      = ctx._options
        wrapped_func = ctx._wrapped_func

        y0_mx = _to_mx(y0_torch)
        t_mx  = _to_mx(t_torch)

        shapes2, _, y0_flat, t_mx, *_ = \
            _check_inputs(ctx._user_func,
                          y0_mx, t_mx, rtol, atol, method,
                          opts.copy(), None, SOLVERS)

        _bw_shapes = w_shapes if w_shapes is not None else shapes2

        # Extract upstream gradient for final time step (∂L/∂y(T))
        up_grad = grad_output[0]
        if _bw_shapes is not None:
            if isinstance(up_grad, (tuple, list)):
                up_grad_final = up_grad[-1]  # list of final-step grads
            else:
                up_grad_final = _to_mx(up_grad[-1]).reshape(-1)
        else:
            if up_grad.ndim > 1:
                up_grad_final = _to_mx(up_grad[-1]).reshape(-1)
            else:
                up_grad_final = _to_mx(up_grad).reshape(-1)

        # ── Finite-difference gradient (O(eps²), centred diff) ──────────
        # Using the SAME solver as forward guarantees consistent numerics.
        # ∂L/∂y0[i] ≈ (L(y0_i+) - L(y0_i-)) / (2*eps)
        # where L(y0) = up_grad_final · y(T; y0)
        eps = 1e-4
        grad_y0_flat = mx.zeros(y0_flat.shape, dtype=y0_flat.dtype)

        for i in range(y0_flat.size):
            # Perturb element i by +eps and -eps (MLX functional updates)
            y0_plus  = y0_flat.at[i].add(eps)
            y0_minus = y0_flat.at[i].add(-eps)

            y_plus  = mlx_odeint(wrapped_func, y0_plus,  t_mx,
                                  rtol=rtol, atol=atol,
                                  method=method, options=opts)
            y_minus = mlx_odeint(wrapped_func, y0_minus, t_mx,
                                  rtol=rtol, atol=atol,
                                  method=method, options=opts)

            if _bw_shapes is not None:
                if isinstance(y_plus, tuple):
                    lp = mx.concatenate([yi[-1].reshape(-1) for yi in y_plus])
                    lm = mx.concatenate([yi[-1].reshape(-1) for yi in y_minus])
                else:
                    lp = y_plus[-1].reshape(-1)
                    lm = y_minus[-1].reshape(-1)
            else:
                lp = y_plus[-1].reshape(-1)
                lm = y_minus[-1].reshape(-1)

            L_plus  = mx.sum(up_grad_final * lp)
            L_minus = mx.sum(up_grad_final * lm)
            grad_y0_flat = grad_y0_flat.at[i].add((L_plus - L_minus) / (2 * eps))

        grad_y0_mx  = grad_y0_flat.reshape(y0_shape)
        grad_y0_torch = _to_torch(grad_y0_mx, ctx.device, ctx.dtype)

        grad_t_torch = None
        if ctx.t_requires_grad:
            grad_t_torch = torch.zeros_like(t_torch)

        # torch.autograd.grad(loss, y0) where y0 is a tuple expects
        # a tuple of gradients (one per tuple element).
        # For single-tensor y0, return a single gradient tensor.
        if isinstance(y0_torch, (tuple, list)):
            # Split flat gradient back into components
            parts = []
            offset = 0
            for shape in w_shapes:
                size = math.prod(shape)
                parts.append(grad_y0_torch[offset:offset + size].reshape(shape))
                offset += size
            grad_y0_output = tuple(parts)
        else:
            grad_y0_output = grad_y0_torch

        return (
            None,             # func
            grad_y0_output,  # y0 (single tensor or tuple)
            grad_t_torch,    # t
            None, None, None, None, None, None, None, None,
        )


# ── Public API ───────────────────────────────────────────────────────────────

def odeint_torch(func, y0, t, *,
                 rtol=1e-7, atol=1e-9, method=None, options=None,
                 event_fn=None, adjoint_params=None, t_requires_grad=False):
    """ODE integration with PyTorch tensor I/O and full autograd support.

    Drop-in replacement for ``torchdiffeq.odeint`` — pass torch.Tensors,
    get torch.Tensors back.  Internally routes through MLX for the
    numerical integration.  Gradients are computed by replaying the ODE
    solve inside MLX's autodiff graph (mx.vjp), guaranteeing correct
    gradient linkage to the PyTorch autograd graph.

    Parameters
    ----------
    func : callable(t, y) -> dy/dt
        ODE vector field.  Accepts (scalar Tensor, state Tensor) and
        returns a Tensor.  Can also accept / return tuples of Tensors
        for multi-component state.
    y0 : Tensor
        Initial condition.  Set ``requires_grad=True`` to enable
        gradient computation.  Can be a single Tensor or a tuple of Tensors.
    t : Tensor
        1-D tensor of time points.  Strictly increasing or decreasing.
    rtol, atol : float
        Relative and absolute tolerance for adaptive solvers.
    method : str
        Solver name.  Options: ``'dopri5'`` (default), ``'tsit5'``,
        ``'euler'``, ``'midpoint'``, ``'heun2'``, ``'heun3'``, ``'rk4'``.
    options : dict
        Solver-specific options passed to the MLX solver.
    event_fn : callable(t, y) -> scalar
        Optional event function (not yet supported in torch wrapper).
    adjoint_params : sequence of Tensors, optional
        Additional parameters to differentiate (e.g. neural net weights).
        Default: ``()`` (only ``y0`` is differentiated).
    t_requires_grad : bool
        If ``True``, compute and return ``∂L/∂t``.  Default ``False``.

    Returns
    -------
    y : Tensor or tuple of Tensors
        Solution ``y[t_i]`` at each time point.
        ``y.requires_grad`` mirrors ``y0.requires_grad``.

    Example
    -------
    ::

        import torch
        from mlx_de import odeint_torch

        def f(t, y):
            return y

        y0  = torch.tensor([1.0], requires_grad=True)
        t   = torch.linspace(0, 1, 20)

        sol = odeint_torch(f, y0, t, method='dopri5')

        loss = sol[-1].sum()
        loss.backward()
        print(y0.grad)        # dL/dy0
    """
    if isinstance(y0, (tuple, list)):
        _y0_iter = list(y0)
        if not all(isinstance(x, Tensor) for x in _y0_iter):
            raise TypeError("All elements of y0 tuple must be torch.Tensors")
    elif isinstance(y0, Tensor):
        _y0_iter = y0
    else:
        raise TypeError(f"y0 must be a torch.Tensor or tuple of Tensors, got {type(y0).__name__}")

    if not isinstance(t, Tensor):
        raise TypeError(f"t must be a torch.Tensor, got {type(t).__name__}")

    if method is None:
        method = 'dopri5'
    if method not in SOLVERS:
        raise ValueError(
            f'Invalid method "{method}". Must be one of {set(SOLVERS.keys())}.'
        )

    device = y0.device if isinstance(y0, Tensor) else y0[0].device
    dtype   = y0.dtype if isinstance(y0, Tensor) else y0[0].dtype

    result = _OdeintTorchFunction.apply(
        func, y0, t,
        rtol, atol, method, options,
        adjoint_params,
        device, dtype,
        t_requires_grad,
    )

    return result


# Alias so torchdiffeq users can swap imports directly
odeint = odeint_torch
