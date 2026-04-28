import warnings
import mlx.core as mx
import mlx.nn as nn
from .odeint import SOLVERS, odeint
from .misc import _check_inputs, _flat_to_shape, _mixed_norm, _all_callback_names, _all_adjoint_callback_names


def _flat_cat(arrays):
    """Concatenate flat (1D) arrays."""
    non_empty = [a.reshape(-1) for a in arrays if a.size > 0]
    if not non_empty:
        return mx.zeros(0)
    return mx.concatenate(non_empty)


class _OdeintAdjointOp:
    """Custom autograd op for adjoint-based backprop through ODE solve.

    func is captured by closure. custom_function only tracks array primals;
    non-array args (method, options) are passed but not differentiated.
    """

    def __init__(self, func, y_shape, event_fn, adjoint_rtol, adjoint_atol,
                 adjoint_method, adjoint_options, t_requires_grad):
        self.func = func
        self.y_shape = y_shape
        self.event_fn = event_fn
        self.adjoint_rtol = adjoint_rtol
        self.adjoint_atol = adjoint_atol
        self.adjoint_method = adjoint_method
        self.adjoint_options = adjoint_options
        self.t_requires_grad = t_requires_grad

        self._op = mx.custom_function(self._forward)
        self._op = self._op.vjp(self._vjp)

    def _forward(self, y0, t, rtol, atol, method, options, *adjoint_params):
        ans = odeint(self.func, y0, t, rtol=rtol, atol=atol,
                     method=method, options=options, event_fn=self.event_fn)
        return ans if self.event_fn is None else (ans[0], ans[1])

    def _vjp(self, primals, outputs, output_grads):
        # custom_function.vjp convention:
        #   outputs     = upstream gradient from loss (or tuple for multi-output)
        #   output_grads = forward output value (or tuple for multi-output)
        # For single-return _forward: both are single arrays.
        # For tuple-return (event mode): both are tuples of arrays.
        #
        # primals includes all positional args: (y0, t, rtol, atol, method, options, *adjoint_params)
        y0, t, rtol, atol, method, options, *adjoint_params = primals
        adjoint_params = tuple(p for p in adjoint_params if p.size > 0)

        event_mode = self.event_fn is not None
        if event_mode:
            # outputs = (grad_event_t, grad_solution)
            # output_grads = (event_t, solution)
            up_grad_t, grad_y = outputs
            event_t, y = output_grads
            _t_original = t
            t = mx.concatenate([t[0:1], event_t.reshape(-1)])
        else:
            grad_y = outputs      # upstream gradient from loss, shape (len(t), *y_shape)
            y = output_grads       # forward solution, shape (len(t), *y_shape)

        func = self.func
        t_requires_grad = self.t_requires_grad
        y_flat = y.reshape(len(y), -1)
        y_dim = y_flat.shape[-1]

        # Flat augmented state: [vjp_t(1), y(y_dim), vjp_y(y_dim), *params]
        aug0 = _flat_cat([
            mx.zeros(1, dtype=y.dtype),
            y_flat[-1],
            grad_y[-1].reshape(-1),
        ] + [mx.zeros(p.size, dtype=y.dtype) for p in adjoint_params])

        def aug_dynamics(t_aug, aug_flat):
            pos = 0
            _vjp_t = aug_flat[pos:pos + 1]; pos += 1
            y_state = aug_flat[pos:pos + y_dim].reshape(self.y_shape); pos += y_dim
            adj_y = aug_flat[pos:pos + y_dim].reshape(self.y_shape); pos += y_dim

            def fn_w(t_in, y_in):
                return func(t_in, y_in)

            out_list, grad_list = mx.vjp(fn_w, (t_aug, y_state), (-adj_y,))
            func_eval = out_list[0]
            vjp_t_grad = grad_list[0]
            vjp_y_grad = grad_list[1]

            if not t_requires_grad:
                vjp_t_grad = mx.zeros_like(vjp_t_grad)

            return _flat_cat([
                vjp_t_grad,
                func_eval,
                vjp_y_grad,
            ] + [mx.zeros(p.size, dtype=y.dtype) for p in adjoint_params])

        # Attach adjoint callbacks
        for cb_name, adj_cb_name in zip(_all_callback_names, _all_adjoint_callback_names):
            try:
                callback = getattr(func, adj_cb_name)
            except AttributeError:
                pass
            else:
                setattr(aug_dynamics, cb_name, callback)

        # Solve adjoint ODE backwards
        time_vjps_list = [] if t_requires_grad else None

        for i in range(len(t) - 1, 0, -1):
            if t_requires_grad:
                fe = func(t[i], y[i])
                dLdt = mx.sum(fe.reshape(-1) * grad_y[i].reshape(-1))
                time_vjps_list.append(dLdt)

            t_rev = t[i - 1:i + 1][::-1]
            aug_res = odeint(aug_dynamics, aug0, t_rev,
                             rtol=self.adjoint_rtol, atol=self.adjoint_atol,
                             method=self.adjoint_method, options=self.adjoint_options)
            aug0 = aug_res[1]

            # Reset y and accumulate gradient
            aug0 = _flat_cat([
                aug0[0:1],
                y_flat[i - 1],
                aug0[1 + y_dim:1 + 2 * y_dim] + grad_y[i - 1].reshape(-1),
            ])

        adj_y_grad = aug0[1 + y_dim:1 + 2 * y_dim].reshape(self.y_shape)

        if t_requires_grad:
            time_vjps = mx.stack(time_vjps_list[::-1])
            if event_mode:
                time_vjps = mx.concatenate([time_vjps[0:1], mx.zeros_like(_t_original[1:])])
        else:
            time_vjps = None

        # Return gradients for (y0, t, rtol, atol, method, options, *adjoint_params)
        adj_params_grad = tuple(mx.zeros_like(p) for p in adjoint_params)
        return (adj_y_grad, time_vjps, None, None, None, None, *adj_params_grad)

    def __call__(self, y0, t, rtol, atol, method, options, *adjoint_params):
        return self._op(y0, t, rtol, atol, method, options, *adjoint_params)


def odeint_adjoint(func, y0, t, *, rtol=1e-7, atol=1e-9, method=None, options=None, event_fn=None,
                   adjoint_rtol=None, adjoint_atol=None, adjoint_method=None, adjoint_options=None,
                   adjoint_params=None):
    """ODE solve with adjoint-based backpropagation (memory efficient).

    Solves the ODE forward and computes gradients by solving an augmented
    adjoint ODE backwards in time. O(1) memory w.r.t. integration steps.

    v1 limitation: func must be a pure function (t, y) -> f.
    Gradients through nn.Module parameters not yet supported.

    Args:
        func: Function (t, y) -> f.
        y0: Initial state (mx.array or tuple).
        t: 1-D time array.
        rtol, atol: Forward tolerances.
        method: Forward solver (default 'dopri5').
        options: Forward solver options.
        event_fn: Optional event function.
        adjoint_rtol, adjoint_atol: Adjoint tolerances.
        adjoint_method: Adjoint solver (default: same as method).
        adjoint_options: Adjoint solver options.
        adjoint_params: Parameters for gradients. Default () for pure functions.

    Returns:
        Solution with autograd support via mx.grad.
    """
    if adjoint_params is None and not isinstance(func, nn.Module):
        adjoint_params = ()
    elif adjoint_params is None and isinstance(func, nn.Module):
        adjoint_params = tuple(func.trainable_parameters().values())

    if adjoint_rtol is None:
        adjoint_rtol = rtol
    if adjoint_atol is None:
        adjoint_atol = atol
    if adjoint_method is None:
        adjoint_method = method

    if adjoint_method != method and options is not None and adjoint_options is None:
        raise ValueError("adjoint_method != method requires adjoint_options.")

    if adjoint_options is None:
        adjoint_options = {k: v for k, v in (options or {}).items() if k != "norm"}
    else:
        adjoint_options = adjoint_options.copy()

    adjoint_params = tuple(p for p in adjoint_params if p.size > 0)

    shapes, wrapped_func, y0, t, rtol, atol, method, options, event_fn, decreasing_time = \
        _check_inputs(func, y0, t, rtol, atol, method, options, event_fn, SOLVERS)

    handle_adjoint_norm_(adjoint_options, shapes, options.get("norm"))

    t_requires_grad = True

    op = _OdeintAdjointOp(
        wrapped_func, y0.shape, event_fn,
        adjoint_rtol, adjoint_atol, adjoint_method, adjoint_options,
        t_requires_grad
    )

    ans = op(y0, t, rtol, atol, method, options, *adjoint_params)

    if event_fn is None:
        solution = ans
    else:
        event_t, solution = ans
        if decreasing_time:
            event_t = -event_t

    if shapes is not None:
        solution = _flat_to_shape(solution, (len(t),), shapes)

    if event_fn is None:
        return solution
    else:
        return event_t, solution


def handle_adjoint_norm_(adjoint_options, shapes, state_norm):
    def default_adjoint_norm(tensor_tuple):
        t, y, adj_y, *adj_params = tensor_tuple
        return max(mx.abs(t), state_norm(y), state_norm(adj_y), _mixed_norm(adj_params))

    if "norm" not in adjoint_options:
        adjoint_options["norm"] = default_adjoint_norm
    else:
        adjoint_norm = adjoint_options.get('norm')
        if adjoint_norm == 'seminorm':
            def adjoint_seminorm(tensor_tuple):
                t, y, adj_y, *adj_params = tensor_tuple
                return max(mx.abs(t), state_norm(y), state_norm(adj_y))
            adjoint_options['norm'] = adjoint_seminorm
        elif adjoint_norm is None:
            adjoint_options['norm'] = default_adjoint_norm


def find_parameters(module):
    assert isinstance(module, nn.Module)
    return list(module.trainable_parameters().values())
