import mlx.core as mx
from .dopri5 import Dopri5Solver
from .tsit5 import Tsit5Solver
from .fixed_grid import Euler, Midpoint, Heun2, Heun3, RK4
from .misc import _check_inputs, _flat_to_shape

SOLVERS = {
    'dopri5': Dopri5Solver,
    'tsit5': Tsit5Solver,
    'euler': Euler,
    'midpoint': Midpoint,
    'heun2': Heun2,
    'heun3': Heun3,
    'rk4': RK4,
}


def odeint(func, y0, t, *, rtol=1e-7, atol=1e-9, method=None, options=None, event_fn=None):
    """Integrate a system of ordinary differential equations.

    Solves the initial value problem:
        dy/dt = func(t, y), y(t[0]) = y0

    Args:
        func: Function mapping scalar time `t` and state `y` to derivatives f(t, y).
             State can be a single mx.array or a tuple of mx.array.
        y0: Initial state (mx.array or tuple of mx.array).
        t: 1-D mx.array of time points, strictly increasing or decreasing.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        method: Solver name. Default 'dopri5'.
        options: Solver-specific options dict.
        event_fn: Optional event function for early termination.

    Returns:
        y: Solution at each time point (first dim = len(t)).
        If event_fn is not None: (event_time, solution).
    """
    shapes, func, y0, t, rtol, atol, method, options, event_fn, t_is_reversed = \
        _check_inputs(func, y0, t, rtol, atol, method, options, event_fn, SOLVERS)

    solver = SOLVERS[method](func=func, y0=y0, rtol=rtol, atol=atol, **options)

    if event_fn is None:
        solution = solver.integrate(t)
    else:
        event_t, solution = solver.integrate_until_event(t[0], event_fn)
        event_t = event_t.astype(t.dtype)
        if t_is_reversed:
            event_t = -event_t

    if shapes is not None:
        solution = _flat_to_shape(solution, (len(t),), shapes)

    if event_fn is None:
        return solution
    else:
        return event_t, solution


def odeint_event(func, y0, t0, *, event_fn, reverse_time=False, odeint_interface=odeint, **kwargs):
    """Solve ODE until event, with gradient routing through event time."""

    if reverse_time:
        t = mx.concatenate([t0.reshape(-1), t0.reshape(-1) - 1.0])
    else:
        t = mx.concatenate([t0.reshape(-1), t0.reshape(-1) + 1.0])

    event_t, solution = odeint_interface(func, y0, t, event_fn=event_fn, **kwargs)

    shapes, _func, _, t, _, _, _, _, event_fn, _ = _check_inputs(
        func, y0, t, 0.0, 0.0, None, None, event_fn, SOLVERS)

    if shapes is not None:
        state_t = mx.concatenate([s[-1].reshape(-1) for s in solution])
    else:
        state_t = solution[-1]

    if reverse_time:
        event_t = -event_t

    event_t, state_t = ImplicitFnGradientRerouting.apply(_func, event_fn, event_t, state_t)

    if reverse_time:
        event_t = -event_t

    if shapes is not None:
        state_t = _flat_to_shape(state_t, (), shapes)
        solution = tuple(
            mx.concatenate([s[:-1], s_t[None]], axis=0)
            for s, s_t in zip(solution, state_t)
        )
    else:
        solution = mx.concatenate([solution[:-1], state_t[None]], axis=0)

    return event_t, solution


def _implicit_fn_forward(func, event_fn, event_t, state_t):
    """Forward: return event_t and state_t as-is (identity)."""
    return event_t, state_t


def _implicit_fn_vjp(primals, outputs, output_grads):
    """Backward: compute gradients through the implicit event function."""
    # primals = (func, event_fn, event_t, state_t)
    func, event_fn, event_t, state_t = primals
    grad_t, grad_state = output_grads

    f_val = func(event_t, state_t)

    # VJP of event_fn w.r.t. (event_t, state_t)
    # mx.vjp: returns (primals_out, vjp_fn)
    c, vjp_fn = mx.vjp(lambda a, b: event_fn(a, b), (event_t, state_t), (mx.ones_like(event_fn(event_t, state_t)),))
    par_dt, dstate = vjp_fn((mx.ones_like(c),))

    # Total derivative: dc/dt = partial dc/dt + dc/dstate * f
    dcdt = par_dt + mx.sum(dstate * f_val)

    # Gradient through final state to event time
    grad_t = grad_t + mx.sum(grad_state * f_val)
    dstate = dstate * (-grad_t / (dcdt + 1e-12))
    grad_state = grad_state + dstate

    return None, None, None, grad_state


# Build custom_function for implicit event gradient routing
_ImplicitFn = mx.custom_function(_implicit_fn_forward)
_ImplicitFn = _ImplicitFn.vjp(_implicit_fn_vjp)


class ImplicitFnGradientRerouting:
    """Custom autograd for routing gradients through event time."""
    apply = _ImplicitFn
