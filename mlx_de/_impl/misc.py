import math
import warnings
import mlx.core as mx
import mlx.nn as nn
from .event_handling import combine_event_functions


def _numel(shape):
    """Total number of elements in a shape tuple."""
    return math.prod(shape) if shape else 1


def _promote_dtype(dt1, dt2):
    """Return the higher-precision dtype of two."""
    return dt1 if dt1.size >= dt2.size else dt2

_all_callback_names = ['callback_step', 'callback_accept_step', 'callback_reject_step']
_all_adjoint_callback_names = [name + '_adjoint' for name in _all_callback_names]
_null_callback = lambda *args, **kwargs: None


def _handle_unused_kwargs(solver, unused_kwargs):
    if len(unused_kwargs) > 0:
        warnings.warn(f'{solver.__class__.__name__}: Unexpected arguments {unused_kwargs}')


def _rms_norm(tensor):
    return mx.sqrt(mx.mean(mx.abs(tensor) ** 2))


def _linf_norm(tensor):
    return mx.abs(tensor).max()


def _zero_norm(tensor):
    return mx.array(0.0)


def _mixed_norm(tensor_tuple):
    if len(tensor_tuple) == 0:
        return mx.array(0.0)
    return max(_rms_norm(tensor) for tensor in tensor_tuple)


def _select_initial_step(func, t0, y0, order, rtol, atol, norm, f0=None):
    """Empirically select a good initial step (Hairer I, Sec. II.4)."""
    dtype = y0.dtype
    t_dtype = t0.dtype
    t0 = t0.astype(t_dtype)

    if f0 is None:
        f0 = func(t0, y0)

    scale = atol + mx.abs(y0) * rtol
    d0 = mx.abs(norm(y0 / scale))
    d1 = mx.abs(norm(f0 / scale))

    if d0.item() < 1e-5 or d1.item() < 1e-5:
        h0 = mx.array(1e-6, dtype=dtype)
    else:
        h0 = mx.array(0.01, dtype=dtype) * d0 / d1
    h0 = mx.abs(h0)

    y1 = y0 + h0 * f0
    f1 = func(t0 + h0.astype(t_dtype), y1)
    d2 = mx.abs(norm((f1 - f0) / scale) / h0)

    d1_val = d1.item() if hasattr(d1, 'item') else d1
    d2_val = d2.item() if hasattr(d2, 'item') else d2
    if d1_val <= 1e-15 and d2_val <= 1e-15:
        h1 = mx.maximum(mx.array(1e-6, dtype=dtype), h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1_val, d2_val)) ** (1.0 / float(order + 1))
        h1 = mx.array(h1, dtype=dtype)
    h1 = mx.abs(h1)

    return mx.minimum(100 * h0, h1).astype(t_dtype)


def _compute_error_ratio(error_estimate, rtol, atol, y0, y1, norm):
    error_tol = atol + rtol * mx.maximum(mx.abs(y0), mx.abs(y1))
    return mx.abs(norm(error_estimate / error_tol))


def _optimal_step_size(last_step, error_ratio, safety, ifactor, dfactor, order):
    """Calculate the optimal size for the next step."""
    error_ratio_val = error_ratio.item() if hasattr(error_ratio, 'item') else error_ratio
    if error_ratio_val == 0:
        return last_step * ifactor
    dfactor_actual = dfactor
    if error_ratio_val < 1:
        dfactor_actual = mx.ones((), dtype=last_step.dtype)
    error_ratio = error_ratio.astype(last_step.dtype)
    exponent = mx.array(1.0 / order, dtype=last_step.dtype)
    factor = mx.minimum(ifactor, mx.maximum(safety / error_ratio ** exponent, dfactor_actual))
    return last_step * factor


def _decreasing(t):
    return mx.all(t[1:] < t[:-1]).item()


def _assert_one_dimensional(name, t):
    assert t.ndim == 1, f"{name} must be one dimensional"


def _assert_increasing(name, t):
    assert mx.all(t[1:] > t[:-1]).item(), f'{name} must be strictly increasing or decreasing'


def _assert_floating(name, t):
    if not mx.issubdtype(t.dtype, mx.floating):
        raise TypeError(f'`{name}` must be a floating point Tensor but is a {t.dtype}')


def _tuple_tol(name, tol, shapes):
    try:
        iter(tol)
    except TypeError:
        return tol
    tol = tuple(tol)
    assert len(tol) == len(shapes), f"If using tupled {name} it must have the same length as the tuple y0"
    tol = [mx.broadcast_to(mx.array(tol_), shape) for tol_, shape in zip(tol, shapes)]
    return mx.concatenate([t.reshape(-1) for t in tol])


def _flat_to_shape(tensor, length, shapes):
    tensor_list = []
    total = 0
    for shape in shapes:
        next_total = total + _numel(shape)
        tensor_list.append(tensor[..., total:next_total].reshape((*length, *shape)))
        total = next_total
    return tuple(tensor_list)


class _TupleFunc(nn.Module):
    def __init__(self, base_func, shapes):
        super().__init__()
        self.base_func = base_func
        self.shapes = shapes

    def __call__(self, t, y):
        f = self.base_func(t, _flat_to_shape(y, (), self.shapes))
        return mx.concatenate([f_.reshape(-1) for f_ in f])


class _TupleInputOnlyFunc(nn.Module):
    def __init__(self, base_func, shapes):
        super().__init__()
        self.base_func = base_func
        self.shapes = shapes

    def __call__(self, t, y):
        return self.base_func(t, _flat_to_shape(y, (), self.shapes))


class _ReverseFunc(nn.Module):
    def __init__(self, base_func, mul=1.0):
        super().__init__()
        self.base_func = base_func
        self.mul = mul

    def __call__(self, t, y):
        return self.mul * self.base_func(-t, y)


class _PerturbFunc(nn.Module):
    """v1: no-op wrapper. Perturb support (nextafter) deferred to v2."""

    def __init__(self, base_func):
        super().__init__()
        self.base_func = base_func

    def __call__(self, t, y):
        return self.base_func(t, y)


def _check_inputs(func, y0, t, rtol, atol, method, options, event_fn, SOLVERS):
    if event_fn is not None:
        if len(t) != 2:
            raise ValueError(f"We require len(t) == 2 when in event handling mode, but got len(t)={len(t)}.")
        event_fn = combine_event_functions(event_fn, t[0], y0)

    original_func = func

    # Normalise to tensor (non-tupled) input
    shapes = None
    is_tuple = not isinstance(y0, mx.array)
    if is_tuple:
        assert isinstance(y0, tuple), 'y0 must be either a mx.array or a tuple'
        shapes = [y0_.shape for y0_ in y0]
        rtol = _tuple_tol('rtol', rtol, shapes)
        atol = _tuple_tol('atol', atol, shapes)
        y0 = mx.concatenate([y0_.reshape(-1) for y0_ in y0])
        func = _TupleFunc(func, shapes)
        if event_fn is not None:
            event_fn = _TupleInputOnlyFunc(event_fn, shapes)

    # Normalise method and options
    if options is None:
        options = {}
    else:
        options = options.copy()
    if method is None:
        method = 'dopri5'
    if method not in SOLVERS:
        raise ValueError(f'Invalid method "{method}". Must be one of {set(SOLVERS.keys())}.')

    if is_tuple:
        if 'norm' in options:
            norm = options['norm']
        else:
            norm = _mixed_norm

        def _norm(tensor):
            y = _flat_to_shape(tensor, (), shapes)
            return norm(y)
        options['norm'] = _norm
    else:
        if 'norm' in options:
            pass
        else:
            options['norm'] = _rms_norm

    # Normalise time
    _check_timelike('t', t, True)
    t_is_reversed = False
    if len(t) > 1 and t[0].item() > t[1].item():
        t_is_reversed = True

    if t_is_reversed:
        t = -t
        func = _ReverseFunc(func, mul=-1.0)
        if event_fn is not None:
            event_fn = _ReverseFunc(event_fn)

        try:
            _grid_constructor = options['grid_constructor']
        except KeyError:
            pass
        else:
            options['grid_constructor'] = lambda func, y0, t: -_grid_constructor(func, y0, -t)

        _flip_option(options, 'step_t')
        _flip_option(options, 'jump_t')

    _assert_increasing('t', t)

    # Backward compatibility: Allow t and y0 to be on different devices (MLX: handled automatically)

    # Add perturb argument wrapper to func (v1: no-op)
    func = _PerturbFunc(func)

    # Add callbacks to func
    callback_names = set()
    for callback_name in _all_callback_names:
        try:
            callback = getattr(original_func, callback_name)
        except AttributeError:
            setattr(func, callback_name, _null_callback)
        else:
            if callback is not _null_callback:
                callback_names.add(callback_name)
                if is_tuple:
                    def make_cb(cb=callback):
                        def wrapped(t0, y0, dt):
                            y0 = _flat_to_shape(y0, (), shapes)
                            return cb(t0, y0, dt)
                        return wrapped
                    callback = make_cb()
                if t_is_reversed:
                    def make_cb(cb=callback):
                        def wrapped(t0, y0, dt):
                            return cb(-t0, y0, dt)
                        return wrapped
                    callback = make_cb()
            setattr(func, callback_name, callback)

    for callback_name in _all_adjoint_callback_names:
        try:
            callback = getattr(original_func, callback_name)
        except AttributeError:
            pass
        else:
            setattr(func, callback_name, callback)

    invalid_callbacks = callback_names - SOLVERS[method].valid_callbacks()
    if len(invalid_callbacks) > 0:
        warnings.warn(f"Solver '{method}' does not support callbacks {invalid_callbacks}")

    return shapes, func, y0, t, rtol, atol, method, options, event_fn, t_is_reversed


def _check_timelike(name, timelike, can_grad):
    assert isinstance(timelike, mx.array), f'{name} must be a mx.array'
    _assert_floating(name, timelike)
    assert timelike.ndim == 1, f"{name} must be one dimensional"
    diff = timelike[1:] > timelike[:-1]
    assert mx.all(diff).item() or mx.all(~diff).item(), f'{name} must be strictly increasing or decreasing'


def _flip_option(options, option_name):
    try:
        option_value = options[option_name]
    except KeyError:
        pass
    else:
        if isinstance(option_value, mx.array):
            options[option_name] = -option_value
