import math
import mlx.core as mx


def find_event(interp_fn, sign0, t0, t1, event_fn, tol):
    nitrs = mx.ceil(mx.log((t1 - t0) / tol) / math.log(2.0))

    for _ in range(int(nitrs.item())):
        t_mid = (t1 + t0) / 2.0
        y_mid = interp_fn(t_mid)
        sign_mid = mx.sign(event_fn(t_mid, y_mid))
        same_as_sign0 = (sign0 == sign_mid)
        t0 = mx.where(same_as_sign0, t_mid, t0)
        t1 = mx.where(same_as_sign0, t1, t_mid)
    event_t = (t0 + t1) / 2.0

    return event_t, interp_fn(event_t)


def combine_event_functions(event_fn, t0, y0):
    """Ensure all event functions are initially positive, then combine by min."""
    initial_signs = mx.sign(event_fn(t0, y0))

    def combined_event_fn(t, y):
        c = event_fn(t, y)
        return mx.min(c * initial_signs)

    return combined_event_fn
