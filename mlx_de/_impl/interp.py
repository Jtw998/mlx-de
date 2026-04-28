import mlx.core as mx


def _interp_fit(y0, y1, y_mid, f0, f1, dt):
    """Fit coefficients for 4th order polynomial interpolation.

    Returns coefficients [a, b, c, d, e] for:
        p(x) = a*x^4 + b*x^3 + c*x^2 + d*x + e,  x in [0, 1]
    """
    a = 2 * dt * (f1 - f0) - 8 * (y1 + y0) + 16 * y_mid
    b = dt * (5 * f0 - 3 * f1) + 18 * y0 + 14 * y1 - 32 * y_mid
    c = dt * (f1 - 4 * f0) - 11 * y0 - 5 * y1 + 16 * y_mid
    d = dt * f0
    e = y0
    return [e, d, c, b, a]


def _interp_evaluate(coefficients, t0, t1, t):
    """Evaluate polynomial interpolation at given time point."""
    x = (t - t0) / (t1 - t0)
    x = x.astype(coefficients[0].dtype)

    total = coefficients[0] + x * coefficients[1]
    x_power = x
    for coefficient in coefficients[2:]:
        x_power = x_power * x
        total = total + x_power * coefficient

    return total
