import bisect
import collections
import mlx.core as mx
from .event_handling import find_event
from .interp import _interp_evaluate, _interp_fit
from .misc import (
    _compute_error_ratio,
    _select_initial_step,
    _optimal_step_size,
    _handle_unused_kwargs,
    _promote_dtype,
)
from .solvers import AdaptiveStepsizeEventODESolver

_ButcherTableau = collections.namedtuple('_ButcherTableau', 'alpha, beta, c_sol, c_error')
_RungeKuttaState = collections.namedtuple('_RungeKuttaState', 'y1, f1, t0, t1, dt, interp_coeff')


def _runge_kutta_step(func, y0, f0, t0, dt, t1, tableau):
    """Take an arbitrary Runge-Kutta step and estimate error.

    Uses list accumulation + mx.stack instead of in-place writes
    since MLX arrays are immutable.
    """
    t_dtype = y0.dtype
    t0 = t0.astype(t_dtype)
    dt = dt.astype(t_dtype)
    t1 = t1.astype(t_dtype)

    k_list = [f0]
    yi = y0  # will be overwritten

    for i, (alpha_i, beta_i) in enumerate(zip(tableau.alpha, tableau.beta)):
        if alpha_i == 1.:
            ti = t1
        else:
            ti = t0 + alpha_i * dt

        k_so_far = mx.stack(k_list, axis=-1)
        yi = y0 + mx.sum(k_so_far * (beta_i * dt), axis=-1)

        f = func(ti, yi)
        k_list.append(f)

    k = mx.stack(k_list, axis=-1)

    # FSAL (First Same As Last) optimization
    if not (tableau.c_sol[-1] == 0 and mx.all(tableau.c_sol[:-1] == tableau.beta[-1]).item()):
        yi = y0 + mx.sum(k * (dt * tableau.c_sol), axis=-1)

    y1 = yi
    f1 = k_list[-1]
    y1_error = mx.sum(k * (dt * tableau.c_error), axis=-1)

    return y1, f1, y1_error, k


# Precompute constants
_one_third = 1 / 3
_two_thirds = 2 / 3
_one_sixth = 1 / 6


def rk4_step_func(func, t0, dt, t1, y0, f0=None):
    k1 = f0
    if k1 is None:
        k1 = func(t0, y0)
    half_dt = dt * 0.5
    k2 = func(t0 + half_dt, y0 + half_dt * k1)
    k3 = func(t0 + half_dt, y0 + half_dt * k2)
    k4 = func(t1, y0 + dt * k3)
    return (k1 + 2 * (k2 + k3) + k4) * dt * _one_sixth


def rk4_alt_step_func(func, t0, dt, t1, y0, f0=None):
    """Smaller error with slightly more compute."""
    k1 = f0
    if k1 is None:
        k1 = func(t0, y0)
    k2 = func(t0 + dt * _one_third, y0 + dt * k1 * _one_third)
    k3 = func(t0 + dt * _two_thirds, y0 + dt * (k2 - k1 * _one_third))
    k4 = func(t1, y0 + dt * (k1 - k2 + k3))
    return (k1 + 3 * (k2 + k3) + k4) * dt * 0.125


def rk3_step_func(func, t0, dt, t1, y0, butcher_tableu=None, f0=None):
    """Generic 3-stage RK step from Butcher tableau."""
    k1 = f0
    if k1 is None:
        k1 = func(t0, y0)
    k2 = func(t0 + dt * butcher_tableu[1][0], y0 + dt * k1 * butcher_tableu[1][1])
    k3 = func(t0 + dt * butcher_tableu[2][0], y0 + dt * (k1 * butcher_tableu[2][1] + k2 * butcher_tableu[2][2]))
    return dt * (k1 * butcher_tableu[3][1] + k2 * butcher_tableu[3][2] + k3 * butcher_tableu[3][3])


def rk2_step_func(func, t0, dt, t1, y0, butcher_tableu=None, f0=None):
    """Generic 2-stage RK step from Butcher tableau."""
    k1 = f0
    if k1 is None:
        k1 = func(t0, y0)
    k2 = func(t0 + dt * butcher_tableu[1][0], y0 + dt * k1 * butcher_tableu[1][1])
    return dt * (k1 * butcher_tableu[2][1] + k2 * butcher_tableu[2][2])


class RKAdaptiveStepsizeODESolver(AdaptiveStepsizeEventODESolver):
    order: int
    tableau: _ButcherTableau
    mid: mx.array

    def __init__(self, func, y0, rtol, atol,
                 min_step=0,
                 max_step=float('inf'),
                 first_step=None,
                 step_t=None,
                 jump_t=None,
                 safety=0.9,
                 ifactor=10.0,
                 dfactor=0.2,
                 max_num_steps=2 ** 31 - 1,
                 dtype=mx.float32,
                 **kwargs):
        super().__init__(dtype=dtype, y0=y0, **kwargs)

        dtype = _promote_dtype(dtype, y0.dtype)

        self.func = func
        self.rtol = mx.array(rtol, dtype=dtype)
        self.atol = mx.array(atol, dtype=dtype)
        self.min_step = mx.array(min_step, dtype=dtype)
        self.max_step = mx.array(max_step, dtype=dtype)
        self.first_step = None if first_step is None else mx.array(first_step, dtype=dtype)
        self.safety = mx.array(safety, dtype=dtype)
        self.ifactor = mx.array(ifactor, dtype=dtype)
        self.dfactor = mx.array(dfactor, dtype=dtype)
        self.max_num_steps = max_num_steps
        self.dtype = dtype

        self.step_t = None if step_t is None else mx.array(step_t, dtype=dtype)
        self.jump_t = None if jump_t is None else mx.array(jump_t, dtype=dtype)

        # Copy tableau coefficients to instance for device/dtype handling
        self.tableau = _ButcherTableau(
            alpha=self.tableau.alpha.astype(y0.dtype),
            beta=[b.astype(y0.dtype) for b in self.tableau.beta],
            c_sol=self.tableau.c_sol.astype(y0.dtype),
            c_error=self.tableau.c_error.astype(y0.dtype),
        )
        self.mid = self.mid.astype(y0.dtype)

    @classmethod
    def valid_callbacks(cls):
        return super().valid_callbacks() | {'callback_step',
                                             'callback_accept_step',
                                             'callback_reject_step'}

    def _before_integrate(self, t):
        t0 = t[0]
        f0 = self.func(t[0], self.y0)
        if self.first_step is None:
            first_step = _select_initial_step(self.func, t[0], self.y0, self.order - 1,
                                               self.rtol, self.atol, self.norm, f0=f0)
        else:
            first_step = self.first_step
        self.rk_state = _RungeKuttaState(self.y0, f0, t[0], t[0], first_step, [self.y0] * 5)

        # Handle step_t and jump_t arguments
        if self.step_t is None:
            step_t = mx.array([], dtype=self.dtype)
        else:
            step_t = _sort_tvals(self.step_t, t0)
            step_t = step_t.astype(self.dtype)
        if self.jump_t is None:
            jump_t = mx.array([], dtype=self.dtype)
        else:
            jump_t = _sort_tvals(self.jump_t, t0)
            jump_t = jump_t.astype(self.dtype)

        self.step_t = step_t
        self.jump_t = jump_t
        self.next_step_index = min(bisect.bisect(step_t.tolist(), t0.item()), len(step_t) - 1)
        self.next_jump_index = min(bisect.bisect(jump_t.tolist(), t0.item()), len(jump_t) - 1)

    def _advance(self, next_t):
        """Interpolate through the next time point, integrating as necessary."""
        n_steps = 0
        while next_t.item() > self.rk_state.t1.item():
            assert n_steps < self.max_num_steps, f'max_num_steps exceeded ({n_steps}>={self.max_num_steps})'
            self.rk_state = self._adaptive_step(self.rk_state)
            n_steps += 1
        return _interp_evaluate(self.rk_state.interp_coeff, self.rk_state.t0, self.rk_state.t1, next_t)

    def _advance_until_event(self, event_fn):
        """Returns t, state(t) such that event_fn(t, state(t)) == 0."""
        if event_fn(self.rk_state.t1, self.rk_state.y1).item() == 0:
            return (self.rk_state.t1, self.rk_state.y1)

        n_steps = 0
        sign0 = mx.sign(event_fn(self.rk_state.t1, self.rk_state.y1))
        while mx.sign(event_fn(self.rk_state.t1, self.rk_state.y1)).item() == sign0.item():
            assert n_steps < self.max_num_steps, f'max_num_steps exceeded ({n_steps}>={self.max_num_steps})'
            self.rk_state = self._adaptive_step(self.rk_state)
            n_steps += 1
        interp_fn = lambda t: _interp_evaluate(self.rk_state.interp_coeff, self.rk_state.t0, self.rk_state.t1, t)
        return find_event(interp_fn, sign0, self.rk_state.t0, self.rk_state.t1, event_fn, self.atol)

    def _adaptive_step(self, rk_state):
        """Take an adaptive Runge-Kutta step to integrate the ODE."""
        y0, f0, _, t0, dt, interp_coeff = rk_state

        # Clamp step size (materialize scalar for control flow)
        if not mx.isfinite(dt).item():
            dt = self.min_step
        dt = mx.clip(dt, self.min_step, self.max_step)

        self.func.callback_step(t0, y0, dt)
        t1 = t0 + dt

        # Handle step_t and jump_t (prescribed grid points)
        on_step_t = False
        if len(self.step_t):
            next_step_t = self.step_t[self.next_step_index]
            cond = (t0.item() < next_step_t.item()) and (next_step_t.item() < (t0 + dt).item())
            if cond:
                on_step_t = True
                t1 = next_step_t
                dt = t1 - t0

        on_jump_t = False
        if len(self.jump_t):
            next_jump_t = self.jump_t[self.next_jump_index]
            cond = (t0.item() < next_jump_t.item()) and (next_jump_t.item() < (t0 + dt).item())
            if cond:
                on_jump_t = True
                on_step_t = False
                t1 = next_jump_t
                dt = t1 - t0

        # Take the RK step
        y1, f1, y1_error, k = _runge_kutta_step(
            self.func, y0, f0, t0, dt, t1, tableau=self.tableau
        )

        # Compute error ratio
        error_ratio = _compute_error_ratio(y1_error, self.rtol, self.atol, y0, y1, self.norm)
        accept_step = error_ratio.item() <= 1

        # Handle min/max stepping
        if dt.item() > self.max_step.item():
            accept_step = False
        if dt.item() <= self.min_step.item():
            accept_step = True

        # Update RK state based on acceptance
        if accept_step:
            self.func.callback_accept_step(t0, y0, dt)
            t_next = t1
            y_next = y1
            interp_coeff = self._interp_fit(y0, y_next, k, dt)
            if on_step_t:
                if self.next_step_index != len(self.step_t) - 1:
                    self.next_step_index += 1
            if on_jump_t:
                if self.next_jump_index != len(self.jump_t) - 1:
                    self.next_jump_index += 1
                f1 = self.func(t_next, y_next)
            f_next = f1
        else:
            self.func.callback_reject_step(t0, y0, dt)
            t_next = t0
            y_next = y0
            f_next = f0

        dt_next = _optimal_step_size(dt, error_ratio, self.safety, self.ifactor, self.dfactor, self.order)
        dt_next = mx.clip(dt_next, self.min_step, self.max_step)

        return _RungeKuttaState(y_next, f_next, t0, t_next, dt_next, interp_coeff)

    def _interp_fit(self, y0, y1, k, dt):
        """Fit an interpolating polynomial to the results of a Runge-Kutta step."""
        dt = dt.astype(y0.dtype)
        y_mid = y0 + mx.sum(k * (dt * self.mid), axis=-1)
        f0 = k[..., 0]
        f1 = k[..., -1]
        return _interp_fit(y0, y1, y_mid, f0, f1, dt)


def _sort_tvals(tvals, t0):
    tvals = tvals[tvals >= t0]
    return mx.sort(tvals)
