import abc
import mlx.core as mx
from .event_handling import find_event
from .misc import _handle_unused_kwargs


class AdaptiveStepsizeODESolver(metaclass=abc.ABCMeta):
    def __init__(self, dtype, y0, norm, **unused_kwargs):
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.y0 = y0
        self.dtype = dtype
        self.norm = norm

    def _before_integrate(self, t):
        pass

    @abc.abstractmethod
    def _advance(self, next_t):
        raise NotImplementedError

    @classmethod
    def valid_callbacks(cls):
        return set()

    def integrate(self, t):
        solution_list = [self.y0]
        t = t.astype(self.dtype)
        self._before_integrate(t)
        for i in range(1, len(t)):
            solution_list.append(self._advance(t[i]))
        return mx.stack(solution_list)


class AdaptiveStepsizeEventODESolver(AdaptiveStepsizeODESolver, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def _advance_until_event(self, event_fn):
        raise NotImplementedError

    def integrate_until_event(self, t0, event_fn):
        t0 = t0.astype(self.dtype)
        self._before_integrate(t0.reshape(-1))
        event_time, y1 = self._advance_until_event(event_fn)
        solution = mx.stack([self.y0, y1])
        return event_time, solution


class FixedGridODESolver(metaclass=abc.ABCMeta):
    order: int

    def __init__(self, func, y0, step_size=None, grid_constructor=None, interp="linear", perturb=False, **unused_kwargs):
        self.atol = unused_kwargs.pop('atol')
        unused_kwargs.pop('rtol', None)
        unused_kwargs.pop('norm', None)
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.func = func
        self.y0 = y0
        self.dtype = y0.dtype
        self.device = y0.dtype  # MLX doesn't expose device on array directly
        self.step_size = step_size
        self.interp = interp
        self.perturb = perturb

        if step_size is None:
            if grid_constructor is None:
                self.grid_constructor = lambda f, y0, t: t
            else:
                self.grid_constructor = grid_constructor
        else:
            if grid_constructor is None:
                self.grid_constructor = self._grid_constructor_from_step_size(step_size)
            else:
                raise ValueError("step_size and grid_constructor are mutually exclusive arguments.")

    @classmethod
    def valid_callbacks(cls):
        return {'callback_step'}

    @staticmethod
    def _grid_constructor_from_step_size(step_size):
        def _grid_constructor(func, y0, t):
            start_time = t[0]
            end_time = t[-1]
            niters = int(mx.ceil((end_time - start_time) / step_size + 1).item())
            t_infer = mx.arange(0, niters, dtype=t.dtype) * step_size + start_time
            t_infer = mx.where(mx.arange(0, niters, dtype=t.dtype) == niters - 1, end_time, t_infer)
            return t_infer
        return _grid_constructor

    @abc.abstractmethod
    def _step_func(self, func, t0, dt, t1, y0):
        pass

    def integrate(self, t):
        time_grid = self.grid_constructor(self.func, self.y0, t)
        assert time_grid[0].item() == t[0].item() and time_grid[-1].item() == t[-1].item()

        solution_list = [self.y0]
        j = 1
        y0 = self.y0
        for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
            dt = t1 - t0
            self.func.callback_step(t0, y0, dt)
            dy, f0 = self._step_func(self.func, t0, dt, t1, y0)
            y1 = y0 + dy

            while j < len(t) and t1.item() >= t[j].item():
                if self.interp == "linear":
                    solution_list.append(self._linear_interp(t0, t1, y0, y1, t[j]))
                elif self.interp == "cubic":
                    f1 = self.func(t1, y1)
                    solution_list.append(self._cubic_hermite_interp(t0, y0, f0, t1, y1, f1, t[j]))
                else:
                    raise ValueError(f"Unknown interpolation method {self.interp}")
                j += 1
            y0 = y1

        return mx.stack(solution_list)

    def integrate_until_event(self, t0, event_fn):
        assert self.step_size is not None, \
            "Event handling for fixed step solvers currently requires `step_size` to be provided in options."

        t0 = t0.astype(self.y0.dtype)
        y0 = self.y0
        dt = self.step_size

        sign0 = mx.sign(event_fn(t0, y0))
        max_itrs = 20000
        itr = 0
        while True:
            itr += 1
            t1 = t0 + dt
            dy, f0 = self._step_func(self.func, t0, dt, t1, y0)
            y1 = y0 + dy

            sign1 = mx.sign(event_fn(t1, y1))

            if sign0.item() != sign1.item():
                if self.interp == "linear":
                    interp_fn = lambda t: self._linear_interp(t0, t1, y0, y1, t)
                elif self.interp == "cubic":
                    f1 = self.func(t1, y1)
                    interp_fn = lambda t: self._cubic_hermite_interp(t0, y0, f0, t1, y1, f1, t)
                else:
                    raise ValueError(f"Unknown interpolation method {self.interp}")
                event_time, y1 = find_event(interp_fn, sign0, t0, t1, event_fn, float(self.atol.item() if hasattr(self.atol, 'item') else self.atol))
                break
            else:
                t0, y0 = t1, y1

            if itr >= max_itrs:
                raise RuntimeError(f"Reached maximum number of iterations {max_itrs}.")
        solution = mx.stack([self.y0, y1])
        return event_time, solution

    def _cubic_hermite_interp(self, t0, y0, f0, t1, y1, f1, t):
        h = (t - t0) / (t1 - t0)
        h00 = (1 + 2 * h) * (1 - h) * (1 - h)
        h10 = h * (1 - h) * (1 - h)
        h01 = h * h * (3 - 2 * h)
        h11 = h * h * (h - 1)
        dt = (t1 - t0)
        return h00 * y0 + h10 * dt * f0 + h01 * y1 + h11 * dt * f1

    def _linear_interp(self, t0, t1, y0, y1, t):
        if t.item() == t0.item():
            return y0
        if t.item() == t1.item():
            return y1
        slope = (t - t0) / (t1 - t0)
        return y0 + slope * (y1 - y0)
