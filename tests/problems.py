"""Standard ODE test problems with known analytical solutions."""
import math
import mlx.core as mx
import mlx.nn as nn


class ExponentialODE:
    """dy/dt = y, y(0)=1 -> y(t) = exp(t)"""

    def __call__(self, t, y):
        return y

    @staticmethod
    def solution(t, y0=1.0):
        return y0 * mx.exp(t)


class HarmonicOscillator(nn.Module):
    """dy/dt = [v, -omega^2 * x], y(0)=[1, 0] -> x=cos(omega*t), v=-omega*sin(omega*t)"""

    def __init__(self, omega=1.0):
        super().__init__()
        self.omega = omega

    def __call__(self, t, y):
        x, v = y[0], y[1]
        return mx.array([v, -(self.omega ** 2) * x])

    def solution(self, t, y0=None):
        omega = self.omega
        # y0 = [x0, v0] -> x(t) = x0*cos(wt) + v0/w*sin(wt), v(t) = -x0*w*sin(wt) + v0*cos(wt)
        if y0 is None:
            y0 = mx.array([1.0, 0.0])
        x0, v0 = y0[0], y0[1]
        x = x0 * mx.cos(omega * t) + (v0 / omega) * mx.sin(omega * t)
        v = -x0 * omega * mx.sin(omega * t) + v0 * mx.cos(omega * t)
        return mx.array([x, v])


class LinearODE:
    """dy/dt = -0.5*y, y(0)=1 -> y(t) = exp(-0.5*t)"""

    def __call__(self, t, y):
        return -0.5 * y

    @staticmethod
    def solution(t, y0=1.0):
        return y0 * mx.exp(-0.5 * t)


class SinusoidalODE:
    """dy/dt = cos(t), y(0)=0 -> y(t) = sin(t)"""

    def __call__(self, t, y):
        return mx.cos(t)

    @staticmethod
    def solution(t, y0=0.0):
        return y0 + mx.sin(t)
