"""Momentum and Nesterov Accelerated Gradient on a scalar objective."""

from __future__ import annotations

from collections.abc import Callable


def momentum(
    grad: Callable[[float], float],
    x0: float,
    learning_rate: float = 0.05,
    momentum_coefficient: float = 0.9,
    steps: int = 100,
) -> float:
    """Minimize a scalar objective using classical momentum.

    Velocity convention:
        v_t = beta * v_{t-1} - eta * grad(x_t)
        x_{t+1} = x_t + v_t
    """
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if not 0.0 <= momentum_coefficient < 1.0:
        raise ValueError("momentum_coefficient must lie in [0, 1).")
    if steps <= 0:
        raise ValueError("steps must be positive.")

    x = float(x0)
    velocity = 0.0
    beta = momentum_coefficient

    for _ in range(steps):
        velocity = beta * velocity - learning_rate * grad(x)
        x += velocity

    return x


def nesterov(
    grad: Callable[[float], float],
    x0: float,
    learning_rate: float = 0.05,
    momentum_coefficient: float = 0.9,
    steps: int = 100,
) -> float:
    """Minimize a scalar objective using Nesterov momentum.

    The gradient is evaluated at the look-ahead point x + beta * v.
    """
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if not 0.0 <= momentum_coefficient < 1.0:
        raise ValueError("momentum_coefficient must lie in [0, 1).")
    if steps <= 0:
        raise ValueError("steps must be positive.")

    x = float(x0)
    velocity = 0.0
    beta = momentum_coefficient

    for _ in range(steps):
        look_ahead = x + beta * velocity
        velocity = beta * velocity - learning_rate * grad(look_ahead)
        x += velocity

    return x


if __name__ == "__main__":
    # f(x) = (x - 3)^2 has gradient 2(x - 3) and minimum at x = 3.
    gradient = lambda x: 2.0 * (x - 3.0)

    print("Momentum solution:", momentum(gradient, x0=-5.0))
    print("Nesterov solution:", nesterov(gradient, x0=-5.0))
