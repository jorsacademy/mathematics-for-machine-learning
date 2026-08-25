"""Reference implementations of AdaGrad and RMSprop for vector parameters."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Array = np.ndarray


def adagrad(
    grad: Callable[[Array], Array],
    x0: Array,
    learning_rate: float = 0.1,
    steps: int = 200,
    epsilon: float = 1e-8,
) -> Array:
    """Minimize an objective using AdaGrad."""
    if learning_rate <= 0.0 or epsilon <= 0.0 or steps <= 0:
        raise ValueError("learning_rate, epsilon, and steps must be positive.")

    x = np.asarray(x0, dtype=float).copy()
    accumulated_squared_gradients = np.zeros_like(x)

    for _ in range(steps):
        g = np.asarray(grad(x), dtype=float)
        if g.shape != x.shape:
            raise ValueError("Gradient shape must match parameter shape.")
        accumulated_squared_gradients += g * g
        x -= learning_rate * g / (np.sqrt(accumulated_squared_gradients) + epsilon)

    return x


def rmsprop(
    grad: Callable[[Array], Array],
    x0: Array,
    learning_rate: float = 0.01,
    decay: float = 0.9,
    steps: int = 200,
    epsilon: float = 1e-8,
) -> Array:
    """Minimize an objective using RMSprop."""
    if learning_rate <= 0.0 or epsilon <= 0.0 or steps <= 0:
        raise ValueError("learning_rate, epsilon, and steps must be positive.")
    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must lie in [0, 1).")

    x = np.asarray(x0, dtype=float).copy()
    mean_squared_gradient = np.zeros_like(x)

    for _ in range(steps):
        g = np.asarray(grad(x), dtype=float)
        if g.shape != x.shape:
            raise ValueError("Gradient shape must match parameter shape.")
        mean_squared_gradient = decay * mean_squared_gradient + (1.0 - decay) * (g * g)
        x -= learning_rate * g / (np.sqrt(mean_squared_gradient) + epsilon)

    return x


if __name__ == "__main__":
    # f(x) = (x_1 - 3)^2 + 4(x_2 + 1)^2
    def gradient(x: Array) -> Array:
        return np.array([2.0 * (x[0] - 3.0), 8.0 * (x[1] + 1.0)])

    start = np.array([8.0, 5.0])
    print("AdaGrad:", adagrad(gradient, start, learning_rate=0.8, steps=500))
    print("RMSprop:", rmsprop(gradient, start, learning_rate=0.05, steps=500))
