"""Reference implementation of the Adam optimizer for vector parameters."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Array = np.ndarray


def adam(
    grad: Callable[[Array], Array],
    x0: Array,
    learning_rate: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    steps: int = 500,
    epsilon: float = 1e-8,
) -> Array:
    """Minimize an objective using Adam with bias-corrected moments."""
    if learning_rate <= 0.0 or epsilon <= 0.0 or steps <= 0:
        raise ValueError("learning_rate, epsilon, and steps must be positive.")
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError("beta1 and beta2 must lie in [0, 1).")

    x = np.asarray(x0, dtype=float).copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)

    for t in range(1, steps + 1):
        g = np.asarray(grad(x), dtype=float)
        if g.shape != x.shape:
            raise ValueError("Gradient shape must match parameter shape.")

        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)

        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)

        x -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    return x


if __name__ == "__main__":
    # f(x) = (x_1 - 3)^2 + 4(x_2 + 1)^2
    def gradient(x: Array) -> Array:
        return np.array([2.0 * (x[0] - 3.0), 8.0 * (x[1] + 1.0)])

    start = np.array([8.0, 5.0])
    solution = adam(gradient, start, learning_rate=0.05, steps=500)
    print("Adam solution:", solution)
