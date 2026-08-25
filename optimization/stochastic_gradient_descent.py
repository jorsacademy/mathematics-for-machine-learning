"""Stochastic Gradient Descent for one-dimensional linear regression."""

from __future__ import annotations

import numpy as np


def fit_linear_regression_sgd(
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    epochs: int = 100,
    seed: int = 42,
) -> tuple[float, float]:
    """Fit y ≈ wx + b using one sample per parameter update."""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    if x.size != y.size or x.size == 0:
        raise ValueError("x and y must be non-empty and have equal length.")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if epochs <= 0:
        raise ValueError("epochs must be positive.")

    rng = np.random.default_rng(seed)
    w = 0.0
    b = 0.0

    for _ in range(epochs):
        for i in rng.permutation(x.size):
            prediction = w * x[i] + b
            error = prediction - y[i]

            # For L_i = 0.5 * (prediction - target)^2:
            grad_w = error * x[i]
            grad_b = error

            w -= learning_rate * grad_w
            b -= learning_rate * grad_b

    return w, b


if __name__ == "__main__":
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = 2.0 * x + 1.0
    w, b = fit_linear_regression_sgd(x, y, learning_rate=0.01, epochs=500)
    print("Estimated weight:", w)
    print("Estimated bias:", b)
