"""Mini-batch Gradient Descent for one-dimensional linear regression."""

from __future__ import annotations

import numpy as np


def fit_linear_regression_minibatch(
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    epochs: int = 100,
    batch_size: int = 32,
    seed: int = 42,
) -> tuple[float, float]:
    """Fit y ≈ wx + b using mini-batch gradient updates."""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    if x.size != y.size or x.size == 0:
        raise ValueError("x and y must be non-empty and have equal length.")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    rng = np.random.default_rng(seed)
    w = 0.0
    b = 0.0

    for _ in range(epochs):
        indices = rng.permutation(x.size)
        x_shuffled = x[indices]
        y_shuffled = y[indices]

        for start in range(0, x.size, batch_size):
            xb = x_shuffled[start : start + batch_size]
            yb = y_shuffled[start : start + batch_size]

            predictions = w * xb + b
            errors = predictions - yb

            # Mean of per-example gradients for 0.5 * squared error.
            grad_w = float(np.mean(errors * xb))
            grad_b = float(np.mean(errors))

            w -= learning_rate * grad_w
            b -= learning_rate * grad_b

    return w, b


if __name__ == "__main__":
    x = np.linspace(0.0, 5.0, 200)
    y = 3.0 * x - 2.0
    w, b = fit_linear_regression_minibatch(
        x,
        y,
        learning_rate=0.01,
        epochs=500,
        batch_size=16,
    )
    print("Estimated weight:", w)
    print("Estimated bias:", b)
