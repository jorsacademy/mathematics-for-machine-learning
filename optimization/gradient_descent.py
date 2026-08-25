"""Optimization fundamentals on differentiable scalar objectives."""

import numpy as np


def gradient_descent(gradient, x0, learning_rate=0.1, steps=100):
    if learning_rate <= 0 or steps < 0:
        raise ValueError("learning_rate must be positive and steps non-negative.")
    x = np.asarray(x0, dtype=float).copy()
    history = [x.copy()]
    for _ in range(steps):
        g = np.asarray(gradient(x), dtype=float)
        if g.shape != x.shape or not np.all(np.isfinite(g)):
            raise ValueError("Gradient must be finite and have the same shape as x.")
        x -= learning_rate * g
        history.append(x.copy())
    return x, np.asarray(history)


def quadratic_gradient(x):
    return 2.0 * (x - np.array([3.0, -1.0]))


if __name__ == "__main__":
    optimum, _ = gradient_descent(quadratic_gradient, [10.0, 5.0], learning_rate=0.1, steps=100)
    print("estimated optimum:", optimum)
