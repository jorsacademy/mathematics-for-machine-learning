"""A small non-convex optimization experiment showing initialization dependence."""

import numpy as np


def objective(x):
    return x**4 - 4.0 * x**2 + 0.25 * x


def gradient(x):
    return 4.0 * x**3 - 8.0 * x + 0.25


def gradient_descent(x0, learning_rate=0.01, steps=1000):
    x = float(x0)
    for _ in range(steps):
        x -= learning_rate * gradient(x)
        if not np.isfinite(x):
            raise FloatingPointError("Optimization diverged.")
    return x, objective(x)


if __name__ == "__main__":
    for start in (-3.0, -0.5, 0.5, 3.0):
        x, value = gradient_descent(start)
        print(f"start={start: .1f}, solution={x: .6f}, objective={value: .6f}")
