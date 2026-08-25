"""Functions and their basic mathematical properties."""

import numpy as np


def affine(x, a=2.0, b=1.0):
    return a * np.asarray(x) + b


def quadratic(x):
    x = np.asarray(x)
    return x**2


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    # Numerically stable sigmoid.
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def is_even(function, points, tol=1e-10):
    x = np.asarray(points, dtype=float)
    return bool(np.allclose(function(x), function(-x), atol=tol, rtol=0.0))


if __name__ == "__main__":
    points = np.linspace(-3.0, 3.0, 7)
    print("quadratic is even:", is_even(quadratic, points))
    print("sigmoid values:", sigmoid(points))
