"""Gradients, directional derivatives, and Hessians for multivariable calculus."""

import numpy as np


def quadratic_value(x, A, b=None, c=0.0):
    x = np.asarray(x, dtype=float)
    A = np.asarray(A, dtype=float)
    b = np.zeros_like(x) if b is None else np.asarray(b, dtype=float)
    return 0.5 * x @ A @ x + b @ x + c


def quadratic_gradient(x, A, b=None):
    x = np.asarray(x, dtype=float)
    A = np.asarray(A, dtype=float)
    b = np.zeros_like(x) if b is None else np.asarray(b, dtype=float)
    return 0.5 * (A + A.T) @ x + b


def directional_derivative(gradient, direction):
    g = np.asarray(gradient, dtype=float)
    d = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(d)
    if norm == 0.0:
        raise ValueError("direction must be non-zero.")
    return float(g @ (d / norm))


def quadratic_hessian(A):
    A = np.asarray(A, dtype=float)
    return 0.5 * (A + A.T)


if __name__ == "__main__":
    A = np.array([[4.0, 1.0], [1.0, 2.0]])
    x = np.array([1.0, 2.0])
    g = quadratic_gradient(x, A)
    print("value:", quadratic_value(x, A))
    print("gradient:", g)
    print("directional derivative:", directional_derivative(g, [1.0, 1.0]))
    print("Hessian:\n", quadratic_hessian(A))
