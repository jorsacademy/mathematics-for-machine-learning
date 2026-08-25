"""Utilities for convexity checks on differentiable quadratic functions."""

import numpy as np


def is_positive_semidefinite(matrix, tol=1e-10):
    A = np.asarray(matrix, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("matrix must be square.")
    if not np.allclose(A, A.T, atol=tol, rtol=0.0):
        return False
    eigenvalues = np.linalg.eigvalsh(A)
    return bool(np.all(eigenvalues >= -tol))


def quadratic_is_convex(Q, tol=1e-10):
    """For f(x)=0.5*x^T Q*x+c^T x, convexity requires symmetric PSD Q."""
    return is_positive_semidefinite(Q, tol=tol)


def convex_combination(x, y, alpha):
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1].")
    return alpha * np.asarray(x, dtype=float) + (1.0 - alpha) * np.asarray(y, dtype=float)


if __name__ == "__main__":
    Q = np.array([[2.0, 0.0], [0.0, 4.0]])
    print("quadratic is convex:", quadratic_is_convex(Q))
