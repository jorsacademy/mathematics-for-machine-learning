"""Kernel functions and Gram matrices."""

import numpy as np


def linear_kernel(x, y):
    return float(np.asarray(x, dtype=float) @ np.asarray(y, dtype=float))


def polynomial_kernel(x, y, degree=2, coef0=1.0):
    if degree < 1 or not isinstance(degree, int):
        raise ValueError("degree must be a positive integer.")
    return (linear_kernel(x, y) + coef0) ** degree


def rbf_kernel(x, y, gamma=1.0):
    if gamma <= 0:
        raise ValueError("gamma must be positive.")
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    return float(np.exp(-gamma * (d @ d)))


def gram_matrix(X, kernel):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    n = X.shape[0]
    K = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            K[i, j] = kernel(X[i], X[j])
    return K


if __name__ == "__main__":
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    K = gram_matrix(X, lambda a, b: rbf_kernel(a, b, gamma=0.5))
    print("RBF Gram matrix:\n", K)
    print("eigenvalues:", np.linalg.eigvalsh(K))
