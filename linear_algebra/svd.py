"""Singular Value Decomposition and low-rank approximation."""

import numpy as np


def svd(A):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be a matrix.")
    return np.linalg.svd(A, full_matrices=False)


def rank_k_approximation(A, k):
    U, s, Vt = svd(A)
    if not 1 <= k <= len(s):
        raise ValueError("k must be between 1 and min(A.shape).")
    return (U[:, :k] * s[:k]) @ Vt[:k, :]


if __name__ == "__main__":
    A = np.array([[3.0, 2.0, 2.0], [2.0, 3.0, -2.0]])
    U, s, Vt = svd(A)
    print("singular values:", s)
    print("reconstruction error:", np.linalg.norm(A - (U * s) @ Vt))
    print("rank-1 approximation:\n", rank_k_approximation(A, 1))
