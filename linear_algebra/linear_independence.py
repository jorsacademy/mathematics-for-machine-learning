"""Finite-dimensional tests for linear independence and basis extraction."""

import numpy as np


def matrix_rank(A, tol=None):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be a matrix.")
    return int(np.linalg.matrix_rank(A, tol=tol))


def linearly_independent(vectors, tol=None):
    V = np.asarray(vectors, dtype=float)
    if V.ndim != 2:
        raise ValueError("vectors must form a 2D array.")
    return matrix_rank(V.T, tol=tol) == V.shape[0]


def basis_from_columns(A, tol=1e-10):
    """Return independent original columns using incremental rank tests."""
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be a matrix.")
    selected = []
    current = np.empty((A.shape[0], 0))
    rank = 0
    for j in range(A.shape[1]):
        candidate = np.column_stack((current, A[:, j]))
        new_rank = np.linalg.matrix_rank(candidate, tol=tol)
        if new_rank > rank:
            selected.append(j)
            current = candidate
            rank = new_rank
    return current, selected


if __name__ == "__main__":
    A = np.array([[1, 2, 3], [2, 4, 5], [3, 6, 7]], dtype=float)
    basis, indices = basis_from_columns(A)
    print("rank:", matrix_rank(A))
    print("basis column indices:", indices)
    print("basis:\n", basis)
