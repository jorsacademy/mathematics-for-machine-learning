"""Linear independence, rank, and basis examples using numerical linear algebra."""

from __future__ import annotations

import numpy as np


def matrix_from_column_vectors(vectors: list[np.ndarray]) -> np.ndarray:
    """Stack vectors as columns of a matrix."""
    if not vectors:
        raise ValueError("At least one vector is required.")
    arrays = [np.asarray(v, dtype=float).reshape(-1) for v in vectors]
    dimension = arrays[0].size
    if any(v.size != dimension for v in arrays):
        raise ValueError("All vectors must have the same dimension.")
    return np.column_stack(arrays)


def are_linearly_independent(vectors: list[np.ndarray], tol: float | None = None) -> bool:
    """Return True exactly when the supplied column vectors have full column rank."""
    A = matrix_from_column_vectors(vectors)
    if A.shape[1] > A.shape[0]:
        return False
    rank = np.linalg.matrix_rank(A, tol=tol)
    return rank == A.shape[1]


def basis_from_spanning_set(vectors: list[np.ndarray], tol: float = 1e-10) -> list[np.ndarray]:
    """Extract a basis from a finite spanning set while preserving input order.

    A vector is retained when it increases the numerical rank of the vectors
    selected so far.
    """
    arrays = [np.asarray(v, dtype=float).reshape(-1) for v in vectors]
    if not arrays:
        return []
    dimension = arrays[0].size
    if any(v.size != dimension for v in arrays):
        raise ValueError("All vectors must have the same dimension.")

    basis: list[np.ndarray] = []
    current_rank = 0
    for vector in arrays:
        candidate = basis + [vector]
        rank = np.linalg.matrix_rank(np.column_stack(candidate), tol=tol)
        if rank > current_rank:
            basis.append(vector.copy())
            current_rank = rank
    return basis


if __name__ == "__main__":
    vectors = [
        np.array([1.0, 2.0, 3.0]),
        np.array([2.0, 4.0, 6.0]),
        np.array([3.0, 5.0, 7.0]),
    ]

    print("Independent:", are_linearly_independent(vectors))
    print("Basis extracted from the spanning set:")
    for vector in basis_from_spanning_set(vectors):
        print(vector)
