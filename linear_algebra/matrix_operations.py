"""Educational examples of core matrix operations.

The functions favor mathematical clarity and explicit validation over convenience.
"""

from __future__ import annotations

import numpy as np


def add(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return A + B after checking that both matrices have the same shape."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.shape != B.shape:
        raise ValueError("Matrix addition requires identical shapes.")
    return A + B


def multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return the matrix product AB."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("Matrix multiplication requires two 2D arrays.")
    if A.shape[1] != B.shape[0]:
        raise ValueError("Inner dimensions must agree for matrix multiplication.")
    return A @ B


def transpose(A: np.ndarray) -> np.ndarray:
    """Return the transpose of A."""
    return np.asarray(A, dtype=float).T


def determinant(A: np.ndarray) -> float:
    """Return det(A) for a square matrix."""
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("The determinant is defined here only for square matrices.")
    return float(np.linalg.det(A))


def inverse(A: np.ndarray) -> np.ndarray:
    """Return A^{-1}; raise a clear error when A is singular."""
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Only square matrices can have a two-sided inverse.")
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Matrix is singular and therefore not invertible.") from exc


def trace(A: np.ndarray) -> float:
    """Return the trace of a square matrix."""
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Trace is defined here only for square matrices.")
    return float(np.trace(A))


if __name__ == "__main__":
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0], [7.0, 8.0]])

    print("A + B:\n", add(A, B))
    print("AB:\n", multiply(A, B))
    print("A^T:\n", transpose(A))
    print("det(A):", determinant(A))
    print("A^{-1}:\n", inverse(A))
    print("tr(A):", trace(A))
