"""Eigenvalues and eigenvectors for real symmetric matrices."""

import numpy as np


def symmetric_eigendecomposition(A, tol=1e-10):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")
    if not np.allclose(A, A.T, atol=tol, rtol=0.0):
        raise ValueError("This example expects a symmetric matrix.")
    values, vectors = np.linalg.eigh(A)
    return values, vectors


def verify_eigenpairs(A, values, vectors, tol=1e-8):
    A = np.asarray(A, dtype=float)
    return all(np.linalg.norm(A @ vectors[:, i] - values[i] * vectors[:, i]) <= tol for i in range(len(values)))


if __name__ == "__main__":
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    values, vectors = symmetric_eigendecomposition(A)
    print("eigenvalues:", values)
    print("eigenvectors:\n", vectors)
    print("verified:", verify_eigenpairs(A, values, vectors))
