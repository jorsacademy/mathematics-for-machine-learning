"""Matrix factorization examples: QR and Cholesky decomposition."""

import numpy as np


def qr_factorization(A):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be a matrix.")
    return np.linalg.qr(A, mode="reduced")


def cholesky_factorization(A, tol=1e-10):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")
    if not np.allclose(A, A.T, atol=tol, rtol=0.0):
        raise ValueError("A must be symmetric.")
    return np.linalg.cholesky(A)


if __name__ == "__main__":
    A = np.array([[4.0, 2.0], [2.0, 3.0]])
    L = cholesky_factorization(A)
    print("Cholesky reconstruction error:", np.linalg.norm(A - L @ L.T))
    Q, R = qr_factorization(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    print("Q^T Q:\n", Q.T @ Q)
