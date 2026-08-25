"""Vector norms, orthogonality, projection, and Gram-Schmidt."""

import numpy as np


def lp_norm(x, p=2):
    x = np.asarray(x, dtype=float)
    if p < 1:
        raise ValueError("p must be at least 1.")
    return float(np.linalg.norm(x, ord=p))


def project(u, v):
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    denominator = v @ v
    if denominator == 0.0:
        raise ValueError("Cannot project onto the zero vector.")
    return ((u @ v) / denominator) * v


def gram_schmidt(vectors, tol=1e-12):
    V = np.asarray(vectors, dtype=float)
    if V.ndim != 2:
        raise ValueError("vectors must be a 2D array.")
    basis = []
    for v in V:
        w = v.copy()
        for q in basis:
            w -= (w @ q) * q
        norm = np.linalg.norm(w)
        if norm > tol:
            basis.append(w / norm)
    return np.asarray(basis)


if __name__ == "__main__":
    vectors = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    Q = gram_schmidt(vectors)
    print("orthonormal basis:\n", Q)
    print("Q Q^T:\n", Q @ Q.T)
