"""Manifold-learning intuition via classical multidimensional scaling (MDS)."""

import numpy as np


def classical_mds(distance_matrix, dimensions=2, tol=1e-12):
    D = np.asarray(distance_matrix, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("distance_matrix must be square.")
    if not np.allclose(D, D.T, atol=tol) or np.any(D < -tol) or not np.allclose(np.diag(D), 0.0, atol=tol):
        raise ValueError("distance_matrix must be symmetric, non-negative, and have a zero diagonal.")
    n = D.shape[0]
    if not 1 <= dimensions < n:
        raise ValueError("dimensions must be between 1 and n-1.")
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D**2) @ J
    values, vectors = np.linalg.eigh(B)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    positive = np.maximum(values[:dimensions], 0.0)
    return vectors[:, :dimensions] * np.sqrt(positive)


if __name__ == "__main__":
    points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    D = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    print(classical_mds(D, 2))
