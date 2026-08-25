"""Principal Component Analysis implemented from centered data using SVD."""

import numpy as np


def pca(X, n_components):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError("X must be a 2D array with at least two samples.")
    if not 1 <= n_components <= min(X.shape):
        raise ValueError("Invalid n_components.")
    mean = X.mean(axis=0)
    Xc = X - mean
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:n_components]
    scores = Xc @ components.T
    explained_variance = (s[:n_components] ** 2) / (X.shape[0] - 1)
    total_variance = np.sum(s**2) / (X.shape[0] - 1)
    ratio = explained_variance / total_variance if total_variance > 0 else np.zeros_like(explained_variance)
    return scores, components, mean, explained_variance, ratio


if __name__ == "__main__":
    X = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0]])
    scores, components, mean, variance, ratio = pca(X, 1)
    print("mean:", mean)
    print("component:", components)
    print("explained variance ratio:", ratio)
    print("scores:\n", scores)
