"""Basic graph and spectral graph operations using adjacency matrices."""

import numpy as np


def graph_laplacian(adjacency, normalized=False):
    A = np.asarray(adjacency, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or np.any(A < 0):
        raise ValueError("adjacency must be a non-negative square matrix.")
    if not np.allclose(A, A.T):
        raise ValueError("This example expects an undirected graph.")
    degrees = A.sum(axis=1)
    L = np.diag(degrees) - A
    if not normalized:
        return L
    inv_sqrt = np.zeros_like(degrees)
    mask = degrees > 0
    inv_sqrt[mask] = 1.0 / np.sqrt(degrees[mask])
    D = np.diag(inv_sqrt)
    return D @ L @ D


def spectral_embedding(adjacency, dimensions=2):
    L = graph_laplacian(adjacency, normalized=True)
    values, vectors = np.linalg.eigh(L)
    if dimensions < 1 or dimensions >= L.shape[0]:
        raise ValueError("dimensions must be between 1 and n-1.")
    return values[1:dimensions + 1], vectors[:, 1:dimensions + 1]


if __name__ == "__main__":
    A = np.array([[0,1,1,0],[1,0,1,0],[1,1,0,1],[0,0,1,0]], dtype=float)
    values, embedding = spectral_embedding(A, 2)
    print("nontrivial eigenvalues:", values)
    print("embedding:\n", embedding)
