"""A minimal graph convolution layer using normalized adjacency."""

import numpy as np


def normalized_adjacency(adjacency, add_self_loops=True):
    A = np.asarray(adjacency, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or np.any(A < 0):
        raise ValueError("adjacency must be a non-negative square matrix.")
    if add_self_loops:
        A = A + np.eye(A.shape[0])
    degrees = A.sum(axis=1)
    if np.any(degrees <= 0):
        raise ValueError("All nodes must have positive degree after self-loop handling.")
    D = np.diag(1.0 / np.sqrt(degrees))
    return D @ A @ D


def graph_convolution(adjacency, features, weights):
    A_hat = normalized_adjacency(adjacency)
    X = np.asarray(features, dtype=float)
    W = np.asarray(weights, dtype=float)
    if X.ndim != 2 or W.ndim != 2 or X.shape[0] != A_hat.shape[0] or X.shape[1] != W.shape[0]:
        raise ValueError("Incompatible graph, feature, or weight shapes.")
    return A_hat @ X @ W


if __name__ == "__main__":
    A = np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=float)
    X = np.eye(3)
    W = np.array([[1,0],[0,1],[1,1]], dtype=float)
    print(graph_convolution(A, X, W))
