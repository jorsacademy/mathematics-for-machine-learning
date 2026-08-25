"""Random walks on finite graphs."""

import numpy as np


def transition_matrix(adjacency):
    A = np.asarray(adjacency, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or np.any(A < 0):
        raise ValueError("adjacency must be a non-negative square matrix.")
    degrees = A.sum(axis=1)
    if np.any(degrees == 0):
        raise ValueError("This simple random walk does not support isolated vertices.")
    return A / degrees[:, None]


def random_walk_distribution(adjacency, start, steps):
    P = transition_matrix(adjacency)
    n = P.shape[0]
    if not 0 <= start < n or steps < 0:
        raise ValueError("Invalid start node or number of steps.")
    p = np.zeros(n)
    p[start] = 1.0
    return p @ np.linalg.matrix_power(P, steps)


if __name__ == "__main__":
    A = np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=float)
    print(random_walk_distribution(A, start=0, steps=5))
