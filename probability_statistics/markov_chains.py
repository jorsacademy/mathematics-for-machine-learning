"""Finite-state Markov chains and stationary distributions."""

import numpy as np


def validate_transition_matrix(P, tol=1e-10):
    P = np.asarray(P, dtype=float)
    return (
        P.ndim == 2
        and P.shape[0] == P.shape[1]
        and np.all(P >= -tol)
        and np.allclose(P.sum(axis=1), 1.0, atol=tol, rtol=0.0)
    )


def evolve(distribution, P, steps=1):
    if steps < 0 or not isinstance(steps, int):
        raise ValueError("steps must be a non-negative integer.")
    P = np.asarray(P, dtype=float)
    p = np.asarray(distribution, dtype=float)
    if not validate_transition_matrix(P) or p.shape != (P.shape[0],):
        raise ValueError("Invalid distribution or transition matrix.")
    if not np.all(p >= 0) or not np.isclose(p.sum(), 1.0):
        raise ValueError("distribution must sum to 1.")
    return p @ np.linalg.matrix_power(P, steps)


def stationary_distribution(P):
    P = np.asarray(P, dtype=float)
    if not validate_transition_matrix(P):
        raise ValueError("Invalid transition matrix.")
    values, vectors = np.linalg.eig(P.T)
    index = np.argmin(np.abs(values - 1.0))
    v = np.real(vectors[:, index])
    v = v / v.sum()
    if np.any(v < -1e-10):
        v = -v
        v = v / v.sum()
    return np.maximum(v, 0.0) / np.maximum(v, 0.0).sum()


if __name__ == "__main__":
    P = np.array([[0.9, 0.1], [0.4, 0.6]])
    print("after 10 steps:", evolve([1.0, 0.0], P, 10))
    print("stationary distribution:", stationary_distribution(P))
