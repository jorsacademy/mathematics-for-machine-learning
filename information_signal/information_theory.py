"""Entropy, cross-entropy, KL divergence, and mutual information."""

import numpy as np


def _probability_vector(p):
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or p.size == 0 or np.any(p < 0) or not np.isclose(p.sum(), 1.0):
        raise ValueError("Input must be a probability vector summing to 1.")
    return p


def entropy(p, base=2.0):
    p = _probability_vector(p)
    nz = p > 0
    return float(-np.sum(p[nz] * (np.log(p[nz]) / np.log(base))))


def cross_entropy(p, q, base=2.0):
    p = _probability_vector(p)
    q = _probability_vector(q)
    if p.shape != q.shape or np.any((p > 0) & (q == 0)):
        return float("inf")
    nz = p > 0
    return float(-np.sum(p[nz] * (np.log(q[nz]) / np.log(base))))


def kl_divergence(p, q, base=2.0):
    return cross_entropy(p, q, base=base) - entropy(p, base=base)


def mutual_information(joint, base=2.0):
    P = np.asarray(joint, dtype=float)
    if P.ndim != 2 or np.any(P < 0) or not np.isclose(P.sum(), 1.0):
        raise ValueError("joint must be a 2D probability table summing to 1.")
    px = P.sum(axis=1, keepdims=True)
    py = P.sum(axis=0, keepdims=True)
    product = px @ py
    mask = P > 0
    return float(np.sum(P[mask] * (np.log(P[mask] / product[mask]) / np.log(base))))


if __name__ == "__main__":
    p = np.array([0.5, 0.25, 0.25])
    q = np.array([0.4, 0.3, 0.3])
    print("entropy:", entropy(p))
    print("cross entropy:", cross_entropy(p, q))
    print("KL divergence:", kl_divergence(p, q))
