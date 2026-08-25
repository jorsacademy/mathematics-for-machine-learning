"""Projected gradient descent for a simple constrained convex problem."""

import numpy as np


def project_probability_simplex(v):
    """Euclidean projection onto {x >= 0, sum(x) = 1}."""
    v = np.asarray(v, dtype=float)
    if v.ndim != 1 or v.size == 0:
        raise ValueError("v must be a non-empty vector.")
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho_candidates = np.nonzero(u - cssv / np.arange(1, v.size + 1) > 0)[0]
    if rho_candidates.size == 0:
        raise RuntimeError("Projection failed.")
    rho = rho_candidates[-1]
    theta = cssv[rho] / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def projected_gradient_descent(gradient, x0, learning_rate=0.1, steps=100):
    x = project_probability_simplex(x0)
    for _ in range(steps):
        g = np.asarray(gradient(x), dtype=float)
        if g.shape != x.shape:
            raise ValueError("Gradient shape mismatch.")
        x = project_probability_simplex(x - learning_rate * g)
    return x


if __name__ == "__main__":
    target = np.array([0.7, 0.2, 0.1])
    gradient = lambda x: 2.0 * (x - target)
    print(projected_gradient_descent(gradient, [1.0, 1.0, 1.0]))
