"""Proximal algorithms for L1-regularized least squares."""

from __future__ import annotations

import numpy as np


def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    if threshold < 0:
        raise ValueError("threshold must be nonnegative")
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def lasso_objective(X, y, w, lam):
    residual = X @ w - y
    return float(0.5 * np.mean(residual ** 2) + lam * np.linalg.norm(w, 1))


def proximal_gradient(X, y, w0, lam, lr, steps=200):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w0, dtype=float).copy()
    history = []
    for _ in range(steps):
        grad = X.T @ (X @ w - y) / X.shape[0]
        w = soft_threshold(w - lr * grad, lr * lam)
        history.append(lasso_objective(X, y, w, lam))
    return w, history


def fista(X, y, w0, lam, lr, steps=200):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w0, dtype=float).copy()
    z = w.copy()
    t = 1.0
    history = []
    for _ in range(steps):
        grad = X.T @ (X @ z - y) / X.shape[0]
        w_new = soft_threshold(z - lr * grad, lr * lam)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = w_new + ((t - 1.0) / t_new) * (w_new - w)
        w, t = w_new, t_new
        history.append(lasso_objective(X, y, w, lam))
    return w, history