"""Mini-batch SGD for least-squares experiments."""

from __future__ import annotations

import numpy as np


def least_squares_loss(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    residual = X @ w - y
    return float(0.5 * np.mean(residual ** 2))


def mini_batch_sgd(
    X: np.ndarray,
    y: np.ndarray,
    w0: np.ndarray,
    lr: float = 1e-2,
    batch_size: int = 32,
    epochs: int = 20,
    seed: int = 0,
):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w0, dtype=float).copy()
    n = X.shape[0]
    if X.ndim != 2 or y.shape != (n,) or w.shape != (X.shape[1],):
        raise ValueError("incompatible X, y, w0 shapes")
    if not 1 <= batch_size <= n:
        raise ValueError("batch_size must be between 1 and n")
    rng = np.random.default_rng(seed)
    history = []
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            xb, yb = X[idx], y[idx]
            grad = xb.T @ (xb @ w - yb) / len(idx)
            w -= lr * grad
        history.append(least_squares_loss(X, y, w))
    return w, history


def gradient_noise_scale(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, batch_size: int, seed: int = 0
) -> float:
    """Estimate E||g_batch-g_full||^2 for one sampled batch."""
    n = X.shape[0]
    if not 1 <= batch_size <= n:
        raise ValueError("invalid batch_size")
    full = X.T @ (X @ w - y) / n
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=batch_size, replace=False)
    batch = X[idx].T @ (X[idx] @ w - y[idx]) / batch_size
    return float(np.sum((batch - full) ** 2))