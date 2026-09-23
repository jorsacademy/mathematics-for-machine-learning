"""Compact BFGS implementation with Armijo backtracking."""

from __future__ import annotations

from typing import Callable

import numpy as np


def _armijo(value_fn, grad_fn, x, direction, c1=1e-4, shrink=0.5):
    alpha = 1.0
    fx = float(value_fn(x))
    gx = np.asarray(grad_fn(x), dtype=float)
    slope = float(gx @ direction)
    if slope >= 0:
        return 0.0
    while float(value_fn(x + alpha * direction)) > fx + c1 * alpha * slope:
        alpha *= shrink
        if alpha < 1e-12:
            break
    return alpha


def bfgs(
    value_fn: Callable[[np.ndarray], float],
    grad_fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    steps: int = 100,
    tol: float = 1e-8,
):
    x = np.asarray(x0, dtype=float).copy()
    n = x.size
    H = np.eye(n)
    history = []
    for _ in range(steps):
        g = np.asarray(grad_fn(x), dtype=float)
        history.append(float(value_fn(x)))
        if np.linalg.norm(g) <= tol:
            break
        p = -H @ g
        alpha = _armijo(value_fn, grad_fn, x, p)
        if alpha == 0.0:
            p = -g
            alpha = _armijo(value_fn, grad_fn, x, p)
        s = alpha * p
        x_new = x + s
        y = np.asarray(grad_fn(x_new), dtype=float) - g
        ys = float(y @ s)
        if ys > 1e-12:
            rho = 1.0 / ys
            I = np.eye(n)
            V = I - rho * np.outer(s, y)
            H = V @ H @ V.T + rho * np.outer(s, s)
        x = x_new
    return x, history