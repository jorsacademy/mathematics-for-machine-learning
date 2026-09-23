"""Small NumPy implementations of Adam and AdamW."""

from __future__ import annotations

from typing import Callable

import numpy as np


def adam(
    grad_fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    steps: int = 1000,
    weight_decay: float = 0.0,
    decoupled: bool = False,
) -> np.ndarray:
    """Optimize with Adam; set decoupled=True for AdamW-style weight decay."""
    x = np.asarray(x0, dtype=float).copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    for t in range(1, steps + 1):
        g = np.asarray(grad_fn(x), dtype=float)
        if weight_decay and not decoupled:
            g = g + weight_decay * x
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)
        if weight_decay and decoupled:
            x *= 1.0 - lr * weight_decay
        x -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return x


def adamw(
    grad_fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    steps: int = 1000,
) -> np.ndarray:
    return adam(
        grad_fn, x0, lr=lr, steps=steps, weight_decay=weight_decay, decoupled=True
    )