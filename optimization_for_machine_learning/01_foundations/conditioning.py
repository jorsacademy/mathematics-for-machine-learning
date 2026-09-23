"""Conditioning utilities for smooth quadratic objectives."""

from __future__ import annotations

import numpy as np


def make_spd_quadratic(dim: int, condition_number: float, seed: int = 0):
    """Return Q, b for 0.5*x.T@Q@x - b.T@x with controlled conditioning."""
    if dim < 2:
        raise ValueError("dim must be at least 2")
    if condition_number < 1.0:
        raise ValueError("condition_number must be >= 1")
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(dim, dim))
    q, _ = np.linalg.qr(a)
    eigenvalues = np.geomspace(1.0, condition_number, dim)
    Q = q @ np.diag(eigenvalues) @ q.T
    b = rng.normal(size=dim)
    return Q, b


def quadratic_value(Q: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    return float(0.5 * x @ Q @ x - b @ x)


def quadratic_gradient(Q: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    return Q @ x - b


def spectral_constants(Q: np.ndarray) -> tuple[float, float, float]:
    """Return strong-convexity mu, smoothness L, and spectral condition number."""
    eig = np.linalg.eigvalsh(Q)
    mu = float(eig[0])
    L = float(eig[-1])
    if mu <= 0:
        raise ValueError("Q must be positive definite")
    return mu, L, L / mu


def optimal_fixed_step(Q: np.ndarray) -> float:
    """Best constant GD step for an SPD quadratic under the spectral model."""
    mu, L, _ = spectral_constants(Q)
    return 2.0 / (L + mu)