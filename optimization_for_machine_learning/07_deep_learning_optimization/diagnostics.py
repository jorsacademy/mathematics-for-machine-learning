"""Framework-agnostic scalar diagnostics for optimizer monitoring."""

from __future__ import annotations

import numpy as np


def gradient_norm(gradients) -> float:
    arrays = [np.asarray(g, dtype=float).ravel() for g in gradients]
    if not arrays:
        return 0.0
    return float(np.linalg.norm(np.concatenate(arrays)))


def update_to_weight_ratio(weights, updates, eps: float = 1e-12) -> float:
    w = np.concatenate([np.asarray(x, dtype=float).ravel() for x in weights])
    u = np.concatenate([np.asarray(x, dtype=float).ravel() for x in updates])
    return float(np.linalg.norm(u) / max(np.linalg.norm(w), eps))


def all_finite(arrays) -> bool:
    return all(bool(np.all(np.isfinite(np.asarray(x)))) for x in arrays)


def diagnostic_snapshot(weights, gradients, updates) -> dict[str, float | bool]:
    return {
        "gradient_norm": gradient_norm(gradients),
        "update_to_weight_ratio": update_to_weight_ratio(weights, updates),
        "all_finite": all_finite([*weights, *gradients, *updates]),
    }