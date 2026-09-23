"""Transparent first-order optimizers for educational experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Array = np.ndarray


@dataclass
class OptimizationTrace:
    x: Array
    values: list[float]
    grad_norms: list[float]


def gradient_descent(
    value_fn: Callable[[Array], float],
    grad_fn: Callable[[Array], Array],
    x0: Array,
    lr: float,
    steps: int,
) -> OptimizationTrace:
    if lr <= 0 or steps < 1:
        raise ValueError("lr must be positive and steps >= 1")
    x = np.asarray(x0, dtype=float).copy()
    values, grad_norms = [], []
    for _ in range(steps):
        g = np.asarray(grad_fn(x), dtype=float)
        values.append(float(value_fn(x)))
        grad_norms.append(float(np.linalg.norm(g)))
        x -= lr * g
    return OptimizationTrace(x=x, values=values, grad_norms=grad_norms)


def momentum(
    value_fn: Callable[[Array], float],
    grad_fn: Callable[[Array], Array],
    x0: Array,
    lr: float,
    beta: float = 0.9,
    steps: int = 100,
) -> OptimizationTrace:
    if not 0 <= beta < 1:
        raise ValueError("beta must be in [0, 1)")
    x = np.asarray(x0, dtype=float).copy()
    velocity = np.zeros_like(x)
    values, grad_norms = [], []
    for _ in range(steps):
        g = np.asarray(grad_fn(x), dtype=float)
        values.append(float(value_fn(x)))
        grad_norms.append(float(np.linalg.norm(g)))
        velocity = beta * velocity + g
        x -= lr * velocity
    return OptimizationTrace(x=x, values=values, grad_norms=grad_norms)


def nesterov(
    value_fn: Callable[[Array], float],
    grad_fn: Callable[[Array], Array],
    x0: Array,
    lr: float,
    beta: float = 0.9,
    steps: int = 100,
) -> OptimizationTrace:
    if not 0 <= beta < 1:
        raise ValueError("beta must be in [0, 1)")
    x = np.asarray(x0, dtype=float).copy()
    velocity = np.zeros_like(x)
    values, grad_norms = [], []
    for _ in range(steps):
        lookahead = x - lr * beta * velocity
        g = np.asarray(grad_fn(lookahead), dtype=float)
        values.append(float(value_fn(x)))
        grad_norms.append(float(np.linalg.norm(g)))
        velocity = beta * velocity + g
        x -= lr * velocity
    return OptimizationTrace(x=x, values=values, grad_norms=grad_norms)