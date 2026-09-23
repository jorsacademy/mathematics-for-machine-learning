"""Controlled benchmark: GD, momentum, AdamW and L-BFGS on logistic regression."""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize


def make_data(n=1200, d=20, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    true_w = rng.normal(size=d)
    logits = X @ true_w + 0.35 * rng.normal(size=n)
    y = (logits > 0).astype(float)
    split = int(0.8 * n)
    return X[:split], y[:split], X[split:], y[split:]


def loss_grad(X, y, w, l2=1e-3):
    z = X @ w
    loss = np.mean(np.logaddexp(0.0, z) - y * z) + 0.5 * l2 * (w @ w)
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
    grad = X.T @ (p - y) / X.shape[0] + l2 * w
    return float(loss), grad


def accuracy(X, y, w):
    return float(np.mean((X @ w >= 0) == y))


def run_gd(X, y, w0, steps=300, lr=0.2):
    w = w0.copy()
    for _ in range(steps):
        _, g = loss_grad(X, y, w)
        w -= lr * g
    return w


def run_momentum(X, y, w0, steps=300, lr=0.1, beta=0.9):
    w = w0.copy()
    v = np.zeros_like(w)
    for _ in range(steps):
        _, g = loss_grad(X, y, w)
        v = beta * v + g
        w -= lr * v
    return w


def run_adamw(X, y, w0, steps=300, lr=0.03, wd=1e-3):
    w = w0.copy()
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        _, g = loss_grad(X, y, w, l2=0.0)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** t)
        vh = v / (1 - b2 ** t)
        w *= 1 - lr * wd
        w -= lr * mh / (np.sqrt(vh) + eps)
    return w


def benchmark():
    Xtr, ytr, Xva, yva = make_data()
    w0 = np.zeros(Xtr.shape[1])
    methods = {
        "GD": lambda: run_gd(Xtr, ytr, w0),
        "Momentum": lambda: run_momentum(Xtr, ytr, w0),
        "AdamW": lambda: run_adamw(Xtr, ytr, w0),
        "L-BFGS": lambda: minimize(
            lambda w: loss_grad(Xtr, ytr, w),
            w0,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 300},
        ).x,
    }
    rows = []
    for name, fn in methods.items():
        start = time.perf_counter()
        w = fn()
        elapsed = time.perf_counter() - start
        train_loss, grad = loss_grad(Xtr, ytr, w)
        val_loss, _ = loss_grad(Xva, yva, w)
        rows.append(
            (name, train_loss, val_loss, np.linalg.norm(grad), accuracy(Xva, yva, w), elapsed)
        )
    print("method      train_loss  val_loss   grad_norm  val_acc  seconds")
    for row in rows:
        print(f"{row[0]:10s} {row[1]:10.5f} {row[2]:9.5f} {row[3]:10.4e} {row[4]:8.3f} {row[5]:8.4f}")


if __name__ == "__main__":
    benchmark()