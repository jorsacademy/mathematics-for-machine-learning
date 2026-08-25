"""A minimal neural-network training example using NumPy and Adam."""

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def binary_cross_entropy(y, p, eps=1e-12):
    p = np.clip(p, eps, 1.0 - eps)
    return -float(np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def train_logistic_neuron(X, y, learning_rate=0.05, steps=1000):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or y.shape != (X.shape[0],):
        raise ValueError("Invalid X/y shapes.")
    w = np.zeros(X.shape[1])
    b = 0.0
    mw = vw = np.zeros_like(w)
    mb = vb = 0.0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        p = sigmoid(X @ w + b)
        error = p - y
        gw = X.T @ error / X.shape[0]
        gb = float(error.mean())
        mw = beta1 * mw + (1 - beta1) * gw
        vw = beta2 * vw + (1 - beta2) * (gw * gw)
        mb = beta1 * mb + (1 - beta1) * gb
        vb = beta2 * vb + (1 - beta2) * (gb * gb)
        w -= learning_rate * (mw / (1 - beta1**t)) / (np.sqrt(vw / (1 - beta2**t)) + eps)
        b -= learning_rate * (mb / (1 - beta1**t)) / (np.sqrt(vb / (1 - beta2**t)) + eps)
    return w, b, binary_cross_entropy(y, sigmoid(X @ w + b))


if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 0, 0, 1], dtype=float)
    print(train_logistic_neuron(X, y))
