"""Maximum likelihood estimation for simple distributions."""

import numpy as np


def bernoulli_mle(samples):
    x = np.asarray(samples, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isin(x, [0.0, 1.0])):
        raise ValueError("Bernoulli samples must be a non-empty binary vector.")
    return float(x.mean())


def normal_mle(samples):
    x = np.asarray(samples, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("samples must be a non-empty vector.")
    mean = float(x.mean())
    variance = float(np.mean((x - mean) ** 2))
    return mean, variance


if __name__ == "__main__":
    data = np.array([1, 0, 1, 1, 0, 1], dtype=float)
    print("Bernoulli p MLE:", bernoulli_mle(data))
    normal_data = np.array([1.2, 0.7, 1.6, 1.1, 0.9])
    print("Normal mean/variance MLE:", normal_mle(normal_data))
