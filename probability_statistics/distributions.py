"""Common discrete and continuous probability distributions."""

import math
import numpy as np


def bernoulli_pmf(x, p):
    if x not in (0, 1) or not 0.0 <= p <= 1.0:
        return 0.0
    return p if x == 1 else 1.0 - p


def binomial_pmf(k, n, p):
    if not (isinstance(n, int) and n >= 0 and isinstance(k, int) and 0 <= k <= n and 0.0 <= p <= 1.0):
        return 0.0
    return math.comb(n, k) * p**k * (1.0 - p) ** (n - k)


def normal_pdf(x, mean=0.0, std=1.0):
    if std <= 0:
        raise ValueError("std must be positive.")
    z = (np.asarray(x, dtype=float) - mean) / std
    return np.exp(-0.5 * z**2) / (std * np.sqrt(2.0 * np.pi))


if __name__ == "__main__":
    print("P(X=3), X~Binomial(10,0.4):", binomial_pmf(3, 10, 0.4))
    print("standard normal density at 0:", float(normal_pdf(0.0)))
