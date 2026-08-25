"""Expectation, variance, covariance matrix, and Monte Carlo estimation."""

import numpy as np


def empirical_expectation(values):
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        raise ValueError("values cannot be empty.")
    return float(x.mean())


def empirical_variance(values, sample=True):
    x = np.asarray(values, dtype=float)
    ddof = 1 if sample else 0
    if x.size <= ddof:
        raise ValueError("Not enough observations.")
    return float(x.var(ddof=ddof))


def covariance_matrix(X, sample=True):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of samples by features.")
    return np.cov(X, rowvar=False, ddof=1 if sample else 0)


def monte_carlo_expectation(function, sampler, n=10000):
    if n <= 0:
        raise ValueError("n must be positive.")
    samples = np.asarray(sampler(n))
    values = np.asarray(function(samples), dtype=float)
    return float(values.mean())


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    estimate = monte_carlo_expectation(lambda x: x**2, lambda n: rng.normal(size=n))
    print("E[X^2] for X~N(0,1), estimate:", estimate)
