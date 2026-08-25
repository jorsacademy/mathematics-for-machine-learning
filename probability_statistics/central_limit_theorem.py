"""Simulation of the Central Limit Theorem using a non-normal population."""

from __future__ import annotations

import numpy as np


def sample_means(
    sample_size: int = 30,
    n_samples: int = 10_000,
    seed: int = 42,
) -> np.ndarray:
    """Return sample means from an exponential population.

    The exponential population is deliberately skewed, making the emergence of
    an approximately normal sampling distribution easier to observe.
    """
    if sample_size <= 0 or n_samples <= 0:
        raise ValueError("sample_size and n_samples must be positive.")

    rng = np.random.default_rng(seed)
    samples = rng.exponential(scale=1.0, size=(n_samples, sample_size))
    return samples.mean(axis=1)


def theoretical_mean_and_se(sample_size: int) -> tuple[float, float]:
    """Return the theoretical mean and standard error for Exp(scale=1)."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    population_mean = 1.0
    population_std = 1.0
    standard_error = population_std / np.sqrt(sample_size)
    return population_mean, float(standard_error)


if __name__ == "__main__":
    means = sample_means(sample_size=30)
    theoretical_mean, theoretical_se = theoretical_mean_and_se(30)

    print("Empirical mean of sample means:", means.mean())
    print("Theoretical mean:", theoretical_mean)
    print("Empirical standard deviation:", means.std(ddof=1))
    print("Theoretical standard error:", theoretical_se)
