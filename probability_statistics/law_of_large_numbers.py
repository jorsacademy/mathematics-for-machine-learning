"""Simulation of the Law of Large Numbers with Bernoulli trials."""

from __future__ import annotations

import numpy as np


def running_mean_of_bernoulli(
    p: float = 0.5,
    n_trials: int = 10_000,
    seed: int = 42,
) -> np.ndarray:
    """Return the running sample mean of Bernoulli(p) observations."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0, 1].")
    if n_trials <= 0:
        raise ValueError("n_trials must be positive.")

    rng = np.random.default_rng(seed)
    observations = rng.binomial(n=1, p=p, size=n_trials)
    cumulative_sum = np.cumsum(observations)
    counts = np.arange(1, n_trials + 1)
    return cumulative_sum / counts


if __name__ == "__main__":
    p = 0.5
    means = running_mean_of_bernoulli(p=p)
    print("Final running mean:", means[-1])
    print("Population mean:", p)
