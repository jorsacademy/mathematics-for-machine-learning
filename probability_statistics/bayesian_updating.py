"""Bayesian updating for the Beta-Bernoulli conjugate model."""

import numpy as np


def beta_bernoulli_posterior(alpha, beta, observations):
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive.")
    x = np.asarray(observations, dtype=int)
    if x.ndim != 1 or not np.all(np.isin(x, [0, 1])):
        raise ValueError("observations must be a binary vector.")
    successes = int(x.sum())
    failures = int(x.size - successes)
    return alpha + successes, beta + failures


def beta_mean(alpha, beta):
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive.")
    return alpha / (alpha + beta)


if __name__ == "__main__":
    posterior = beta_bernoulli_posterior(2.0, 2.0, [1, 0, 1, 1, 1])
    print("posterior alpha, beta:", posterior)
    print("posterior mean:", beta_mean(*posterior))
