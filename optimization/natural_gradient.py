"""Natural gradient for a Bernoulli model parameterized by its probability."""

import numpy as np


def bernoulli_fisher(p):
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie strictly between 0 and 1.")
    return 1.0 / (p * (1.0 - p))


def natural_gradient_step(p, euclidean_gradient, learning_rate=0.1, eps=1e-8):
    """Take a natural-gradient descent step in probability coordinates."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    fisher = bernoulli_fisher(p)
    natural_gradient = euclidean_gradient / fisher
    updated = p - learning_rate * natural_gradient
    return float(np.clip(updated, eps, 1.0 - eps))


if __name__ == "__main__":
    p = 0.25
    # Example Euclidean derivative dJ/dp.
    p = natural_gradient_step(p, euclidean_gradient=-2.0, learning_rate=0.1)
    print("updated probability:", p)
