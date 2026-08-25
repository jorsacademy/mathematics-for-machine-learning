"""Basic probability calculations and simulation."""

import numpy as np


def conditional_probability(p_a_and_b, p_b):
    if not 0.0 <= p_a_and_b <= 1.0 or not 0.0 < p_b <= 1.0:
        raise ValueError("Probabilities must be valid and P(B) must be positive.")
    if p_a_and_b > p_b:
        raise ValueError("P(A and B) cannot exceed P(B).")
    return p_a_and_b / p_b


def bayes_theorem(p_b_given_a, p_a, p_b):
    if not all(0.0 <= p <= 1.0 for p in (p_b_given_a, p_a)) or not 0.0 < p_b <= 1.0:
        raise ValueError("Invalid probability.")
    return p_b_given_a * p_a / p_b


def simulate_coin_flips(n=10000, p_heads=0.5, seed=42):
    if n <= 0 or not 0.0 <= p_heads <= 1.0:
        raise ValueError("Invalid simulation parameters.")
    rng = np.random.default_rng(seed)
    samples = rng.binomial(1, p_heads, size=n)
    return float(samples.mean())


if __name__ == "__main__":
    print("estimated P(heads):", simulate_coin_flips())
