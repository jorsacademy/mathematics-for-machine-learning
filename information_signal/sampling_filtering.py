"""Sampling, Nyquist condition, and a simple finite impulse response moving-average filter."""

import numpy as np


def nyquist_rate(max_frequency):
    if max_frequency < 0:
        raise ValueError("max_frequency must be non-negative.")
    return 2.0 * max_frequency


def satisfies_nyquist(sample_rate, max_frequency):
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    return sample_rate >= nyquist_rate(max_frequency)


def moving_average_filter(signal, window_size):
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or not isinstance(window_size, int) or not 1 <= window_size <= x.size:
        raise ValueError("Invalid signal or window_size.")
    kernel = np.ones(window_size) / window_size
    return np.convolve(x, kernel, mode="same")


if __name__ == "__main__":
    print("Nyquist rate for 40 Hz:", nyquist_rate(40.0))
    x = np.array([0.0, 1.0, 3.0, 2.0, 1.0, 0.0])
    print("filtered:", moving_average_filter(x, 3))
