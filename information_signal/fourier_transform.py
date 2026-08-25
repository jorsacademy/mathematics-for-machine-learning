"""Discrete Fourier Transform, inverse DFT, and FFT frequency analysis."""

import numpy as np


def dft(x):
    x = np.asarray(x, dtype=complex)
    n = x.size
    k = np.arange(n).reshape(-1, 1)
    j = np.arange(n).reshape(1, -1)
    W = np.exp(-2j * np.pi * k * j / n)
    return W @ x


def idft(X):
    X = np.asarray(X, dtype=complex)
    n = X.size
    k = np.arange(n).reshape(-1, 1)
    j = np.arange(n).reshape(1, -1)
    W = np.exp(2j * np.pi * k * j / n)
    return (W @ X) / n


def dominant_frequency(signal, sample_rate):
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or x.size < 2 or sample_rate <= 0:
        raise ValueError("Invalid signal or sample rate.")
    spectrum = np.fft.rfft(x - x.mean())
    frequencies = np.fft.rfftfreq(x.size, d=1.0 / sample_rate)
    index = np.argmax(np.abs(spectrum[1:])) + 1
    return float(frequencies[index])


if __name__ == "__main__":
    sample_rate = 100.0
    t = np.arange(0.0, 1.0, 1.0 / sample_rate)
    x = np.sin(2.0 * np.pi * 7.0 * t)
    print("DFT reconstruction error:", np.linalg.norm(x - idft(dft(x)).real))
    print("dominant frequency:", dominant_frequency(x, sample_rate))
