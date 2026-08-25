"""Covariance and correlation with explicit sample/population conventions."""

import numpy as np


def covariance(x, y, sample=True):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("x and y must be one-dimensional arrays of equal length.")
    ddof = 1 if sample else 0
    if x.size <= ddof:
        raise ValueError("Not enough observations.")
    return float(np.sum((x - x.mean()) * (y - y.mean())) / (x.size - ddof))


def correlation(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size or x.size < 2:
        raise ValueError("Need equally sized arrays with at least two observations.")
    sx, sy = x.std(ddof=1), y.std(ddof=1)
    if sx == 0.0 or sy == 0.0:
        raise ValueError("Correlation is undefined for a constant variable.")
    return covariance(x, y, sample=True) / (sx * sy)


if __name__ == "__main__":
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 5, 8, 10]
    print("sample covariance:", covariance(x, y))
    print("correlation:", correlation(x, y))
