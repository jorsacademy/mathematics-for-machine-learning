"""Illustration of local versus global extrema for a non-convex polynomial."""

import numpy as np


def objective(x):
    return x**4 - 4.0 * x**2 + 2.0


def derivative(x):
    return 4.0 * x**3 - 8.0 * x


def second_derivative(x):
    return 12.0 * x**2 - 8.0


def stationary_points():
    points = np.array([-np.sqrt(2.0), 0.0, np.sqrt(2.0)])
    classification = []
    for x in points:
        curvature = second_derivative(x)
        kind = "local minimum" if curvature > 0 else "local maximum" if curvature < 0 else "inconclusive"
        classification.append((float(x), float(objective(x)), kind))
    return classification


if __name__ == "__main__":
    for item in stationary_points():
        print(item)
