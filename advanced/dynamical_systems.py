"""Simple dynamical systems and numerical integration with Euler's method."""

import numpy as np


def euler_integrate(vector_field, x0, dt, steps):
    if dt <= 0 or steps < 0:
        raise ValueError("dt must be positive and steps non-negative.")
    x = np.asarray(x0, dtype=float).copy()
    trajectory = [x.copy()]
    for _ in range(steps):
        dx = np.asarray(vector_field(x), dtype=float)
        if dx.shape != x.shape:
            raise ValueError("vector_field output shape mismatch.")
        x = x + dt * dx
        trajectory.append(x.copy())
    return np.asarray(trajectory)


def logistic_field(r=1.0, carrying_capacity=1.0):
    if r <= 0 or carrying_capacity <= 0:
        raise ValueError("r and carrying_capacity must be positive.")
    return lambda x: r * x * (1.0 - x / carrying_capacity)


if __name__ == "__main__":
    trajectory = euler_integrate(logistic_field(r=2.0, carrying_capacity=1.0), [0.1], dt=0.01, steps=500)
    print("final state:", trajectory[-1])
