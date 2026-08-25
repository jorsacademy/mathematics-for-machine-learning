"""Linear transformations represented by matrices."""

import numpy as np


def apply_linear_map(A, x):
    A = np.asarray(A, dtype=float)
    x = np.asarray(x, dtype=float)
    if A.ndim != 2 or x.ndim != 1 or A.shape[1] != x.size:
        raise ValueError("Incompatible dimensions.")
    return A @ x


def compose(A, B):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[0]:
        raise ValueError("Incompatible transformation dimensions.")
    return A @ B


def rotation_2d(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


if __name__ == "__main__":
    R = rotation_2d(np.pi / 2)
    print("rotated vector:", apply_linear_map(R, [1.0, 0.0]))
