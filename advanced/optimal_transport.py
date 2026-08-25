"""Discrete optimal transport as a linear program."""

import numpy as np
from scipy.optimize import linprog


def optimal_transport(source, target, cost):
    a = np.asarray(source, dtype=float)
    b = np.asarray(target, dtype=float)
    C = np.asarray(cost, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or C.shape != (a.size, b.size):
        raise ValueError("Invalid source, target, or cost dimensions.")
    if np.any(a < 0) or np.any(b < 0) or not np.isclose(a.sum(), b.sum()):
        raise ValueError("Source and target masses must be non-negative and have equal total mass.")

    n, m = C.shape
    c = C.ravel()
    A_eq = []
    b_eq = []

    for i in range(n):
        row = np.zeros((n, m))
        row[i, :] = 1.0
        A_eq.append(row.ravel())
        b_eq.append(a[i])

    for j in range(m):
        col = np.zeros((n, m))
        col[:, j] = 1.0
        A_eq.append(col.ravel())
        b_eq.append(b[j])

    result = linprog(c, A_eq=np.asarray(A_eq), b_eq=np.asarray(b_eq), bounds=(0.0, None), method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    return result.x.reshape(n, m), float(result.fun)


if __name__ == "__main__":
    a = np.array([0.5, 0.5])
    b = np.array([0.25, 0.75])
    C = np.array([[0.0, 1.0], [1.0, 0.0]])
    plan, cost = optimal_transport(a, b, C)
    print("transport plan:\n", plan)
    print("transport cost:", cost)
