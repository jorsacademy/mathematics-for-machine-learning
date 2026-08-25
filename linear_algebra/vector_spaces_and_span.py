"""Vector spaces, subspaces, linear combinations, and span."""

import numpy as np


def in_span(vectors, target, tol=1e-10):
    """Return whether target belongs to the column span of vectors."""
    A = np.asarray(vectors, dtype=float).T
    b = np.asarray(target, dtype=float)
    if A.ndim != 2 or b.ndim != 1 or A.shape[0] != b.size:
        raise ValueError("Incompatible dimensions.")
    coefficients, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    residual = np.linalg.norm(A @ coefficients - b)
    return residual <= tol, coefficients


def is_subspace_under_linear_equations(A, vectors, tol=1e-10):
    """Check whether sample vectors satisfy the homogeneous system A x = 0."""
    A = np.asarray(A, dtype=float)
    X = np.asarray(vectors, dtype=float)
    return bool(np.all(np.linalg.norm(X @ A.T, axis=1) <= tol))


if __name__ == "__main__":
    generators = np.array([[1.0, 2.0], [2.0, 3.0]])
    target = np.array([3.0, 5.0])
    belongs, coefficients = in_span(generators, target)
    print("belongs to span:", belongs)
    print("coefficients:", coefficients)
