"""Basic vector operations used throughout machine learning mathematics."""

import numpy as np


def unit_vector(vector: np.ndarray) -> np.ndarray:
    """Return the normalized version of a non-zero vector."""
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if np.isclose(norm, 0.0):
        raise ValueError("The zero vector cannot be normalized.")
    return vector / norm


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """Return the angle between two non-zero vectors in radians."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)
    if np.isclose(u_norm, 0.0) or np.isclose(v_norm, 0.0):
        raise ValueError("The angle is undefined for the zero vector.")

    cosine = np.dot(u, v) / (u_norm * v_norm)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.arccos(cosine))


def are_orthogonal(u: np.ndarray, v: np.ndarray, atol: float = 1e-10) -> bool:
    """Return True when two vectors are orthogonal within a numerical tolerance."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    return bool(np.isclose(np.dot(u, v), 0.0, atol=atol))


if __name__ == "__main__":
    u = np.array([3.0, 4.0])
    v = np.array([1.0, 2.0])

    print("u + v =", u + v)
    print("2u =", 2 * u)
    print("u dot v =", np.dot(u, v))
    print("||u|| =", np.linalg.norm(u))
    print("unit(u) =", unit_vector(u))
    print("angle(u, v) =", angle_between(u, v))
