"""Tensor operations used frequently in machine learning."""

import numpy as np


def tensor_shape(x):
    return np.asarray(x).shape


def tensor_contraction(A, B, axes=1):
    return np.tensordot(np.asarray(A), np.asarray(B), axes=axes)


def batched_matrix_multiply(A, B):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.ndim < 2 or B.ndim < 2 or A.shape[-1] != B.shape[-2]:
        raise ValueError("Incompatible tensor shapes for matrix multiplication.")
    return A @ B


def outer_product(a, b):
    return np.outer(np.asarray(a, dtype=float), np.asarray(b, dtype=float))


if __name__ == "__main__":
    A = np.arange(12.0).reshape(2, 2, 3)
    B = np.arange(12.0).reshape(2, 3, 2)
    print("batched product shape:", batched_matrix_multiply(A, B).shape)
    print("outer product:\n", outer_product([1, 2], [3, 4, 5]))
