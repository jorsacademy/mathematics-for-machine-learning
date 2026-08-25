"""Small test suite for selected mathematical invariants in advanced examples."""

import numpy as np

from linear_algebra.eigenvalues_eigenvectors import symmetric_eigendecomposition, verify_eigenpairs
from linear_algebra.svd import rank_k_approximation
from probability_statistics.information_theory import entropy, kl_divergence
from advanced.graph_spectral import graph_laplacian
from advanced.kernels import gram_matrix, rbf_kernel


def test_symmetric_eigendecomposition():
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    values, vectors = symmetric_eigendecomposition(A)
    assert verify_eigenpairs(A, values, vectors)


def test_rank_one_approximation_shape():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    approximation = rank_k_approximation(A, 1)
    assert approximation.shape == A.shape
    assert np.linalg.matrix_rank(approximation, tol=1e-10) <= 1


def test_information_theory_nonnegativity():
    p = np.array([0.5, 0.25, 0.25])
    q = np.array([0.4, 0.3, 0.3])
    assert entropy(p) >= 0.0
    assert kl_divergence(p, q) >= -1e-12


def test_graph_laplacian_psd():
    A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    L = graph_laplacian(A)
    assert np.all(np.linalg.eigvalsh(L) >= -1e-12)


def test_rbf_gram_matrix_psd():
    X = np.array([[0.0], [1.0], [2.0]])
    K = gram_matrix(X, lambda a, b: rbf_kernel(a, b, gamma=0.5))
    assert np.all(np.linalg.eigvalsh(K) >= -1e-12)
