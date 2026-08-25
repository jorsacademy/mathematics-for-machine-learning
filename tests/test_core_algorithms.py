"""Basic numerical checks for the educational implementations."""

import numpy as np

from linear_algebra.linear_independence import are_linearly_independent, basis_from_spanning_set
from linear_algebra.matrix_operations import determinant, inverse, multiply
from optimization.adagrad_rmsprop import adagrad, rmsprop
from optimization.adam import adam
from optimization.mini_batch_gradient_descent import fit_linear_regression_minibatch
from optimization.momentum_nesterov import momentum, nesterov
from optimization.stochastic_gradient_descent import fit_linear_regression_sgd


def test_matrix_operations() -> None:
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0], [7.0, 8.0]])
    expected_product = np.array([[19.0, 22.0], [43.0, 50.0]])

    assert np.allclose(multiply(A, B), expected_product)
    assert np.isclose(determinant(A), -2.0)
    assert np.allclose(A @ inverse(A), np.eye(2))


def test_linear_independence_and_basis() -> None:
    independent = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    dependent = [np.array([1.0, 2.0]), np.array([2.0, 4.0])]

    assert are_linearly_independent(independent)
    assert not are_linearly_independent(dependent)
    assert len(basis_from_spanning_set(dependent)) == 1


def test_sgd_linear_regression() -> None:
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = 2.0 * x + 1.0
    w, b = fit_linear_regression_sgd(x, y, learning_rate=0.01, epochs=500)

    assert np.isclose(w, 2.0, atol=1e-3)
    assert np.isclose(b, 1.0, atol=1e-3)


def test_minibatch_linear_regression() -> None:
    x = np.linspace(0.0, 5.0, 200)
    y = 3.0 * x - 2.0
    w, b = fit_linear_regression_minibatch(
        x,
        y,
        learning_rate=0.01,
        epochs=500,
        batch_size=16,
    )

    assert np.isclose(w, 3.0, atol=1e-3)
    assert np.isclose(b, -2.0, atol=1e-3)


def test_momentum_variants() -> None:
    grad = lambda x: 2.0 * (x - 3.0)

    assert np.isclose(momentum(grad, -5.0, steps=300), 3.0, atol=1e-4)
    assert np.isclose(nesterov(grad, -5.0, steps=300), 3.0, atol=1e-4)


def test_adaptive_optimizers() -> None:
    def grad(x: np.ndarray) -> np.ndarray:
        return np.array([2.0 * (x[0] - 3.0), 8.0 * (x[1] + 1.0)])

    start = np.array([8.0, 5.0])
    optimum = np.array([3.0, -1.0])

    assert np.allclose(adagrad(grad, start, learning_rate=0.8, steps=500), optimum, atol=1e-4)
    assert np.allclose(rmsprop(grad, start, learning_rate=0.05, steps=500), optimum, atol=1e-4)
    assert np.allclose(adam(grad, start, learning_rate=0.05, steps=500), optimum, atol=1e-4)
