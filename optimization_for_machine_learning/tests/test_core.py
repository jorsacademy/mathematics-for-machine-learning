from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(relative, name):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


conditioning = load("01_foundations/conditioning.py", "conditioning")
gradient_methods = load("02_gradient_methods/gradient_methods.py", "gradient_methods")
sgd = load("03_stochastic_optimization/sgd.py", "sgd")
adaptive = load("04_adaptive_optimizers/adaptive.py", "adaptive")
proximal = load("06_constrained_and_nonsmooth/proximal.py", "proximal")


def test_condition_number_is_controlled():
    Q, _ = conditioning.make_spd_quadratic(8, 100.0, seed=1)
    _, _, kappa = conditioning.spectral_constants(Q)
    assert np.isclose(kappa, 100.0, rtol=1e-10)


def test_gradient_descent_reduces_spd_quadratic():
    Q, b = conditioning.make_spd_quadratic(6, 20.0, seed=2)
    f = lambda x: conditioning.quadratic_value(Q, b, x)
    g = lambda x: conditioning.quadratic_gradient(Q, b, x)
    trace = gradient_methods.gradient_descent(
        f, g, np.zeros(6), conditioning.optimal_fixed_step(Q), 80
    )
    assert trace.values[-1] < trace.values[0]
    assert np.linalg.norm(g(trace.x)) < 1e-3


def test_minibatch_sgd_learns_least_squares():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, 5))
    true_w = rng.normal(size=5)
    y = X @ true_w + 0.05 * rng.normal(size=400)
    w, hist = sgd.mini_batch_sgd(X, y, np.zeros(5), lr=0.05, batch_size=40, epochs=40, seed=3)
    assert hist[-1] < hist[0]
    assert np.linalg.norm(w - true_w) < 0.1


def test_adamw_optimizes_quadratic():
    target = np.array([1.5, -2.0, 0.25])
    grad = lambda x: x - target
    x = adaptive.adamw(grad, np.zeros(3), lr=0.05, weight_decay=0.0, steps=300)
    assert np.linalg.norm(x - target) < 1e-4


def test_soft_threshold():
    x = np.array([-2.0, -0.5, 0.25, 3.0])
    got = proximal.soft_threshold(x, 0.5)
    expected = np.array([-1.5, 0.0, 0.0, 2.5])
    assert np.allclose(got, expected)