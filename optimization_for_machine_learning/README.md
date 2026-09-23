# Optimization for Machine Learning

A method-centered bridge between mathematical optimization foundations and modern machine-learning training.

This module complements the compact `optimization/` examples in the parent repository. It focuses on why optimization algorithms behave differently, how to diagnose failures, and how to compare methods under controlled experiments.

## Scope

- conditioning, smoothness, strong convexity, and convergence rates
- line-search and first-order methods
- stochastic optimization and gradient noise
- adaptive optimizers, including AdamW
- Newton, quasi-Newton, and Hessian-vector ideas
- projected, proximal, and non-smooth methods
- optimization diagnostics for deep learning
- reproducible optimizer benchmarks

## What is intentionally not duplicated

Advanced AI-for-optimization topics already have dedicated repositories in the JORS Academy portfolio. This module links to those repositories instead of reimplementing them:

- Bayesian optimization
- differentiable optimization
- predict-then-optimize / SPO+
- neural combinatorial optimization
- reinforcement learning for scheduling
- GNN-guided optimization and learning-to-branch

See `LEARNING_PATH.md` for the map.

## Layout

    01_foundations/                 conditioning and convergence geometry
    02_gradient_methods/            GD, momentum, Nesterov, line-search ideas
    03_stochastic_optimization/      mini-batch SGD and gradient-noise experiments
    04_adaptive_optimizers/          Adam and AdamW
    05_second_order/                 BFGS and curvature methods
    06_constrained_and_nonsmooth/    proximal gradient and FISTA
    07_deep_learning_optimization/   gradient/update diagnostics
    08_optimizer_benchmarks/         controlled empirical comparisons
    tests/                           numerical regression tests

## Run

From the repository root:

    python optimization_for_machine_learning/08_optimizer_benchmarks/benchmark_logistic.py
    pytest -q optimization_for_machine_learning/tests

## Design principles

1. Compare algorithms on the same objective and initialization.
2. Separate iteration count from actual computational work.
3. Report optimization metrics and generalization metrics separately.
4. Make numerical assumptions explicit.
5. Prefer small transparent implementations before framework abstractions.

## Course

The full course sequence is in `COURSE.md`.