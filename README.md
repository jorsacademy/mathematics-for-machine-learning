# Mathematics for Machine Learning

A collection of reviewed Python implementations that accompany core mathematical topics used in machine learning and AI.

The emphasis is mathematical correctness, transparent implementations, input validation, and educational clarity. Framework-specific abstractions are avoided when they would hide the mathematics.

## Topics and code

### Linear algebra

- `linear_algebra/vector_operations.py` — vector addition, dot product, norms, normalization, and angles
- `linear_algebra/matrix_operations.py` — matrix addition, multiplication, transpose, determinant, inverse, and trace
- `linear_algebra/linear_independence.py` — rank-based linear-independence checks and basis extraction

### Calculus

- `calculus/derivatives.py` — symbolic examples of common differentiation rules
- `calculus/chain_rule.py` — scalar chain rule and a small explicit backpropagation example

### Probability and statistics

- `probability_statistics/central_limit_theorem.py` — CLT simulation using a skewed exponential population
- `probability_statistics/law_of_large_numbers.py` — LLN simulation using Bernoulli trials

### Optimization

- `optimization/stochastic_gradient_descent.py` — SGD for linear regression
- `optimization/mini_batch_gradient_descent.py` — mini-batch gradient descent for linear regression
- `optimization/momentum_nesterov.py` — classical momentum and Nesterov Accelerated Gradient
- `optimization/adagrad_rmsprop.py` — AdaGrad and RMSprop from first principles
- `optimization/adam.py` — Adam with first/second moments and bias correction
- `optimization/quadratic_programming.py` — a checked convex quadratic-programming example

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Verification

The numerical examples were checked against their mathematical definitions. A small regression test suite is included in `tests/test_core_algorithms.py`.

You can run the tests with Python's installed test runner of your choice after installing the dependencies. The test file uses ordinary `assert` statements and is compatible with pytest.

## Design principles

- Prefer explicit mathematics over opaque library calls.
- Validate dimensions, shapes, domains, and optimizer hyperparameters where appropriate.
- Use deterministic random-number generators in stochastic demonstrations.
- Keep examples small enough to inspect by hand.
- Do not include code merely for the sake of including code; an example is added only when it clarifies the mathematics.

## License

This repository is intended for educational and non-commercial use only. Commercial use is not permitted. See `LICENSE` for the complete terms.
