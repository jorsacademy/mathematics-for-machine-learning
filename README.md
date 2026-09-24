# Mathematics for Machine Learning

<!-- portfolio-umbrella:start -->
## Portfolio role

This repository is intentionally maintained as a standalone primary repository in the consolidated Jors Academy portfolio. It is not used as a container for other projects.
<!-- portfolio-umbrella:end -->

Verified educational Python implementations for mathematical topics used in machine learning and AI.

This repository is intended for educational and non-commercial use only.

## Coverage

### Linear algebra
- Vector operations
- Vector spaces and subspaces
- Linear combinations and span
- Linear independence and basis
- Matrix operations and properties
- Eigenvalues and eigenvectors
- Singular Value Decomposition
- Principal Component Analysis
- Vector norms, projections, and Gram-Schmidt orthogonalization
- Linear transformations

### Calculus
- Functions and their properties
- Derivatives and differentiation rules
- Chain rule and backpropagation structure
- Multivariable gradients
- Directional derivatives
- Hessians

### Probability and statistics
- Basic probability concepts
- Common probability distributions
- Expectation and variance
- Covariance and correlation
- Central Limit Theorem
- Law of Large Numbers
- Hypothesis testing
- Maximum likelihood estimation
- Bayesian updating with conjugate priors
- Markov chains
- Monte Carlo estimation

### Optimization
- Optimization fundamentals
- Local and global extrema
- Convex sets and convex functions
- Quadratic programming
- Gradient descent
- Stochastic and mini-batch gradient descent
- Momentum and Nesterov momentum
- AdaGrad and RMSprop
- Adam
- Second-order and quasi-Newton methods
- Natural gradient descent
- Constrained optimization and projection
- Non-convex optimization
- Optimization in neural networks

For a deeper method-centered sequence on conditioning, convergence, stochastic optimization, AdamW, BFGS, proximal methods, deep-learning diagnostics, and controlled optimizer benchmarks, see `optimization_for_machine_learning/`.

### Information theory and signal processing
- Entropy
- Cross-entropy
- Kullback-Leibler divergence
- Mutual information
- Discrete Fourier Transform and inverse DFT
- FFT-based frequency analysis
- Sampling and the Nyquist condition
- Simple digital filtering

### Advanced mathematical ML
- Tensor operations
- Graph Laplacians
- Spectral graph embeddings
- Random walks on graphs
- Kernel functions and Gram matrices
- Discrete optimal transport
- Dynamical systems and Euler integration
- A minimal graph convolution layer

## Scope decision

The repository intentionally stops short of adding code merely to attach an implementation to every theoretical topic. Subjects such as measure theory, category theory, algebraic topology, differential forms, Lie theory, and functional analysis are mathematically important, but small isolated code examples can be more misleading than educational. They are better developed as rigorous lecture notes before adding computational demonstrations.

The current codebase therefore covers the parts of the curriculum where numerical experiments materially improve mathematical understanding.

The new `optimization_for_machine_learning/` track keeps this principle: it deepens optimization where numerical experiments expose convergence behavior, optimizer trade-offs, and diagnostics, while linking rather than duplicating dedicated repositories for Bayesian optimization, differentiable optimization, predict-then-optimize, neural combinatorial optimization, and learning for solvers.

## Design principles

- Mathematical correctness before convenience.
- Explicit formulas and transparent implementations.
- Numerical safeguards for invalid shapes, zero denominators, singular systems, and probability-domain errors where appropriate.
- NumPy and SciPy preferred over opaque framework abstractions.
- Code is included only when it provides a useful computational interpretation.

## Installation

```bash
python -m pip install -r requirements.txt
```

For the deeper optimization track:

```bash
python -m pip install -r optimization_for_machine_learning/requirements.txt
```

## Testing

```bash
pytest
pytest -q optimization_for_machine_learning/tests
```

## License

Commercial use is not permitted. See `LICENSE` for the repository terms.
