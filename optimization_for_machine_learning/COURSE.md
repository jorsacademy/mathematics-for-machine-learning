# Optimization for Machine Learning — Course

40 lectures plus practical benchmark work.

The course is deliberately narrower than a general Optimization for AI survey. Its center of gravity is the optimization of statistical and machine-learning models. AI-for-optimization topics are handled by dedicated companion repositories.

## Module 1 — Foundations

1. Optimization problem formulation for ML
   - objectives, constraints, feasible sets, local/global solutions
2. Convex sets and convex functions
   - Jensen inequality, operations preserving convexity
3. Smoothness and strong convexity
   - Lipschitz gradients and curvature bounds
4. Taylor models and local approximation
   - first- and second-order models
5. Conditioning
   - eigenvalues, condition numbers, ill-conditioning
6. Convergence language
   - linear, sublinear, superlinear, iteration and oracle complexity

## Module 2 — First-Order Methods

7. Gradient descent
   - descent lemma and fixed step sizes
8. Step-size selection
   - backtracking, Armijo and Wolfe conditions
9. Momentum
   - heavy-ball dynamics
10. Nesterov acceleration
    - accelerated convex rates and practical interpretation
11. Learning-rate schedules
    - step, exponential, cosine, warmup
12. Coordinate descent
    - cyclic and randomized variants
13. Preconditioning
    - diagonal scaling and geometry of the parameter space

## Module 3 — Stochastic Optimization

14. Stochastic gradient descent
    - unbiased estimates and noisy descent
15. Mini-batch optimization
    - batch size, throughput, gradient variance
16. Gradient noise
    - signal-to-noise ratio and optimization regimes
17. Variance reduction
    - SAG, SAGA and SVRG concepts
18. Sampling strategies
    - importance sampling and data ordering effects

## Module 4 — Adaptive Optimizers

19. AdaGrad
    - coordinate-wise accumulated scaling
20. RMSProp
    - exponential second-moment tracking
21. Adam
    - bias correction and effective step sizes
22. AdamW
    - decoupled weight decay versus L2 regularization
23. Adaptive optimizer variants
    - AMSGrad, RAdam, AdaBelief, LAMB and LARS

## Module 5 — Second-Order and Curvature Methods

24. Newton's method
    - Hessian systems and local quadratic convergence
25. Modified Newton methods
    - damping and trust regions
26. Quasi-Newton methods
    - secant condition, BFGS and L-BFGS
27. Hessian-vector products
    - matrix-free curvature
28. Conjugate-gradient and truncated Newton methods

## Module 6 — Constrained and Non-Smooth Optimization

29. Lagrange multipliers and KKT conditions
30. Projected gradient methods
31. Mirror descent and Bregman geometry
32. Frank-Wolfe / conditional gradient
33. Proximal gradient and FISTA
    - sparse learning and composite objectives

## Module 7 — Deep Learning Optimization

34. Initialization and conditioning
    - variance propagation and scale
35. Gradient flow
    - vanishing, exploding, clipping
36. Normalization and residual connections
    - optimization effects rather than architecture cataloging
37. Regularization versus optimization
    - weight decay, early stopping, implicit bias
38. Large-batch training
    - scaling rules, sharpness and generalization caveats

## Module 8 — Empirical Optimization Practice

39. Optimization debugging
    - gradient norms, update ratios, loss curves, numerical pathologies
40. Optimizer benchmarking
    - common initialization, work-normalized comparisons, wall-clock time, validation metrics

## Practical sequence

- Lab A: condition number versus gradient-descent convergence
- Lab B: GD versus momentum versus Nesterov
- Lab C: SGD batch-size and gradient-variance experiment
- Lab D: Adam versus AdamW under explicit weight decay
- Lab E: BFGS versus first-order methods on smooth objectives
- Lab F: proximal gradient and FISTA for sparse regression
- Lab G: gradient-flow and update-ratio diagnostics
- Lab H: controlled optimizer benchmark on logistic regression

## Companion repositories

The following topics are valuable but intentionally live outside this course because they represent AI-for-optimization or decision-focused learning rather than core ML training optimization:

- `bayesian-optimization-industrial-engineering`
- `differentiable-optimization-pytorch`
- `predict-then-optimize-production-planning-spo-plus-pytorch`
- `neural-combinatorial-optimization-tsp-attention-model-pytorch`
- `learning-to-branch-mip-gnn-scip-pytorch`
- `reinforcement-learning-job-shop-scheduling-pytorch`

## Assessment idea

- theory exercises: 25%
- implementation exercises: 25%
- benchmark reports: 25%
- final reproducible optimization study: 25%