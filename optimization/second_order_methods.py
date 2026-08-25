"""Newton and BFGS examples for second-order and quasi-Newton optimization."""

import numpy as np
from scipy.optimize import minimize


def newton_optimize(gradient, hessian, x0, steps=20, tolerance=1e-10):
    x = np.asarray(x0, dtype=float).copy()
    for _ in range(steps):
        g = np.asarray(gradient(x), dtype=float)
        if np.linalg.norm(g) <= tolerance:
            break
        H = np.asarray(hessian(x), dtype=float)
        if H.shape != (x.size, x.size):
            raise ValueError("Hessian has an invalid shape.")
        # Solve H p = g instead of explicitly computing H^{-1}.
        step = np.linalg.solve(H, g)
        x -= step
    return x


def bfgs_optimize(objective, gradient, x0):
    result = minimize(objective, np.asarray(x0, dtype=float), jac=gradient, method="BFGS")
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


if __name__ == "__main__":
    objective = lambda x: (x[0] - 3.0) ** 2 + 2.0 * (x[1] + 1.0) ** 2
    gradient = lambda x: np.array([2.0 * (x[0] - 3.0), 4.0 * (x[1] + 1.0)])
    hessian = lambda x: np.array([[2.0, 0.0], [0.0, 4.0]])
    print("Newton:", newton_optimize(gradient, hessian, [0.0, 0.0]))
    print("BFGS:", bfgs_optimize(objective, gradient, [0.0, 0.0]))
