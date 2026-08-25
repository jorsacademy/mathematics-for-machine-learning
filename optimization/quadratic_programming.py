"""A checked convex quadratic-programming example using CVXPY."""

from __future__ import annotations

import cvxpy as cp
import numpy as np


def solve_example() -> tuple[np.ndarray, float]:
    """Solve a small convex QP and return the optimizer and objective value.

    Problem
    -------
    minimize    x1^2 + x2^2 - 4*x1 - 6*x2
    subject to  x1 + x2 <= 1
                x1 >= 0
                x2 >= 0

    The Hessian is 2I, which is positive definite, so the objective is strictly
    convex. The feasible set is convex, therefore the optimizer is unique.
    """
    Q = np.array([[2.0, 0.0], [0.0, 2.0]])
    c = np.array([-4.0, -6.0])

    x = cp.Variable(2)
    objective = cp.Minimize(0.5 * cp.quad_form(x, Q) + c @ x)
    constraints = [x[0] + x[1] <= 1.0, x >= 0.0]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"QP solver failed with status: {problem.status}")
    if x.value is None or problem.value is None:
        raise RuntimeError("QP solver returned no numerical solution.")

    solution = np.asarray(x.value, dtype=float)
    value = float(problem.value)
    return solution, value


if __name__ == "__main__":
    solution, value = solve_example()

    print("Optimal x:", solution)
    print("Optimal objective:", value)

    # Analytical solution: x* = [0, 1], f(x*) = -5.
    print("Matches analytical optimizer:", np.allclose(solution, [0.0, 1.0], atol=1e-5))
    print("Matches analytical objective:", np.isclose(value, -5.0, atol=1e-5))
