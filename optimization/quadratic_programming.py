"""A small convex quadratic programming example using CVXPY."""

import cvxpy as cp
import numpy as np


# Minimize: x1^2 + x2^2 - 4*x1 - 6*x2
# Subject to: x1 + x2 <= 1, x1 >= 0, x2 >= 0
Q = np.array([[2.0, 0.0], [0.0, 2.0]])
c = np.array([-4.0, -6.0])

x = cp.Variable(2)
objective = cp.Minimize(0.5 * cp.quad_form(x, Q) + c @ x)
constraints = [x[0] + x[1] <= 1, x >= 0]
problem = cp.Problem(objective, constraints)
problem.solve()

print("status:", problem.status)
print("optimal x:", x.value)
print("optimal objective:", problem.value)

# The analytical optimum is x = [0, 1] with objective value -5.
