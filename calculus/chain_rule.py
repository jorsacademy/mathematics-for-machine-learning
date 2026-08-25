"""Chain rule examples, including a simple scalar computational graph."""

from __future__ import annotations

import math


def composite_derivative(x: float) -> float:
    """Differentiate y = cos(3x^2 + 5x) using the chain rule."""
    inner = 3.0 * x**2 + 5.0 * x
    d_outer_d_inner = -math.sin(inner)
    d_inner_dx = 6.0 * x + 5.0
    return d_outer_d_inner * d_inner_dx


def simple_backprop(x: float, w: float, b: float, target: float) -> dict[str, float]:
    """Backpropagate through z = wx+b, a = sigmoid(z), L = 0.5(a-y)^2.

    This is intentionally scalar so every chain-rule factor is explicit.
    """
    z = w * x + b
    a = 1.0 / (1.0 + math.exp(-z))
    loss = 0.5 * (a - target) ** 2

    dloss_da = a - target
    da_dz = a * (1.0 - a)
    dz_dw = x
    dz_db = 1.0

    dloss_dw = dloss_da * da_dz * dz_dw
    dloss_db = dloss_da * da_dz * dz_db

    return {
        "z": z,
        "activation": a,
        "loss": loss,
        "dL_dw": dloss_dw,
        "dL_db": dloss_db,
    }


if __name__ == "__main__":
    print("d/dx cos(3x^2+5x) at x=1:", composite_derivative(1.0))
    print(simple_backprop(x=2.0, w=0.5, b=-0.1, target=1.0))
