"""Symbolic differentiation examples for core calculus rules."""

from __future__ import annotations

import sympy as sp


x = sp.symbols("x", real=True)


def derivative(expr: sp.Expr) -> sp.Expr:
    """Return d(expr)/dx."""
    return sp.simplify(sp.diff(expr, x))


def examples() -> dict[str, sp.Expr]:
    """Return verified examples for common differentiation rules."""
    f = x**3 - 4 * x + 5
    product = x**2 * sp.sin(x)
    quotient = x / (x**2 + 1)
    exponential = sp.exp(x**2)

    return {
        "polynomial": derivative(f),
        "product_rule": derivative(product),
        "quotient_rule": derivative(quotient),
        "chain_rule": derivative(exponential),
    }


if __name__ == "__main__":
    for name, result in examples().items():
        print(f"{name}: {result}")
