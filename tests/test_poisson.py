import numpy as np
import pytest

from minipde import (
    uniform_grid,
    assemble_poisson_1d,
    solve_poisson_1d,
    residual_norm,
)


def source(x):
    return np.pi**2 * np.sin(np.pi * x)


def test_boundaries():
    x = uniform_grid(0, 1, 21)

    u = solve_poisson_1d(
        x,
        source,
        1.25,
        -0.5,
    )

    assert u[0] == pytest.approx(1.25)
    assert u[-1] == pytest.approx(-0.5)


def test_order_two():
    errors = []

    for n in (21, 41, 81):
        x = uniform_grid(0, 1, n)

        u = solve_poisson_1d(x, source)

        errors.append(
            np.max(
                np.abs(
                    u - np.sin(np.pi * x)
                )
            )
        )

    orders = np.log2(
        np.array(errors[:-1]) / np.array(errors[1:])
    )

    assert np.all(orders > 1.8)


def test_residual():
    x = uniform_grid(0, 1, 41)

    A, b = assemble_poisson_1d(
        x,
        source,
        sparse=True,
    )

    u = solve_poisson_1d(x, source)

    assert residual_norm(A, u, b) < 1e-12