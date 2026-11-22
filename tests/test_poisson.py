import numpy as np
import pytest

from minipde.mesh import uniform_grid
from minipde.poisson import assemble_poisson_1d, solve_poisson_1d
from minipde.verification import observed_order


def f_sin(x):
    return np.pi**2 * np.sin(np.pi * x)


def test_assemble_shape_and_boundary_rows():
    x = uniform_grid(0.0, 1.0, 6)
    matrix, rhs = assemble_poisson_1d(x, f_sin, ua=2.0, ub=-1.0)
    assert matrix.shape == (6, 6)
    assert rhs.shape == (6,)
    assert matrix[0, 0] == 1.0
    assert matrix[-1, -1] == 1.0
    assert np.count_nonzero(matrix[0]) == 1
    assert np.count_nonzero(matrix[-1]) == 1
    assert rhs[0] == 2.0
    assert rhs[-1] == -1.0


def test_interior_stencil():
    x = uniform_grid(0.0, 1.0, 5)
    h = x[1] - x[0]
    matrix, _ = assemble_poisson_1d(x, f_sin)
    np.testing.assert_allclose(matrix[2, 1:4] * h**2, [-1.0, 2.0, -1.0])


def test_quadratic_is_exact():
    # u = x(1-x)  ->  -u'' = 2 ; el esquema centrado es exacto para cuadráticas
    x = uniform_grid(0.0, 1.0, 11)
    u, _ = solve_poisson_1d(x, lambda s: 2.0 * np.ones_like(s))
    np.testing.assert_allclose(u, x * (1 - x), atol=1e-12)


def test_small_residual():
    x = uniform_grid(0.0, 1.0, 41)
    _, res = solve_poisson_1d(x, f_sin)
    assert res < 1e-8


def test_second_order_convergence():
    hs, errors = [], []
    for n in [21, 41, 81, 161]:
        x = uniform_grid(0.0, 1.0, n)
        u, _ = solve_poisson_1d(x, f_sin)
        hs.append(x[1] - x[0])
        errors.append(np.max(np.abs(u - np.sin(np.pi * x))))
    p = observed_order(hs, errors)
    np.testing.assert_allclose(p, 2.0, atol=0.05)


def test_non_uniform_grid_raises():
    x = np.array([0.0, 0.1, 0.5, 1.0])
    with pytest.raises(ValueError):
        assemble_poisson_1d(x, f_sin)


def test_too_few_points_raises():
    with pytest.raises(ValueError):
        assemble_poisson_1d(np.array([0.0, 1.0]), f_sin)