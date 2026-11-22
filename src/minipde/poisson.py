"""Poisson 1D with Dirichlet boundary conditions."""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def _grid(x):
    x = np.asarray(x, float)

    if x.ndim != 1:
        raise ValueError("invalid grid")

    if x.size < 3:
        raise ValueError("invalid grid")

    if not np.all(np.isfinite(x)):
        raise ValueError("invalid grid")

    dx = np.diff(x)

    if np.any(dx <= 0):
        raise ValueError("uniform grid required")

    if not np.allclose(dx, dx[0], rtol=1e-12, atol=1e-14):
        raise ValueError("uniform grid required")

    return x, float(dx[0])


def assemble_poisson_1d(
    x,
    source,
    left=0.0,
    right=0.0,
    sparse=True,
):
    """
    Assemble the finite-difference system

        -u'' = f

    on a uniform 1D mesh with Dirichlet boundary conditions.
    """

    x, h = _grid(x)

    f = np.asarray(source(x), float)

    if f.shape != x.shape:
        raise ValueError("source(x) must match x")

    if not np.all(np.isfinite(f)):
        raise ValueError("source(x) must match x")

    n = x.size

    if sparse:
        A = diags(
            [
                -np.ones(n - 1),
                2.0 * np.ones(n),
                -np.ones(n - 1),
            ],
            offsets=[-1, 0, 1],
            format="lil",
        )

        A[0, :] = 0.0
        A[0, 0] = 1.0

        A[-1, :] = 0.0
        A[-1, -1] = 1.0

        A = A.tocsr()

    else:
        A = np.zeros((n, n), dtype=float)

        A[0, 0] = 1.0
        A[-1, -1] = 1.0

        i = np.arange(1, n - 1)

        A[i, i - 1] = -1.0
        A[i, i] = 2.0
        A[i, i + 1] = -1.0

    b = h * h * f

    b[0] = left
    b[-1] = right

    return A, b


def solve_poisson_1d(
    x,
    source,
    left=0.0,
    right=0.0,
):
    """
    Solve the Poisson problem using SciPy sparse solvers.
    """

    A, b = assemble_poisson_1d(
        x,
        source,
        left,
        right,
        sparse=True,
    )

    return spsolve(A, b)


def residual_norm(A, u, b):
    """
    Infinity norm of the residual.
    """

    r = A @ u - b

    return float(np.linalg.norm(r, ord=np.inf))