"""Problema de Poisson 1D con condiciones de Dirichlet.

Resuelve  -u''(x) = f(x)  en [a, b],  u(a) = ua,  u(b) = ub,
con diferencias finitas centradas de segundo orden en malla uniforme.
"""

from collections.abc import Callable

import numpy as np


def assemble_poisson_1d(
    x: np.ndarray,
    f: Callable[[np.ndarray], np.ndarray],
    ua: float = 0.0,
    ub: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Monta el sistema A u = b para -u'' = f con Dirichlet.

    Filas interiores: (-u[i-1] + 2 u[i] - u[i+1]) / h**2 = f(x[i]).
    Filas 0 y N: se sustituye la ecuación por la condición de contorno.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 3:
        raise ValueError("Se necesitan al menos 3 puntos de malla.")
    h = x[1] - x[0]
    if not np.allclose(np.diff(x), h):
        raise ValueError("La malla debe ser equiespaciada.")

    matrix = np.zeros((n, n))
    rhs = np.zeros(n)

    i = np.arange(1, n - 1)
    matrix[i, i - 1] = -1.0 / h**2
    matrix[i, i] = 2.0 / h**2
    matrix[i, i + 1] = -1.0 / h**2
    rhs[i] = f(x[i])

    matrix[0, 0] = 1.0
    rhs[0] = ua
    matrix[-1, -1] = 1.0
    rhs[-1] = ub
    return matrix, rhs


def solve_poisson_1d(
    x: np.ndarray,
    f: Callable[[np.ndarray], np.ndarray],
    ua: float = 0.0,
    ub: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Resuelve el sistema con np.linalg.solve.

    Devuelve la solución discreta u y el residuo ||A u - b||_inf.
    """
    matrix, rhs = assemble_poisson_1d(x, f, ua, ub)
    u = np.linalg.solve(matrix, rhs)
    residual = float(np.max(np.abs(matrix @ u - rhs)))
    return u, residual