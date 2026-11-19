"""Operadores de diferencias finitas de orden 2 en mallas 1D equiespaciadas.

Esquema (paso h = x[1] - x[0]):

- Nodos interiores: diferencias CENTRADAS, O(h^2).
- Extremo izquierdo (x[0]): diferencias ADELANTADAS (usa x[0], x[1], x[2], ...).
- Extremo derecho (x[-1]): diferencias ATRASADAS (usa x[-1], x[-2], x[-3], ...).

En los extremos se usan fórmulas adelantadas/atrasadas de orden 2 (con un
punto más que las de orden 1) para que todo el esquema mantenga O(h^2).
"""

from __future__ import annotations

import numpy as np


def _malla_equiespaciada(x, u, min_puntos: int):
    """Comprueba los datos y devuelve copias de x, u y el paso h."""
    x = np.array(x, dtype=float)
    u = np.array(u, dtype=float)

    if x.ndim != 1 or x.shape != u.shape:
        raise ValueError("x y u deben ser arrays 1D del mismo tamaño")
    if len(x) < min_puntos:
        raise ValueError(f"se requieren al menos {min_puntos} nodos")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(u))):
        raise ValueError("x y u deben contener solo valores finitos")

    h = x[1] - x[0]
    if h <= 0 or not np.allclose(np.diff(x), h, rtol=1e-10, atol=0.0):
        raise ValueError("la malla debe ser creciente y equiespaciada")

    return x, u, h


def first_derivative(x, u) -> np.ndarray:
    """Primera derivada u'(x) de orden 2 en todos los nodos (mínimo 3 nodos)."""
    x, u, h = _malla_equiespaciada(x, u, min_puntos=3)
    du = np.empty_like(u)

    # Interior, centrada:  u'_i ≈ (u_{i+1} - u_{i-1}) / (2h)
    du[1:-1] = (u[2:] - u[:-2]) / (2 * h)

    # Extremo izquierdo, ADELANTADA de orden 2:
    #   u'_0 ≈ (-3 u_0 + 4 u_1 - u_2) / (2h)
    du[0] = (-3 * u[0] + 4 * u[1] - u[2]) / (2 * h)

    # Extremo derecho, ATRASADA de orden 2:
    #   u'_N ≈ (3 u_N - 4 u_{N-1} + u_{N-2}) / (2h)
    du[-1] = (3 * u[-1] - 4 * u[-2] + u[-3]) / (2 * h)

    return du


def second_derivative(x, u) -> np.ndarray:
    """Segunda derivada u''(x) de orden 2 en todos los nodos (mínimo 4 nodos)."""
    x, u, h = _malla_equiespaciada(x, u, min_puntos=4)
    d2u = np.empty_like(u)

    # Interior, centrada:  u''_i ≈ (u_{i-1} - 2 u_i + u_{i+1}) / h^2
    d2u[1:-1] = (u[:-2] - 2 * u[1:-1] + u[2:]) / h**2

    # Extremo izquierdo, ADELANTADA de orden 2 (4 puntos):
    #   u''_0 ≈ (2 u_0 - 5 u_1 + 4 u_2 - u_3) / h^2
    d2u[0] = (2 * u[0] - 5 * u[1] + 4 * u[2] - u[3]) / h**2

    # Extremo derecho, ATRASADA de orden 2 (4 puntos):
    #   u''_N ≈ (2 u_N - 5 u_{N-1} + 4 u_{N-2} - u_{N-3}) / h^2
    d2u[-1] = (2 * u[-1] - 5 * u[-2] + 4 * u[-3] - u[-4]) / h**2

    return d2u