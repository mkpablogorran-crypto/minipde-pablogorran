"""Mallas unidimensionales para MiniPDE."""

from __future__ import annotations

import math

import numpy as np


def uniform_grid(a: float, b: float, n: int) -> np.ndarray:
    """Genera una malla 1D equiespaciada en el intervalo [a, b].

    Parameters
    ----------
    a, b : float
        Extremos del dominio. Deben ser finitos y cumplir b > a.
    n : int
        Número de puntos de la malla (incluye los dos extremos). n >= 2.

    Returns
    -------
    np.ndarray
        Array 1D de n nodos estrictamente crecientes, con x[0] = a y
        x[-1] = b. El paso es h = (b - a) / (n - 1) y se puede obtener
        como x[1] - x[0], por eso no se devuelve aparte.

    Raises
    ------
    ValueError
        Si n no es un entero >= 2, si a o b no son finitos o si b <= a.
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise TypeError("n debe ser un número entero")
    if n < 2:
        raise ValueError("n debe ser al menos 2")
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError("a y b deben ser números finitos")
    if not b > a:
        raise ValueError("se requiere b > a")

    return np.linspace(float(a), float(b), int(n))