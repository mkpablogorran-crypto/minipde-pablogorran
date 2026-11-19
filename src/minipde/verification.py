"""Herramientas de verificación: orden de convergencia observado."""

from __future__ import annotations

import numpy as np


def observed_order(h, error) -> np.ndarray:
    """Orden observado entre mallas consecutivas.

    p_k = log(E_k / E_{k+1}) / log(h_k / h_{k+1})

    Si h se divide entre 2 en cada malla, queda p = log2(E_h / E_{h/2}).
    """
    h = np.asarray(h, dtype=float)
    error = np.asarray(error, dtype=float)

    if h.ndim != 1 or h.shape != error.shape:
        raise ValueError("h y error deben ser arrays 1D del mismo tamaño")
    if len(h) < 2:
        raise ValueError("se necesitan al menos 2 mallas")
    if np.any(h <= 0) or np.any(error <= 0):
        raise ValueError("h y error deben ser positivos")

    return np.log(error[:-1] / error[1:]) / np.log(h[:-1] / h[1:])