"""Numerical verification utilities."""
import numpy as np

def max_error(numerical, exact):
    a, b = np.asarray(numerical, float), np.asarray(exact, float)
    if a.shape != b.shape: raise ValueError("arrays must have the same shape")
    return float(np.max(np.abs(a-b)))

def relative_residual(A,x,b):
    r=np.asarray(b)-A@np.asarray(x)
    den=np.linalg.norm(b)
    return float(np.linalg.norm(r)/(den if den else 1.0))


def observed_order(h, error):
    h, e = np.asarray(h, float), np.asarray(error, float)
    if h.ndim != 1 or e.shape != h.shape or h.size < 2 or np.any(h <= 0) or np.any(e <= 0):
        raise ValueError("positive compatible refinement data required")
    return np.log(e[:-1]/e[1:]) / np.log(h[:-1]/h[1:])
