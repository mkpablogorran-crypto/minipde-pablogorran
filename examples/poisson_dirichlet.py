"""Verificación de Poisson 1D con solución manufacturada.

u(x) = sin(pi x),  f(x) = pi^2 sin(pi x),  u(0) = u(1) = 0.
Se mide el error L-infinito, el residuo ||Au - b||_inf y el orden observado.
"""

import numpy as np

from minipde.mesh import uniform_grid
from minipde.poisson import solve_poisson_1d
from minipde.verification import observed_order


def u_exact(x: np.ndarray) -> np.ndarray:
    return np.sin(np.pi * x)


def f(x: np.ndarray) -> np.ndarray:
    return np.pi**2 * np.sin(np.pi * x)


def main() -> None:
    n_values = [11, 21, 41, 81, 161, 321]
    hs, errors, residuals = [], [], []

    for n in n_values:
        x = uniform_grid(0.0, 1.0, n)
        u, res = solve_poisson_1d(x, f, ua=0.0, ub=0.0)
        hs.append(x[1] - x[0])
        errors.append(np.max(np.abs(u - u_exact(x))))
        residuals.append(res)

    orders = observed_order(hs, errors)

    print(f"{'N':>5} {'h':>10} {'error Linf':>12} {'residuo':>12} {'orden':>7}")
    for k, n in enumerate(n_values):
        p = f"{orders[k - 1]:7.3f}" if k > 0 else f"{'-':>7}"
        print(f"{n:5d} {hs[k]:10.5f} {errors[k]:12.4e} {residuals[k]:12.4e} {p}")


if __name__ == "__main__":
    main()