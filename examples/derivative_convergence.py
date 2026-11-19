"""Estudio de convergencia de los operadores de diferencias finitas.

Caso manufacturado: u(x) = sin(x) en [0, 2*pi],
u'(x) = cos(x), u''(x) = -sin(x).

Ejecutar desde la raíz del repositorio:
    python examples/derivative_convergence.py
"""

import numpy as np

from minipde.mesh import uniform_grid
from minipde.operators import first_derivative, second_derivative
from minipde.verification import observed_order

N_VALUES = (21, 41, 81, 161, 321)


def main() -> None:
    hs, err1, err2 = [], [], []
    for n in N_VALUES:
        x = uniform_grid(0.0, 2 * np.pi, n)
        u = np.sin(x)
        hs.append(x[1] - x[0])
        err1.append(np.max(np.abs(first_derivative(x, u) - np.cos(x))))
        err2.append(np.max(np.abs(second_derivative(x, u) + np.sin(x))))

    p1 = observed_order(hs, err1)
    p2 = observed_order(hs, err2)

    e1_name, e2_name = "error u'", "error u''"
    print(f"{'N':>5} {'h':>10} {e1_name:>12} {'p1':>6} {e2_name:>12} {'p2':>6}")
    for k, n in enumerate(N_VALUES):
        o1 = f"{p1[k - 1]:6.2f}" if k > 0 else f"{'-':>6}"
        o2 = f"{p2[k - 1]:6.2f}" if k > 0 else f"{'-':>6}"
        print(f"{n:5d} {hs[k]:10.5f} {err1[k]:12.4e} {o1} {err2[k]:12.4e} {o2}")


if __name__ == "__main__":
    main()