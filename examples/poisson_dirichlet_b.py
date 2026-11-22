"""Ejemplo B de Poisson 1D: fuente constante y Dirichlet no homogénea.

    -u''(x) = 1   en [0, 1],    u(0) = 0,   u(1) = 1.

Solución analítica (integrando dos veces y aplicando las condiciones):
    u(x) = -x^2/2 + 3x/2 = x (3 - x) / 2.

Como u es un polinomio de grado 2, la diferencia centrada de segundo orden
es EXACTA (su error de truncamiento depende de u'''' = 0). Por tanto el error
no converge como h^2: queda al nivel del redondeo para cualquier malla.
"""

import matplotlib.pyplot as plt
import numpy as np

from minipde.mesh import uniform_grid
from minipde.poisson import solve_poisson_1d

UA, UB = 0.0, 1.0


def u_exact(x: np.ndarray) -> np.ndarray:
    return x * (3.0 - x) / 2.0


def f(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x)


def main() -> None:
    n_values = [11, 21, 41, 81, 161, 321]
    hs, errors, residuals, solutions = [], [], [], []

    for n in n_values:
        x = uniform_grid(0.0, 1.0, n)
        u, res = solve_poisson_1d(x, f, ua=UA, ub=UB)
        hs.append(x[1] - x[0])
        errors.append(np.max(np.abs(u - u_exact(x))))
        residuals.append(res)
        solutions.append((x, u))

    print(f"{'N':>5} {'h':>10} {'error Linf':>12} {'residuo':>12}")
    for n, h, e, r in zip(n_values, hs, errors, residuals, strict=True):
        print(f"{n:5d} {h:10.5f} {e:12.4e} {r:12.4e}")
    print("\nEl esquema es exacto para soluciones cuadráticas: el error es solo")
    print("redondeo, así que no tiene sentido calcular un orden observado.")

    plot_results(n_values, hs, errors, residuals, solutions)


def plot_results(n_values, hs, errors, residuals, solutions) -> None:
    x_fine = np.linspace(0.0, 1.0, 400)
    eps = np.finfo(float).eps

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(r"Poisson 1D (ejemplo B): $-u'' = 1$,  $u(0)=0$,  $u(1)=1$")

    # 1. Solución numérica vs exacta
    ax = axes[0, 0]
    x, u = solutions[0]
    ax.plot(x_fine, u_exact(x_fine), "k-", label=r"exacta $x(3-x)/2$")
    ax.plot(x, u, "o", color="tab:blue", label=f"numérica (N = {n_values[0]})")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.set_title("Solución")
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Error puntual: nivel de redondeo
    ax = axes[0, 1]
    for (x, u), n in zip(solutions[:4], n_values[:4], strict=True):
        ax.plot(x, np.abs(u - u_exact(x)), ".-", label=f"N = {n}")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$|u_h - u|$")
    ax.set_title("Error puntual (solo redondeo)")
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. Error L-infinito frente a h: no sigue la recta h^2
    ax = axes[1, 0]
    hs_arr = np.array(hs)
    ax.loglog(hs_arr, errors, "o-", color="tab:blue", label=r"error $L^\infty$")
    ax.loglog(hs_arr, hs_arr**2, "k--", label=r"referencia $h^2$")
    ax.axhline(eps, color="gray", ls=":", label="épsilon de máquina")
    ax.set_xlabel("h")
    ax.set_ylabel(r"$\|u_h - u\|_\infty$")
    ax.set_title("Error frente a h")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    # 4. Residuo del sistema lineal
    ax = axes[1, 1]
    ax.loglog(hs_arr, residuals, "s-", color="tab:red", label=r"$\|Au - b\|_\infty$")
    ax.axhline(eps, color="gray", ls=":", label="épsilon de máquina")
    ax.set_xlabel("h")
    ax.set_ylabel("residuo")
    ax.set_title("Residuo del sistema lineal")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()