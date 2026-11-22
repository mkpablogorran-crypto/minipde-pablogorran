"""Verificación de Poisson 1D con solución manufacturada (con gráficas).

u(x) = sin(pi x),  f(x) = pi^2 sin(pi x),  u(0) = u(1) = 0.
Se mide el error L-infinito, el residuo ||Au - b||_inf y el orden observado,
y se dibujan:
  1. Solución numérica frente a la exacta (malla gruesa).
  2. Error puntual |u_h - u| para varias mallas.
  3. Convergencia del error L-infinito en escala log-log (pendiente 2).
  4. Residuo ||Au - b||_inf frente a h.
"""

import matplotlib.pyplot as plt
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
    hs, errors, residuals, solutions = [], [], [], []

    for n in n_values:
        x = uniform_grid(0.0, 1.0, n)
        u, res = solve_poisson_1d(x, f, ua=0.0, ub=0.0)
        hs.append(x[1] - x[0])
        errors.append(np.max(np.abs(u - u_exact(x))))
        residuals.append(res)
        solutions.append((x, u))

    orders = observed_order(hs, errors)

    print(f"{'N':>5} {'h':>10} {'error Linf':>12} {'residuo':>12} {'orden':>7}")
    for k, n in enumerate(n_values):
        p = f"{orders[k - 1]:7.3f}" if k > 0 else f"{'-':>7}"
        print(f"{n:5d} {hs[k]:10.5f} {errors[k]:12.4e} {residuals[k]:12.4e} {p}")

    plot_results(n_values, hs, errors, residuals, solutions)


def plot_results(n_values, hs, errors, residuals, solutions) -> None:
    hs = np.array(hs)
    errors = np.array(errors)
    x_fine = np.linspace(0.0, 1.0, 400)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(r"Poisson 1D: $-u'' = \pi^2 \sin(\pi x)$,  $u(0)=u(1)=0$")

    # 1. Solución numérica vs exacta (malla gruesa, para que se vean los nodos)
    ax = axes[0, 0]
    x, u = solutions[0]
    ax.plot(x_fine, u_exact(x_fine), "k-", label=r"exacta $\sin(\pi x)$")
    ax.plot(x, u, "o", color="tab:blue", label=f"numérica (N = {n_values[0]})")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.set_title("Solución")
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Error puntual para varias mallas
    ax = axes[0, 1]
    for (x, u), n in zip(solutions[:4], n_values[:4], strict=True):
        ax.plot(x, np.abs(u - u_exact(x)), ".-", label=f"N = {n}")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$|u_h - u|$")
    ax.set_title("Error puntual")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    # 3. Convergencia log-log con recta de referencia de pendiente 2
    ax = axes[1, 0]
    ax.loglog(hs, errors, "o-", color="tab:blue", label=r"error $L^\infty$")
    ref = errors[0] * (hs / hs[0]) ** 2
    ax.loglog(hs, ref, "k--", label=r"referencia $O(h^2)$")
    ax.set_xlabel("h")
    ax.set_ylabel(r"$\|u_h - u\|_\infty$")
    ax.set_title("Convergencia")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    # 4. Residuo del sistema lineal
    ax = axes[1, 1]
    ax.loglog(hs, residuals, "s-", color="tab:red", label=r"$\|Au - b\|_\infty$")
    ax.axhline(np.finfo(float).eps, color="gray", ls=":", label="épsilon de máquina")
    ax.set_xlabel("h")
    ax.set_ylabel("residuo")
    ax.set_title("Residuo del sistema lineal")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
    