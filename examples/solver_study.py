"""Estudio de solvers lineales en Poisson 1D (ejercicios de clase, MiniPDE v0.4).

Problema: -u'' = pi^2 sin(pi x) en (0, 1), u(0) = u(1) = 0, u = sin(pi x).

Para cada malla n se resuelve con los cuatro métodos y se imprime una tabla
con iteraciones, residuo relativo final, error máximo, tiempo y si el método
ha alcanzado la tolerancia. Para cada n se guarda la gráfica semilogarítmica
del residuo de Jacobi y Gauss-Seidel en figures/.

Ejemplos:
    python examples/solver_study.py                 # n = 41, 81, 161, 321
    python examples/solver_study.py --n 81           # solo n = 81
    python examples/solver_study.py --n 161 321 --skip gauss_seidel
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from minipde import assemble_poisson_1d, max_error, solve_linear_system, uniform_grid

METHODS = ["sparse_direct", "thomas", "jacobi", "gauss_seidel"]
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"


def source(x):
    return np.pi**2 * np.sin(np.pi * x)


def exact(x):
    return np.sin(np.pi * x)


def run_mesh(n, methods, tol, maxiter):
    x = uniform_grid(0, 1, n)
    A, b = assemble_poisson_1d(x, source, sparse=True)

    print(f"\n=== n = {n}  (h = {1 / (n - 1):.5f}) ===")
    print(
        f"{'método':16s} {'iter':>7s} {'residuo':>12s} {'error':>12s} "
        f"{'tiempo(s)':>12s} {'convergido':>11s}"
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    for m in methods:
        r = solve_linear_system(A, b, method=m, tol=tol, maxiter=maxiter)
        res = r.residual_history[-1]
        ok = "sí" if res <= tol else "NO (maxiter)"
        print(
            f"{m:16s} {r.iterations:7d} {res:12.3e} "
            f"{max_error(r.x, exact(x)):12.3e} {r.elapsed:12.3e} {ok:>11s}"
        )
        if len(r.residual_history) > 1:
            ax.semilogy(r.residual_history, label=m)

    ax.axhline(tol, color="gray", ls=":", label=f"tolerancia {tol:g}")
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Residuo relativo")
    ax.set_title(f"Convergencia de los métodos iterativos (n = {n})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / f"residuo_iterativos_n{n}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Figura guardada en {out.relative_to(FIG_DIR.parent)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, nargs="+", default=[41, 81, 161, 321])
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--maxiter", type=int, default=20000)
    parser.add_argument("--skip", nargs="*", default=[], choices=METHODS)
    args = parser.parse_args()

    methods = [m for m in METHODS if m not in args.skip]
    for n in args.n:
        run_mesh(n, methods, args.tol, args.maxiter)


if __name__ == "__main__":
    main()