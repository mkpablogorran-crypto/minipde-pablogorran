import numpy as np
import matplotlib.pyplot as plt
from minipde import uniform_grid, assemble_poisson_1d, solve_linear_system, max_error

def source(x): return np.pi**2*np.sin(np.pi*x)
def exact(x): return np.sin(np.pi*x)

def main():
    x=uniform_grid(0,1,81); A,b=assemble_poisson_1d(x,source,sparse=True)
    methods=['sparse_direct','thomas','jacobi','gauss_seidel']
    print(f"{'método':16s} {'iter':>7s} {'residuo':>12s} {'error':>12s} {'tiempo(s)':>12s}")
    for m in methods:
        r=solve_linear_system(A,b,method=m,tol=1e-9,maxiter=20000)
        print(f"{m:16s} {r.iterations:7d} {r.residual_history[-1]:12.3e} {max_error(r.x,exact(x)):12.3e} {r.elapsed:12.3e}")
        if len(r.residual_history)>1: plt.semilogy(r.residual_history,label=m)
    plt.xlabel('Iteración'); plt.ylabel('Residuo relativo'); plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()
if __name__=='__main__': main()
