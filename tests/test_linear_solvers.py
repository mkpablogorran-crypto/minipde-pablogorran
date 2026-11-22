import numpy as np
import pytest
from minipde import uniform_grid, assemble_poisson_1d, solve_linear_system, max_error

def src(x): return np.pi**2*np.sin(np.pi*x)
@pytest.mark.parametrize('method',['sparse_direct','thomas','jacobi','gauss_seidel'])
def test_poisson(method):
    x=uniform_grid(0,1,41); A,b=assemble_poisson_1d(x,src,sparse=True)
    r=solve_linear_system(A,b,method=method,tol=1e-9,maxiter=20000)
    assert r.converged
    assert r.residual_history[-1] < 1e-8
    assert max_error(r.x,np.sin(np.pi*x)) < 1e-3
