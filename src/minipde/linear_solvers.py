from dataclasses import dataclass
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

@dataclass
class SolverResult:
    x: np.ndarray
    converged: bool
    iterations: int
    residual_history: list[float]
    elapsed: float
    method: str

def _relres(A,x,b):
    r=b-A@x; d=np.linalg.norm(b)
    return float(np.linalg.norm(r)/(d if d else 1.0))

def thomas(lower,diag,upper,b):
    a=np.array(lower,float,copy=True); d=np.array(diag,float,copy=True)
    c=np.array(upper,float,copy=True); q=np.array(b,float,copy=True)
    n=d.size
    for i in range(1,n):
        m=a[i-1]/d[i-1]; d[i]-=m*c[i-1]; q[i]-=m*q[i-1]
    x=np.empty(n); x[-1]=q[-1]/d[-1]
    for i in range(n-2,-1,-1): x[i]=(q[i]-c[i]*x[i+1])/d[i]
    return x

def solve_linear_system(A,b,method='auto',tol=1e-10,maxiter=10000,x0=None):
    b=np.asarray(b,float); n=b.size
    if method=='auto': method='sparse_direct' if sparse.issparse(A) else 'dense_direct'
    t=time.perf_counter(); hist=[]
    if method=='dense_direct':
        x=np.linalg.solve(np.asarray(A),b); hist=[_relres(A,x,b)]; ok=True; it=1
    elif method=='sparse_direct':
        x=spsolve(A,b); hist=[_relres(A,x,b)]; ok=True; it=1
    elif method in {'jacobi','gauss_seidel'}:
        M=A.tocsr() if sparse.issparse(A) else np.asarray(A)
        x=np.zeros(n) if x0 is None else np.array(x0,float,copy=True)
        diag=M.diagonal() if sparse.issparse(M) else np.diag(M)
        if np.any(diag==0): raise ValueError('zero on diagonal')
        ok=False
        for k in range(1,maxiter+1):
            if method=='jacobi':
                r=b-M@x; xnew=x+r/diag
            else:
                xnew=x.copy()
                for i in range(n):
                    row=M.getrow(i).toarray().ravel() if sparse.issparse(M) else M[i]
                    xnew[i]=(b[i]-row[:i]@xnew[:i]-row[i+1:]@x[i+1:])/diag[i]
            x=xnew; hist.append(_relres(M,x,b))
            if hist[-1] <= tol: ok=True; break
        it=k
    elif method=='thomas':
        D=A.toarray() if sparse.issparse(A) else np.asarray(A)
        x=thomas(np.diag(D,-1),np.diag(D),np.diag(D,1),b)
        hist=[_relres(A,x,b)]; ok=True; it=1
    else: raise ValueError(f'unknown method: {method}')
    return SolverResult(np.asarray(x),ok,it,hist,time.perf_counter()-t,method)
