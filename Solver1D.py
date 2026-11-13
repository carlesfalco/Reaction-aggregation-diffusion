import numpy as np
from numba import njit, prange
from scipy.integrate import solve_ivp

@njit
def mob(rho, model='I'):

    if model == 'II':
        return np.maximum(rho*(1-rho), 0.0)

    return np.maximum(rho, 0.0)

@njit(parallel=True, fastmath=False)
def laplacian_1d(u, dx):
    n = u.shape[0]
    out = np.empty_like(u)
    dx2 = dx * dx

    # interior
    for i in prange(1, n - 1):
        out[i] = (u[i - 1] - 2.0 * u[i] + u[i + 1]) / dx2

    # periodic boundaries
    out[0]     = (u[n - 1] - 2.0 * u[0]     + u[1]) / dx2
    out[n - 1] = (u[n - 2] - 2.0 * u[n - 1] + u[0]) / dx2

    return out


@njit
def entropy(rho, mu, dx, model='I', alpha=5):

    if model == 'II':
        omega = alpha - mu
        return laplacian_1d(rho, dx) + alpha*np.log(1-rho) + omega*rho

    return laplacian_1d(rho, dx) - mu*rho
    


@njit(parallel = True)
def velocities(rho, mu, dx, model = 'I', alpha = 5):
    
    n = rho.shape[0]
    v = np.empty(n, dtype=rho.dtype)
    a = entropy(rho, mu, dx, model, alpha)

    inv_2dx = 1.0 / (2.0 * dx)

    for i in prange(n):
        im = (i - 1) % n
        ip = (i + 1) % n
        ipp = (i + 2) % n

        v[i] = (a[ip] - a[i])/dx
        
    return v
    


@njit(parallel = True)
def flux(rho, mu, dx, model='I', alpha=5):

    v = velocities(rho, mu, dx, model, alpha)
    rhomob = mob(rho, model)

    n = len(rho)
    F = np.empty(n)

    for i in prange(n):
        if v[i] >= 0:
            F[i] = v[i] * rhomob[i]
        else:
            F[i] = v[i] * rhomob[(i+1) % n]

    return F

@njit(parallel = True)
def rhs(rho, mu, dx, model = 'I', alpha = 5):

    F = flux(rho, mu, dx, model, alpha)
    n = rho.shape[0]
    out = np.empty(n, dtype=F.dtype)

    for j in prange(n):
        jm = (j - 1) % n
        out[j] = (F[j] - F[jm]) / dx
    return -out


def lsoda_step(t0, rho0, tau, mu, dx, model = 'I', alpha = 5):

    def f_t(t, y):
        return rhs(y, mu, dx, model, alpha)
        
    sol = solve_ivp(fun = f_t, t_span = (t0, t0 + tau), y0 = rho0, method = 'LSODA', t_eval=[t0 + tau])
  
    return sol.y[:,0]

def logistic(rho, tau):
    
    return rho / ((1 - rho)*np.exp(-tau)  + rho)

def splitting_solver(init_rho, mu, tmax, dt, dx, model = 'I', alpha = 5, proliferation = True):
    t = 0
    rhos = [init_rho]
    tspan = [0]
    rho = init_rho
    tf = 0
    while t < tmax:

        if proliferation:
            rho = logistic(rho,dt*.5)
          
            rho = lsoda_step(t, rho, dt, mu, dx, model, alpha)
       
            rho = logistic(rho,dt*.5)
            
            t += dt
        else:
            t += dt
            rho = lsoda_step(t, rho, dt, mu, dx, model, alpha)
            
        if t-tf > .1:
            print(t, end = '\r')
            rhos.append(rho)
            tspan.append(t)
            tf = t
        
    return np.array(tspan), np.array(rhos)