import numpy as np
from numba import njit, prange
from scipy.integrate import solve_ivp

@njit
def mob(rho):
    return np.maximum(rho, 0.0)

# --- 2D Laplacian with periodic BCs ---
@njit(parallel=True, fastmath=False, cache=False)
def laplacian_2d(u, dx):
    nx, ny = u.shape
    out = np.empty_like(u)
    dx2 = dx * dx

    for i in prange(nx):
        im = (i - 1) % nx
        ip = (i + 1) % nx
        for j in range(ny):
            jm = (j - 1) % ny
            jp = (j + 1) % ny
            out[i, j] = (u[im, j] - 2.0 * u[i, j] + u[ip, j]) / dx2 + \
                        (u[i, jm] - 2.0 * u[i, j] + u[i, jp]) / dx2
    return out

# --- entropy: laplacian - mu * rho ---
@njit
def entropy_2d(rho, mu, dx):
    return laplacian_2d(rho, dx) - mu * rho

# --- velocities: compute components v_x, v_y from gradient ---
@njit(parallel=True, fastmath=False, cache=False)
def velocities_2d(rho, mu, dx):
    nx, ny = rho.shape
    a = entropy_2d(rho, mu, dx)

    v_x = np.empty((nx, ny), dtype=rho.dtype)
    v_y = np.empty((nx, ny), dtype=rho.dtype)

    inv_dx = 1.0 / dx

    for i in prange(nx):
        ip = (i + 1) % nx
        for j in range(ny):
            jp = (j + 1) % ny
            # forward difference (periodic) for velocity components
            v_x[i, j] = (a[ip, j] - a[i, j]) * inv_dx
            v_y[i, j] = (a[i, jp] - a[i, j]) * inv_dx

    return v_x, v_y

@njit(parallel=True)
def flux_2d(rho, mu, dx):
    v_x, v_y = velocities_2d(rho, mu, dx)
    rhomob = mob(rho)

    nx, ny = rho.shape
    F_x = np.empty_like(rho)
    F_y = np.empty_like(rho)

    for i in prange(nx):
        ip = (i + 1) % nx
        for j in range(ny):
            jp = (j + 1) % ny

            vx = v_x[i, j]
            vy = v_y[i, j]

            # Upwind flux in x
            if vx >= 0:
                F_x[i, j] = vx * rhomob[i, j]
            else:
                F_x[i, j] = vx * rhomob[ip, j]

            # Upwind flux in y
            if vy >= 0:
                F_y[i, j] = vy * rhomob[i, j]
            else:
                F_y[i, j] = vy * rhomob[i, jp]

    return F_x, F_y

# --- rhs: divergence of fluxes, returned as 2D array ---
@njit(parallel=True, fastmath=False)
def rhs_2d(rho, mu, dx):
    nx, ny = rho.shape
    F_x, F_y = flux_2d(rho, mu, dx)

    out = np.empty((nx, ny), dtype=rho.dtype)
    inv_dx = 1.0 / dx

    for i in prange(nx):
        im = (i - 1) % nx
        for j in range(ny):
            jm = (j - 1) % ny
        
            div_x = (F_x[i, j] - F_x[im, j]) * inv_dx
            div_y = (F_y[i, j] - F_y[i, jm]) * inv_dx
            out[i, j] = -(div_x + div_y)  
    return out

# --- wrapper for solve_ivp: takes flattened y, returns flattened rhs ---
def rhs_flat(t, y_flat, mu, dx, shape):

    rho = y_flat.reshape(shape)
    res = rhs_2d(rho, mu, dx)
    return res.ravel()


def lsoda_step_2d(t0, rho0_2d, tau, mu, dx):
    shape = rho0_2d.shape
    y0 = rho0_2d.ravel()

    def f_t(t, y):
        return rhs_flat(t, y, mu, dx, shape)

    sol = solve_ivp(fun=f_t, t_span=(t0, t0 + tau), y0=y0, method='LSODA', t_eval=[t0 + tau])
    y_new = sol.y[:, 0]
    return y_new.reshape(shape)

# --- logistic reaction (applied pointwise) ---
@njit
def logistic_2d(rho, tau):

    out = rho / ((1 - rho) * np.exp(-tau) + rho)
    return out

# --- splitting solver for 2D ---
def splitting_solver_2d(init_rho, mu, tmax, dt, dx, proliferation = True):
    """
    init_rho: 2D array (nx, ny)
    mu, tmax, dt, dx: scalars
    returns tspan array and rhos array with shape (n_snapshots, nx, ny)
    """
    t = 0.0
    rhos = [init_rho.copy()]
    tspan = [0.0]
    rho = init_rho.copy()
    tf = 0.0

    while t < tmax:
        # half logistic
        if proliferation:
 
            rho = logistic_2d(rho, dt * 0.5)
    
            # LSODA step for diffusion/transport
            
            rho = lsoda_step_2d(t, rho, dt, mu, dx)
            # half logistic
         
            rho = logistic_2d(rho, dt * 0.5)
            t += dt
            
        else:
            t += dt
            rho = lsoda_step_2d(t, rho, dt, mu, dx)

        if t - tf > 0.25:
            print(t, end='\r')
            rhos.append(rho.copy())
            tspan.append(t)
            tf = t

    return np.array(tspan), np.array(rhos)

