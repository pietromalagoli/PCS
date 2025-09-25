import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import concurrent.futures
from scipy.integrate import odeint
from tqdm.contrib.telegram import tqdm

import aux      # it's a file with some generic auxiliary functions I often use

# Utility functions
def stability(par):         # In which regime are we?
    print(f'In which regime are we? {'First regime:(0,y_0) unstable, (x*,y*) stable' \
        if par['alpha']*par['sigma']>(par['lambda']*par['gamma']) else 'Second regime:(0,y_0) stable, (x*,y*) non-biological'}')
    if par['alpha']*par['sigma']>(par['lambda']*par['gamma']):
        if par['lambda']**2 * par['gamma']**2 > 4*par['alpha']**2 * (par['sigma']*par['alpha']-par['lambda']*par['gamma']):
            print('The attractor (x*,y*) is a stable node (positive disciminant -> no oscillations)') 
        elif par['lambda']**2 * par['gamma']**2 == 4*par['alpha']**2 * (par['sigma']*par['alpha']-par['lambda']*par['gamma']):
            print('The attractor (x*,y*) is a degenerate node (null discriminant)')
        else:
            print(f'The attractor (x*,y*) is a stable spiral (negative discriminant -> damped oscillations)')

def add_par_box(par,coord=[0.98,0.60]):
    # Build lines for any numerical parameters in the dict, formatted with .2f
    lines = []
    for k in sorted(par.keys()):
        v_raw = par[k]
        try:
            v = float(v_raw)
        except Exception:
            continue
        # escape underscores for safe rendering in text
        key_label = str(k).replace('_', r'\_')
        # display integers with no decimals, but skip booleans
        if not isinstance(v_raw, bool) and isinstance(v_raw, (int, np.integer)):
            lines.append(f'{key_label}={v:.0f}')
        else:
            lines.append(f'{key_label}={v:.2f}')

    if not lines:
        return

    textstr = '\n'.join(lines)

    props = dict(boxstyle='round', facecolor='white', alpha=0.3)
    ax = plt.gca()
    ax.text(coord[0], coord[1], textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', horizontalalignment='right', bbox=props)

def plot_evolution(params):
    dim, N, time_steps, dt, z0, par, mode = params
    models = [lattice1,lattice2,lattice3]
    model = models[dim-1]
    t_s = 0 + dt * np.arange(time_steps)
    # Solve again the ODEs so to have it with the right parameters, in case that they're different for the stochastic case
    sol = odeint(system,z0,t_s,args=(par,))
    X_ode, Y_ode = sol[:,0], sol[:,1]
    local_par = par.copy()
    local_par['Dx'] = 0
    local_par['Dy'] = 0
    X_stoc, Y_stoc = model([N,time_steps,dt,local_par,0])
    if mode == 2:
        X_ode = 0
        Y_ode = 0
    args = [N,time_steps,dt,par,mode]     # [N,time_steps,par,mode]
    X_diff, Y_diff = model(args)

    if mode == 0:
        scalex = np.max(X_diff)/(np.max(X_ode) or 1)
        scaley = np.max(Y_diff)/(np.max(Y_ode) or 1)
    else:
        scalex = 1
        scaley = 1
    fig, ax = plt.subplots(2,2,figsize=(14,12))
    # plot the evolution of x(t)
    ax[0,0].plot(X_ode,c='deepskyblue',label='x(t)-MFT',alpha=0.7)
    ax[0,0].plot(X_stoc,c='orange', label=f'x(t)-Stochastic,Dx={local_par['Dx']}',alpha=0.7)      
    ax[0,0].plot(X_diff/scalex,c='green',label=f'x(t)-Stochastic,Dx={par['Dx']}',alpha=0.7)
    ax[0,0].set_title([f'Population({dim}D)' if mode == 0 else f'Filling Fraction ({dim}D)'])
    ax[0,0].set_xlabel('Time t')
    ax[0,0].set_ylabel('X')
    ax[0,0].set_xscale('log')
    ax[0,0].grid(True, which="both",alpha=0.4,linestyle='--')

    ax[0,1].plot(X_ode,c='deepskyblue',label='x(t)-MFT',alpha=0.7)
    ax[0,1].plot(X_stoc/scalex,c='orange',label=f'x(t)-Stochastic,Dx={local_par['Dx']}',alpha=0.7)
    ax[0,1].plot(X_diff/scalex,c='green',label=f'x(t)-Stochastic,Dx={par['Dx']}',alpha=0.7)
    ax[0,1].set_title('ZOOM')
    ax[0,1].set_xlim(100,time_steps)
    ax[0,1].set_ylim(0,100)
    ax[0,1].set_xlabel('Time t')
    ax[0,1].set_ylabel('X')
    ax[0,1].set_xscale('log')
    ax[0,1].grid(True, which="both",alpha=0.4,linestyle='--')

    ax[1,0].plot(X_ode, Y_ode, c='deepskyblue', label='MFT',alpha=0.7)
    ax[1,0].plot(X_stoc/scalex, Y_stoc/scaley, c='green', label=f'x(t)-Stochastic,Dx={local_par['Dx']}',alpha=0.7)
    ax[1,0].plot(X_diff/scalex, Y_diff/scaley, c='green', label=f'x(t)-Stochastic,Dx={par['Dx']}',alpha=0.7)
    ax[1,0].set_title(f'Trajectory of the solutions ({dim}D)')
    ax[1,0].set_xlabel('X')
    ax[1,0].set_ylabel('Y')
    ax[1,0].grid(True, which="both",alpha=0.4,linestyle='--')

    ax[1,1].plot(X_ode, Y_ode, c='deepskyblue', label='MFT',alpha=0.7)
    ax[1,1].plot(X_stoc/scalex, Y_stoc/scaley, c='green', label=f'x(t)-Stochastic,Dx={local_par['Dx']}',alpha=0.7)
    ax[1,1].plot(X_diff/scalex, Y_diff/scaley, c='green', label=f'x(t)-Stochastic,Dx={par['Dx']}',alpha=0.7)
    ax[1,1].set_title('ZOOM')
    ax[1,1].axvline(x=1,c='r',label='x=1',linestyle='--',alpha=0.4)
    ax[1,1].set_xlim(-1,10)
    ax[1,1].set_xlabel('X')
    ax[1,1].set_ylabel('Y')
    ax[1,1].grid(True, which="both",alpha=0.4,linestyle='--')
    add_par_box(par)
    plt.legend(loc='upper right')
    plt.show()

# Mean Field
def system(z:np.ndarray ,t:np.ndarray ,par:dict):
    """ODEs system for the ISP problem.

    Args:
        z (np.ndarray): _description_
        t (np.ndarray): _description_
        par (dict): _description_

    Returns:
        _type_: _description_
    """
    x, y = z
    dxdt = par['alpha']*x - par['gamma']*x*y
    dydt = par['lambda'] + par['nu']*x*y - par['sigma']*y
    if x < 0:
        x = 0.
    if y < 0:
        y = 0.
    return [dxdt,dydt]

# Stochastic system
def lattice1(args,rng=None):
    """
    Vectorized stochastic lattice. Uses a numpy Generator for faster poisson draws.
    """
    N, time_steps, dt, par, mode = args
    if rng is None:
        rng = np.random.default_rng()
    alpha = par['alpha']        # pre-extract parameters (avoid repeated dict lookups)
    gamma = par['gamma']
    lam = par['lambda']
    nu = par['nu']
    sigma = par['sigma']
    Dx = par['Dx']
    Dy = par['Dy']
    lattice = np.zeros((N, 2), dtype=int)      # integer lattice for counts
    lattice[:, 0] = 3       # initialization
    if Dx == 0 and Dy == 0:      # no diffusion
        lattice[:, 1] = lam/sigma
    else:
        lattice[:, 1] = rng.poisson(lam=1)
    if mode == 2:  
        densityx = np.zeros(time_steps)
        densityy = np.zeros(time_steps)
    lattice_old = lattice.copy()
    X_stoc = np.zeros(time_steps, dtype=float)
    Y_stoc = np.zeros(time_steps, dtype=float)
    for t in range(1,time_steps+1):
        prod = lattice_old[:, 0] * lattice_old[:, 1]            # elementwise xy
        # vectorized Poisson draws
        births_x = rng.poisson(alpha * lattice_old[:, 0] * dt)
        kills_x = rng.poisson(gamma * prod * dt)
        np.minimum(kills_x,lattice[:,0],out=kills_x)    # check that I'm not killing more individuals than the actual population of the site
        births_y = rng.poisson(lam * dt, size=N)
        dup_y = rng.poisson(nu * prod * dt)
        deaths_y = rng.poisson(sigma * lattice_old[:, 1] * dt)
        np.minimum(deaths_y,lattice[:,1],out=deaths_y)  # check that I'm not killing more individuals than the actual population of the site
        # update lattice (vectorized)
        lattice[:, 0] += births_x - kills_x 
        lattice[:, 1] += births_y + dup_y - deaths_y
        if Dx or Dy > 0:    # diffusion
            diffX = rng.poisson(Dx*lattice_old[:,0] * dt)
            diffY = rng.poisson(Dy*lattice_old[:,1] * dt)
            np.minimum(diffX,lattice_old[:,0],out=diffX)    # the maximum of individuals to diffuse out is the population of the site
            np.minimum(diffY,lattice_old[:,1],out=diffY)
            # Split diffused individuals between left/right with a binomial draw (vectorized)
            leftX = rng.binomial(diffX, 0.5)
            rightX = diffX - leftX
            leftY = rng.binomial(diffY, 0.5)
            rightY = diffY - leftY
            # Add incoming migrants to neighbours on the circular lattice
            lattice[:, 0] += np.roll(leftX, -1) + np.roll(rightX, +1)
            lattice[:, 1] += np.roll(leftY, -1) + np.roll(rightY, +1)
            lattice[:,0] -= diffX   # take out the diffused individuals
            lattice[:,1] -= diffY
        np.maximum(lattice, 0, out=lattice,dtype=int)         # clip negatives in-place (no negative populations)
        np.minimum(lattice,int(1e9),out=lattice,dtype=int)
        lattice_old = lattice.copy()           # prepare for next step (must copy to avoid aliasing)
        # mean
        X_stoc[t-1] = lattice[:, 0].mean()
        Y_stoc[t-1] = lattice[:, 1].mean()
        if mode == 2:   # density
            densityx[t-1] = np.count_nonzero(lattice[:, 0]) / N
            densityy[t-1] = np.count_nonzero(lattice[:, 1]) / N
    if mode == 1:   # filling fraction
        return np.count_nonzero(lattice[:, 0]) / N
    elif mode == 2: 
        return densityx, densityy
    return X_stoc, Y_stoc   # mean populations on the whole lattice

def lattice2(args,rng=None):
    """
    Vectorized stochastic lattice. Uses a numpy Generator for faster poisson draws.
    """
    N, time_steps, dt, par, mode = args
    if rng is None:
        rng = np.random.default_rng()
    alpha = par['alpha']        # pre-extract parameters (avoid repeated dict lookups)
    gamma = par['gamma']
    lam = par['lambda']
    nu = par['nu']
    sigma = par['sigma']
    Dx = par['Dx']
    Dy = par['Dy']
    lattice = np.zeros((N, N, 2), dtype=int)      # integer lattice for counts
    lattice[:, :, 0] = 3
    if Dx == 0 and Dy == 0:      # no diffusion
        lattice[:, :, 1] = lam/sigma
    else:
        lattice[:, :, 1] = rng.poisson(lam=1)
    if mode == 2:
        densityx = np.zeros(time_steps)
        densityy = np.zeros(time_steps)
    lattice_old = lattice.copy()
    X_stoc = np.zeros(time_steps, dtype=float)
    Y_stoc = np.zeros(time_steps, dtype=float)
    for t in range(1,time_steps+1):
        try:
            prod = lattice_old[:, :, 0] * lattice_old[:, :, 1]            # elementwise xy
            # vectorized Poisson draws
            births_x = rng.poisson(alpha * lattice_old[:, :, 0] * dt)
            kills_x = rng.poisson(gamma * prod * dt)
            np.minimum(kills_x,lattice[:,:,0],out=kills_x)
            births_y = rng.poisson(lam * dt, size=N)
            dup_y = rng.poisson(nu * prod * dt)
            deaths_y = rng.poisson(sigma * lattice_old[:, :, 1] * dt)
            np.minimum(deaths_y,lattice[:,:,1],out=deaths_y)
            # update lattice (vectorized)
            lattice[:, :, 0] += births_x - kills_x
            lattice[:, :, 1] += births_y + dup_y - deaths_y
            if Dx or Dy > 0:    # diffusion
                diffX = rng.poisson(Dx*lattice_old[:,:,0] * dt)
                diffY = rng.poisson(Dy*lattice_old[:,:,1] * dt)
                np.minimum(diffX,lattice_old[:,:,0],out=diffX)    # the maximum of individuals to diffuse out is the population of the site
                np.minimum(diffY,lattice_old[:,:,1],out=diffY)
                # Split each site's diff count into updown/leftright, then split each into left/right or up/down.
                updownX = rng.binomial(diffX,0.5)
                leftrightX = diffX - updownX
                upX = rng.binomial(updownX,0.5)
                downX = updownX - upX
                leftX = rng.binomial(leftrightX,0.5)
                rightX = leftrightX - leftX

                updownY = rng.binomial(diffY,0.5)
                leftrightY = diffY - updownY
                upY = rng.binomial(updownY,0.5)
                downY = updownY - upY
                leftY = rng.binomial(leftrightY,0.5)
                rightY = leftrightY - leftY
                # Add incoming migrants to neighbours on the circular lattice (roll shifts source -> destination)
                lattice[:, :, 0] += (
                    np.roll(leftX, -1, axis=1) + np.roll(rightX, +1, axis=1)
                    + np.roll(upX, -1, axis=0) + np.roll(downX, +1, axis=0)
                )
                lattice[:, :, 1] += (
                    np.roll(leftY, -1, axis=1) + np.roll(rightY, +1, axis=1)
                    + np.roll(upY, -1, axis=0) + np.roll(downY, +1, axis=0)
                )
                lattice[:,:,0] -= diffX   # take out the diffused individuals
                lattice[:,:,1] -= diffY
            np.maximum(lattice, 0, out=lattice,dtype=np.int64)         # clip negatives in-place (no negative populations)
            np.minimum(lattice,int(1e9),out=lattice,dtype=np.int64)
            lattice_old = lattice.copy()           # prepare for next step (must copy to avoid aliasing)
            X_stoc[t-1] = lattice[:, :, 0].mean()    # mean
            Y_stoc[t-1] = lattice[:, :, 1].mean()
            if mode == 2:
                densityx[t-1] = np.count_nonzero(lattice[:, :, 0]) / N**2
                densityy[t-1] = np.count_nonzero(lattice[:, :, 1]) / N**2
        except ValueError as e:
            print(f'ValueError:{e}')
            print(f'Maximum population in a site: x:{np.max(lattice[:,:,0])}; y:{np.max(lattice[:,:,1])}')
            break
    if mode == 1:
        return np.count_nonzero(lattice[:, :, 0]) / N**2
    elif mode == 2: 
        return densityx, densityy
    return X_stoc, Y_stoc

def lattice3(args,rng=None):
    """
    Vectorized stochastic lattice. Uses a numpy Generator for faster poisson draws.
    """
    N, time_steps, dt, par, mode = args
    if rng is None:
        rng = np.random.default_rng()
    alpha = par['alpha']        # pre-extract parameters (avoid repeated dict lookups)
    gamma = par['gamma']
    lam = par['lambda']
    nu = par['nu']
    sigma = par['sigma']
    Dx = par['Dx']
    Dy = par['Dy']
    lattice = np.zeros((N, N, N, 2), dtype=int)      # integer lattice for counts
    lattice[:, :, :, 0] = 3
    if Dx == 0 and Dy == 0:      # no diffusion
        lattice[:, :, :, 1] = lam/sigma
    else:
        lattice[:, :, :, 1] = rng.poisson(lam=1)
    if mode == 2:
        density = np.zeros((2,time_steps))
    lattice_old = lattice.copy()
    X_stoc = np.zeros(time_steps, dtype=float)
    Y_stoc = np.zeros(time_steps, dtype=float)
    for t in range(1,time_steps+1):
        prod = lattice_old[:, :, :, 0] * lattice_old[:, :, :, 1]            # elementwise xy
        # vectorized Poisson draws
        births_x = rng.poisson(alpha * lattice_old[:, :, :, 0] * dt)
        kills_x = rng.poisson(gamma * prod * dt)
        np.minimum(kills_x,lattice[:,:,:,0],out=kills_x)
        births_y = rng.poisson(lam * dt, size=N)
        dup_y = rng.poisson(nu * prod * dt)
        deaths_y = rng.poisson(sigma * lattice_old[:, :, :, 1] * dt)
        np.minimum(deaths_y,lattice[:,:,:,1],out=deaths_y)
        # update lattice (vectorized)
        lattice[:, :, :, 0] += births_x - kills_x
        lattice[:, :, :, 1] += births_y + dup_y - deaths_y
        if Dx or Dy > 0:    # diffusion
            diffX = rng.poisson(Dx*lattice_old[:,:,:,0] * dt)
            diffY = rng.poisson(Dy*lattice_old[:,:,:,1] * dt)
            np.minimum(diffX,lattice_old[:,:,:,0],out=diffX)    # the maximum of individuals to diffuse out is the population of the site
            np.minimum(diffY,lattice_old[:,:,:,1],out=diffY)
            # Split each site's diff count into updown/leftright/frontback, then split each into left/right, up/down or front/back.
            updownX = rng.binomial(diffX,1/3)
            leftrightX = rng.binomial(diffX-updownX,0.5) 
            frontbackX = diffX - updownX - leftrightX
            upX = rng.binomial(updownX,0.5)
            downX = updownX - upX
            leftX = rng.binomial(leftrightX,0.5)
            rightX = leftrightX - leftX
            frontX = rng.binomial(frontbackX,0.5)
            backX = frontbackX - frontX

            updownY = rng.binomial(diffY,1/3)
            leftrightY = rng.binomial(diffY-updownY,0.5) 
            frontbackY = diffY - updownY - leftrightY
            upY = rng.binomial(updownY,0.5)
            downY = updownY - upY
            leftY = rng.binomial(leftrightY,0.5)
            rightY = leftrightY - leftY
            frontY = rng.binomial(frontbackY,0.5)
            backY = frontbackY - frontY
            # Add incoming migrants to neighbours on the circular lattice (roll shifts source -> destination)
            lattice[:, :, :, 0] += (
                np.roll(frontX, -1, axis=2) + np.roll(backX, +1, axis=2)
                + np.roll(leftX, -1, axis=1) + np.roll(rightX, +1, axis=1)
                + np.roll(upX, -1, axis=0) + np.roll(downX, +1, axis=0)
            )
            lattice[:, :, :, 1] += (
                np.roll(frontY, -1, axis=2) + np.roll(backY, +1, axis=2)
                + np.roll(leftY, -1, axis=1) + np.roll(rightY, +1, axis=1)
                + np.roll(upY, -1, axis=0) + np.roll(downY, +1, axis=0)
            )
            lattice[:,:,:,0] -= diffX   # take out the diffused individuals
            lattice[:,:,:,1] -= diffY
        np.maximum(lattice, 0, out=lattice)         # clip negatives in-place (no negative populations)
        np.minimum(lattice,int(1e9),out=lattice,dtype=int)
        lattice_old = lattice.copy()           # prepare for next step (must copy to avoid aliasing)
        X_stoc[t-1] = lattice[:, :, :, 0].mean()    # mean
        Y_stoc[t-1] = lattice[:, :, :, 1].mean()
        if mode == 2:
            density[0,t-1] = np.count_nonzero(lattice[:, :, :, 0]) / N**3
            density[1,t-1] = np.count_nonzero(lattice[:, :, :, 1]) / N**3
    if mode == 1:
        return np.count_nonzero(lattice[:, :, :, 0]) / N**3
    elif mode == 2: 
        return density
    return X_stoc, Y_stoc

# Evaluating probability of disappearance
def _run_model_check(model,args):
    X_stoc, _ = model(args)
    return 1 if X_stoc[-1] == 0 else 0

def P_diss(model, params):
    """Evaluate statistically the probability of disappearance at the first pass near-zero.

    This optimized version:
    - minimizes dict lookups by using locals,
    - computes distances vectorized,
    - parallelizes independent stochastic realizations using ProcessPoolExecutor.
    """
    # copy base parameters to avoid mutating caller dict
    num_nu, N, time_steps, dt, par, n_iter, mode = params
    base_par = par.copy()
    alpha = base_par['alpha']
    gamma = base_par['gamma']
    lam = base_par['lambda']
    sigma = base_par['sigma']
    # nu sweep (reverse so distance ~ 1/nu is monotone)
    nu_values = np.linspace(1e-5, 1.5e-3, num_nu)[::-1]
    # Precompute distances between fixed points (vectorized)
    y_star_local = alpha / gamma            # x* = (sigma - lambda*gamma/alpha)/nu ; y* = alpha/gamma
    P_d = np.zeros((num_nu, 2), dtype=float)
    P_d[:,0] = np.linalg.norm(
        np.stack([np.zeros_like(nu_values),
                  np.full_like(nu_values, lam / sigma)]).T
        - np.stack([(sigma - lam * gamma / alpha) / nu_values,
                    np.full_like(nu_values, y_star_local)]).T,
        axis=1
    )
    # prepare parallel executor size (limit to avoid oversubscription)
    max_workers = min(os.cpu_count() or 1, n_iter, 8)    # 8 is an hard cap on the number of workes to avoid overwhelming the system. os.cpu_count() or 1 to avoid errors if cpu_count returns None, which can happen
    for i, nu in enumerate(nu_values):
        print(f'Distance #{i+1}...')
        par_local = base_par.copy()
        par_local['nu'] = nu            # use a fresh copy for this nu
        # build args for each independent run
        args = [(N, time_steps, dt, par_local, mode) for _ in range(n_iter)]
        # run in parallel and sum successes
        success = 0
        if n_iter == 1:
            success = _run_model_check(model,args[0])     # avoid executor overhead for single run
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as exe:
                for res in exe.map(_run_model_check, (model,args)):
                    success += res
        P_d[i,1] = success / n_iter

    return P_d

# Direct Percolation Analysis
def DP1d(args,rng=None,verb:int=0):
    """Direct percolation model on a 1D circular lattice.

    Args:
        N (int, optional): size of the lattice. Defaults to 10000.
        timesteps (int, optional): number of iterations. Defaults to 500.
        Pdis (float, optional): probability of disappearance. Defaults to 0.
        Pinv (float, optional): probability of invasion. Defaults to 0.
        rng (np.random.Generator, optional): NumPy random generator. Defaults to None.
        verb (int, optional): if > 0, it will return the filling fraction and prints the lattice at each timestep. Defaults to 0.

    Returns:
        float: filling fraction at the end of the simulation.
    """
    N, timesteps, Pdis, Pinv = args
    if rng is None:
        rng = np.random.default_rng()
    lattice = np.ones(N,dtype=np.int64)    # initialize the lattice with all 1s
    lattice_old = lattice.copy()
    if verb > 0:
        filling_history = np.zeros(timesteps,dtype=float)
    for t in range(timesteps):
        Mdis = rng.choice(2,N,p=np.array([Pdis,1-Pdis]))     # evaluate the Pdis for each lattice site
        # Evaluate invasion probabilty of a site to its neighbours (note that they're two independent extractions)
        MinvR = rng.choice(2,N,p=np.array([1-Pinv,Pinv]))     # evaluate the Pinv to the right for each lattice site 
        MinvL = rng.choice(2,N,p=np.array([1-Pinv,Pinv]))     # evaluate the Pinv to the left for each lattice site
        lattice *= Mdis        # disappearance step
        Rinv = lattice_old*MinvR     # only lattice sites which were occupied at the previous step (i.e. had a value of 1) can invade
        Linv = lattice_old*MinvL 
        # invasion step (note that we consider a circular lattice, since for np.roll elements that roll beyond the last position are re-introduced at the first
        lattice += np.roll(Rinv,+1) + np.roll(Linv,-1)
        lattice = (lattice > 0).astype(np.int64)     # clip the lattice to 0s and 1s
        lattice_old = lattice.copy()
        if verb > 0:
            print(f'Lattice:{lattice}')
            filling_history[t] = np.sum(lattice)/N
    if verb > 0:
        return filling_history
    return np.sum(lattice)/N

def DP2d(args,rng=None,verb:int=0):
    # anche qua è 'circolare'
    N, timesteps, Pdis, Pinv = args
    if rng is None:
        rng = np.random.default_rng()
    lattice = np.ones((N,N),dtype=np.int64)   # initialize the lattice with all 1s
    lattice_old = lattice.copy()
    if verb > 0:
        filling_history = np.zeros(timesteps, dtype=float)
    for t in range(timesteps):
        # Evaulate the probabilities out of the loop for better performance
        Mdis = rng.choice(2, size=(N, N), p=np.array([Pdis, 1 - Pdis]))
        Minv = rng.choice(2, size=(N, N, 4), p=np.array([1 - Pinv, Pinv]))  # 0:right,1:left,2:down,3:up
        lattice *= Mdis       # disappearance step
        lattice += (np.roll(lattice_old * Minv[:, :, 0], +1, axis=1)  # right invasion
                    + np.roll(lattice_old * Minv[:, :, 1], -1, axis=1)  # left
                    + np.roll(lattice_old * Minv[:, :, 2], +1, axis=0)  # down
                    + np.roll(lattice_old * Minv[:, :, 3], -1, axis=0))  # up
        lattice = (lattice > 0).astype(np.int64)     # clip the lattice to 0s and 1s
        lattice_old = lattice.copy()
        if verb > 0:
            print(lattice)
            filling_history[t] = np.sum(lattice)/N**2
    if verb > 0:
        return filling_history
    return np.sum(lattice)/N**2

def DP3d(args,rng=None,verb:int=0):
    # anche qua è 'circolare'
    N, timesteps, Pdis, Pinv = args
    if rng is None:
        rng = np.random.default_rng()
    lattice = np.ones((N,N,N),dtype=np.int64)   # initialize the lattice with all 1s
    lattice_old = lattice.copy()
    if verb > 0:
        filling_history = np.zeros(timesteps, dtype=float)
    for t in range(timesteps):
        # Evaulate the probabilities out of the loop for better performance
        Mdis = rng.choice(2, size=(N, N, N), p=np.array([Pdis, 1 - Pdis]))
        Minv = rng.choice(2, size=(N, N, N, 6), p=np.array([1 - Pinv, Pinv]))  # 0:right,1:left,2:down,3:up,4:in,5:out
        lattice *= Mdis       # disappearance step
        lattice += (np.roll(lattice_old * Minv[:, :, :, 0], +1, axis=1)  # right invasion
                    + np.roll(lattice_old * Minv[:, :, :, 1], -1, axis=1)  # left
                    + np.roll(lattice_old * Minv[:, :, :, 2], +1, axis=0)  # down
                    + np.roll(lattice_old * Minv[:, :, :, 3], -1, axis=0)  # up
                    + np.roll(lattice_old * Minv[:, :, :, 4], +1, axis=2)  # in
                    + np.roll(lattice_old * Minv[:, :, :, 5], -1, axis=2))  # out
        lattice = (lattice > 0).astype(np.int64)     # clip the lattice to 0s and 1s
        lattice_old = lattice.copy()
        if verb > 0:
            print(lattice)
            filling_history[t] = np.sum(lattice)/N**3
    if verb > 0:
        return filling_history
    return np.sum(lattice)/N**3
    
def filling_fraction_DP(model,Pspan:np.ndarray,params:list):
    """Function to evaluate the filling fraction for a DP model. Utilizes ProcessPollExecutor for parallel execution, leading to better performances.

    Args:
        model (_type_): DP model
        Pspan (np.ndarray): set of Pinv to test.
        N (int, optional): lattice size. Defaults to 10000.
        timesteps (int, optional): number of timesteps of each execution. Defaults to 500.
        Pdis (float, optional): probabiilty of disappearence. Defaults to 0.
        n_iter (int, optional): number of iterations for each set of parameters. Defaults to 50.

    Returns:
        _type_: _description_
    """
    N, timesteps, Pdis, n_iter = params
    F = np.stack([Pspan,np.zeros(len(Pspan))],dtype=float)
    max_workers = min(os.cpu_count() or 1,n_iter,8)
    i = 0
    for Pinv in tqdm(Pspan,token='8277133179:AAFE5QNsGn4rEAR3HzYtxmasGUarUWfbZwY',chat_id='1288694314'):
        args = [(N, timesteps, Pdis, Pinv)] * n_iter
        success = 0
        if n_iter == 1:
            success = model(args[0])
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as exe:
                for res in exe.map(model,args):
                    success += res
        F[1,i] = success/n_iter    
        i += 1        
    return F

def filling_fraction_ST(model,Pspan:np.ndarray,params:list):
    """Function to evaluate the filling fraction for a STOCHASTIC model. Utilizes ProcessPollExecutor for parallel execution, leading to better performances.

    Args:
        model (_type_): DP model
        Pspan (np.ndarray): set of Pinv to test.
        N (int, optional): lattice size. Defaults to 10000.
        timesteps (int, optional): number of timesteps of each execution. Defaults to 500.
        Pdis (float, optional): probabiilty of disappearence. Defaults to 0.
        n_iter (int, optional): number of iterations for each set of parameters. Defaults to 50.

    Returns:
        _type_: _description_
    """
    N, timesteps, dt, par, n_iter = params
    F = np.stack([Pspan,np.zeros(len(Pspan))],dtype=float)
    max_workers = min(os.cpu_count() or 1,n_iter,8)
    i = 0
    for Dx in tqdm(Pspan,token='8277133179:AAFE5QNsGn4rEAR3HzYtxmasGUarUWfbZwY',chat_id='1288694314'):
        local_par = par.copy()
        local_par['Dx'] = Dx
        args = [(N, timesteps, dt, local_par, 1)] * n_iter
        success = 0
        if n_iter == 1:
            success = model(args[0])
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as exe:
                for res in exe.map(model,args):
                    success += res
        F[1,i] = success/n_iter 
        i += 1           
    return F

def figure9(model,NSpan:np.ndarray,params:list):
    """Function to evaluate the filling fraction for different system sizes for a STOCHASTIC model. Utilizes ProcessPollExecutor for parallel execution, leading to better performances.

    Args:
        model (_type_): DP model
        Pspan (np.ndarray): set of Pinv to test.
        N (int, optional): lattice size. Defaults to 10000.
        timesteps (int, optional): number of timesteps of each execution. Defaults to 500.
        Pdis (float, optional): probabiilty of disappearence. Defaults to 0.
        n_iter (int, optional): number of iterations for each set of parameters. Defaults to 50.

    Returns:
        _type_: _description_
    """
    timesteps, dt, par, n_iter, mode = params
    F = np.stack([NSpan,np.zeros(NSpan.shape[0])],dtype=float)
    max_workers = min(os.cpu_count() or 1,n_iter,8)
    i = 0
    for N in tqdm(NSpan,token='8277133179:AAFE5QNsGn4rEAR3HzYtxmasGUarUWfbZwY',chat_id='1288694314'):
        args = [(N, timesteps, dt, par, mode)] * n_iter
        success = 0
        if n_iter == 1:
            success = model(args[0])
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as exe:
                for res in exe.map(model,args):
                    success += res
        F[1,i] = success/n_iter 
        i += 1           
    return F