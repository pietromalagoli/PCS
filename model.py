import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import concurrent.futures

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
            
def check_sign(arr:np.ndarray):
    arr[arr<0] = 0.

def add_par_box(par):
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
    ax.text(0.98, 0.60, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)

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
    N, time_steps, par, mode = args
    if rng is None:
        rng = np.random.default_rng()
    alpha = par['alpha']        # pre-extract parameters (avoid repeated dict lookups)
    gamma = par['gamma']
    lam = par['lambda']
    nu = par['nu']
    sigma = par['sigma']
    Dx = par['Dx']
    Dy = par['Dy']
    lattice = np.zeros((N, 2), dtype=np.int64)      # integer lattice for counts
    lattice[:, 0] = 3       # initialization
    if Dx and Dy == 0:      # no diffusion
        lattice[:, 1] = 1
    else:
        lattice[:, 1] = rng.poisson(lam=1)
    if mode == 2:  
        density = np.zeros(time_steps)
    lattice_old = lattice.copy()
    X_stoc = np.zeros(time_steps, dtype=float)
    Y_stoc = np.zeros(time_steps, dtype=float)
    for t in range(time_steps):
        prod = lattice_old[:, 0] * lattice_old[:, 1]            # elementwise xy
        # vectorized Poisson draws
        births_x = rng.poisson(alpha * lattice_old[:, 0])
        kills_x = rng.poisson(gamma * prod)
        births_y = rng.poisson(lam, size=N)
        dup_y = rng.poisson(nu * prod)
        deaths_y = rng.poisson(sigma * lattice_old[:, 1])
        # update lattice (vectorized)
        lattice[:, 0] = births_x - kills_x
        lattice[:, 1] = births_y + dup_y - deaths_y
        if Dx or Dy > 0:    # diffusion
            diffX = rng.poisson(np.sqrt(Dx*lattice_old[:,0]))
            diffY = rng.poisson(np.sqrt(Dy*lattice_old[:,1]))
            np.minimum(diffX,lattice_old[:,0],out=diffX)    # the maximum of individuals to diffuse out is the population of the site
            np.minimum(diffY,lattice_old[:,1],out=diffY)
            diffX = diffX // 2      # divide the # of individuals to diffuse from each site by the # of neighbours
            diffY = diffY // 2      # approximate down
            lattice[:,0] -= diffX   # take out the diffused individuals
            lattice[:,1] -= diffY
            lattice[:,0] += np.roll(diffX,+1) + np.roll(diffX,-1)  # add the diffused individuals
            lattice[:,1] += np.roll(diffY,+1) + np.roll(diffY,-1)  
        np.maximum(lattice, 0, out=lattice)         # clip negatives in-place (no negative populations)
        lattice_old = lattice.copy()           # prepare for next step (must copy to avoid aliasing)
        # mean
        X_stoc[t] = lattice[:, 0].mean()
        Y_stoc[t] = lattice[:, 1].mean()
        if mode == 2:
            density[t] = np.count_nonzero(lattice[:, 0]) / N
    if mode == 1:
        return np.count_nonzero(lattice[:, 0]) / N
    elif mode == 2: 
        return density
    return X_stoc, Y_stoc

def lattice2(args,rng=None):
    """
    Vectorized stochastic lattice. Uses a numpy Generator for faster poisson draws.
    """
    N, time_steps, par, mode = args
    if rng is None:
        rng = np.random.default_rng()
    alpha = par['alpha']        # pre-extract parameters (avoid repeated dict lookups)
    gamma = par['gamma']
    lam = par['lambda']
    nu = par['nu']
    sigma = par['sigma']
    Dx = par['Dx']
    Dy = par['Dy']
    lattice = np.zeros((N, N, 2), dtype=np.int64)      # integer lattice for counts
    lattice[:, :, 0] = 3
    if Dx and Dy == 0:      # no diffusion
        lattice[:, :, 1] = 1
    else:
        lattice[:, :, 1] = rng.poisson(lam=1)
    if mode == 2:
        density = np.zeros(time_steps)
    lattice_old = lattice.copy()
    X_stoc = np.zeros(time_steps, dtype=float)
    Y_stoc = np.zeros(time_steps, dtype=float)
    for t in range(time_steps):
        prod = lattice_old[:, :, 0] * lattice_old[:, :, 1]            # elementwise xy
        # vectorized Poisson draws
        births_x = rng.poisson(alpha * lattice_old[:, :, 0])
        kills_x = rng.poisson(np.sqrt((gamma * prod)**2))
        births_y = rng.poisson(lam, size=N)
        dup_y = rng.poisson(np.sqrt((nu * prod)**2))
        deaths_y = rng.poisson(sigma * lattice_old[:, :, 1])
        # update lattice (vectorized)
        lattice[:, :, 0] = births_x - kills_x
        lattice[:, :, 1] = births_y + dup_y - deaths_y
        if Dx or Dy > 0:    # diffusion
            diffX = rng.poisson(np.sqrt((Dx*lattice_old[:,:,0])**2))
            diffY = rng.poisson(np.sqrt((Dy*lattice_old[:,:,1])**2))
            np.minimum(diffX,lattice_old[:,:,0],out=diffX)    # the maximum of individuals to diffuse out is the population of the site
            np.minimum(diffY,lattice_old[:,:,1],out=diffY)
            diffX = diffX // 4      # divide the # of individuals to diffuse from each site by the # of neighbours
            diffY = diffY // 4      # approximate down
            lattice[:,:,0] -= diffX   # take out the diffused individuals
            lattice[:,:,1] -= diffY
            lattice[:,:,0] += np.roll(diffX,+1,axis=0) + np.roll(diffX,-1,axis=0)  # add the diffused individuals 
            lattice[:,:,0] += np.roll(diffX,+1,axis=1) + np.roll(diffX,-1,axis=1)  
            lattice[:,:,1] += np.roll(diffY,+1,axis=0) + np.roll(diffY,-1,axis=0)  
            lattice[:,:,1] += np.roll(diffY,+1,axis=1) + np.roll(diffY,-1,axis=1)  
        np.maximum(lattice, 0, out=lattice)         # clip negatives in-place (no negative populations)
        lattice_old = lattice.copy()           # prepare for next step (must copy to avoid aliasing)
        X_stoc[t] = lattice[:, :, 0].mean()    # mean
        Y_stoc[t] = lattice[:, :, 1].mean()
        if mode == 2:
            density[t] = np.count_nonzero(lattice[:, :, 0]) / N**2
    if mode == 1:
        return np.count_nonzero(lattice[:, :, 0]) / N**2
    elif mode == 2: 
        return density
    return X_stoc, Y_stoc

def lattice3(args,rng=None):
    """
    Vectorized stochastic lattice. Uses a numpy Generator for faster poisson draws.
    """
    N, time_steps, par, mode = args
    if rng is None:
        rng = np.random.default_rng()
    alpha = par['alpha']        # pre-extract parameters (avoid repeated dict lookups)
    gamma = par['gamma']
    lam = par['lambda']
    nu = par['nu']
    sigma = par['sigma']
    Dx = par['Dx']
    Dy = par['Dy']
    lattice = np.zeros((N, N, N, 2), dtype=np.int64)      # integer lattice for counts
    lattice[:, :, 0] = 3
    if Dx and Dy == 0:      # no diffusion
        lattice[:, :, 1] = 1
    else:
        lattice[:, :, 1] = rng.poisson(lam=1)
    if mode == 2:
        density = np.zeros(time_steps)
    lattice_old = lattice.copy()
    X_stoc = np.zeros(time_steps, dtype=float)
    Y_stoc = np.zeros(time_steps, dtype=float)
    for t in range(time_steps):
        prod = lattice_old[:, :, :, 0] * lattice_old[:, :, :, 1]            # elementwise xy
        # vectorized Poisson draws
        births_x = rng.poisson(alpha * lattice_old[:, :, :, 0])
        kills_x = rng.poisson(gamma * prod)
        births_y = rng.poisson(lam, size=N)
        dup_y = rng.poisson(nu * prod)
        deaths_y = rng.poisson(sigma * lattice_old[:, :, :, 1])
        # update lattice (vectorized)
        lattice[:, :, :, 0] = births_x - kills_x
        lattice[:, :, :, 1] = births_y + dup_y - deaths_y
        if Dx or Dy > 0:    # diffusion
            diffX = rng.poisson(np.sqrt((Dx*lattice_old[:,:,:,0])**2))
            diffY = rng.poisson(np.sqrt((Dy*lattice_old[:,:,:,1])**2))
            np.minimum(diffX,lattice_old[:,:,:,0],out=diffX)    # the maximum of individuals to diffuse out is the population of the site
            np.minimum(diffY,lattice_old[:,:,:,1],out=diffY)
            diffX = diffX // 6      # divide the # of individuals to diffuse from each site by the # of neighbours
            diffY = diffY // 6      # approximate down
            lattice[:,:,:,0] -= diffX   # take out the diffused individuals
            lattice[:,:,:,1] -= diffY
            lattice[:,:,:,0] += np.roll(diffX,+1,axis=0) + np.roll(diffX,-1,axis=0)  # add the diffused individuals 
            lattice[:,:,:,0] += np.roll(diffX,+1,axis=1) + np.roll(diffX,-1,axis=1)  
            lattice[:,:,:,0] += np.roll(diffX,+1,axis=2) + np.roll(diffX,-1,axis=2)  
            lattice[:,:,:,1] += np.roll(diffY,+1,axis=0) + np.roll(diffY,-1,axis=0)  
            lattice[:,:,:,1] += np.roll(diffY,+1,axis=1) + np.roll(diffY,-1,axis=1)  
            lattice[:,:,:,1] += np.roll(diffY,+1,axis=2) + np.roll(diffY,-1,axis=2)  
        np.maximum(lattice, 0, out=lattice)         # clip negatives in-place (no negative populations)
        lattice_old = lattice.copy()           # prepare for next step (must copy to avoid aliasing)
        X_stoc[t] = lattice[:, :, :, 0].mean()    # mean
        Y_stoc[t] = lattice[:, :, :, 1].mean()
        if mode == 2:
            density[t] = np.count_nonzero(lattice[:, :, :, 0]) / N**3
    if mode == 1:
        return np.count_nonzero(lattice[:, :, :, 0]) / N**3
    elif mode == 2: 
        return density
    return X_stoc, Y_stoc

# Evaluating probability of disappearance
def _run_model_check(args):
    model, N, time_steps, par = args
    X_stoc, _ = model(N=N, time_steps=time_steps, par=par)
    return 1 if X_stoc[-1] == 0 else 0

def P_diss(model, num_nu:int, n_iter:int, N:int, time_steps:int, par:dict):
    """Evaluate statistically the probability of disappearance at the first pass near-zero.

    This optimized version:
    - minimizes dict lookups by using locals,
    - computes distances vectorized,
    - parallelizes independent stochastic realizations using ProcessPoolExecutor.
    """
    # copy base parameters to avoid mutating caller dict
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
        args = [(model, N, time_steps, par_local) for _ in range(n_iter)]
        # run in parallel and sum successes
        success = 0
        if n_iter == 1:
            success = _run_model_check(args[0])     # avoid executor overhead for single run
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as exe:
                for res in exe.map(_run_model_check, args):
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
    # Evaulate the probabilities out of the loop for better performance
    Mdis = rng.choice(2,(N,timesteps),p=np.array([Pdis,1-Pdis]))     # evaluate the Pdis for each lattice site
    # Evaluate invasion probabilty of a site to its neighbours (note that they're two independent extractions)
    MinvR = rng.choice(2,(N,timesteps),p=np.array([1-Pinv,Pinv]))     # evaluate the Pinv to the right for each lattice site 
    MinvL = rng.choice(2,(N,timesteps),p=np.array([1-Pinv,Pinv]))     # evaluate the Pinv to the left for each lattice site
    for t in range(timesteps):
        lattice *= Mdis[:,t]        # disappearance step
        Rinv = lattice_old*MinvR[:,t]     # only lattice sites which were occupied at the previous step (i.e. had a value of 1) can invade
        Linv = lattice_old*MinvL[:,t] 
        # invasion step (note that we consider a circular lattice, since for np.roll elements that roll beyond the last position are re-introduced at the first
        lattice = (lattice+np.roll(Rinv,+1)) + (lattice+np.roll(Linv,-1))    
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
    # Evaulate the probabilities out of the loop for better performance
    Mdis = rng.choice(2, size=(N, N, timesteps), p=[Pdis, 1 - Pdis])
    Minv = rng.choice(2, size=(N, N, 4, timesteps), p=[1 - Pinv, Pinv])  # 0:right,1:left,2:down,3:up
    for t in range(timesteps):
        lattice *= Mdis[:, :, t]       # disappearance step
        lattice += np.roll(lattice_old * Minv[:, :, 0, t], +1, axis=1)  # right invasion
        lattice += np.roll(lattice_old * Minv[:, :, 1, t], -1, axis=1)  # left
        lattice += np.roll(lattice_old * Minv[:, :, 2, t], +1, axis=0)  # down
        lattice += np.roll(lattice_old * Minv[:, :, 3, t], -1, axis=0)  # up
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
    # Evaulate the probabilities out of the loop for better performance
    Mdis = rng.choice(2, size=(N, N, N, timesteps), p=[Pdis, 1 - Pdis])
    Minv = rng.choice(2, size=(N, N, N, 6, timesteps), p=[1 - Pinv, Pinv])  # 0:right,1:left,2:down,3:up,4:in,5:out
    for t in range(timesteps):
        lattice *= Mdis[:, :, :, t]       # disappearance step
        lattice += np.roll(lattice_old * Minv[:, :, :, 0, t], +1, axis=1)  # right invasion
        lattice += np.roll(lattice_old * Minv[:, :, :, 1, t], -1, axis=1)  # left
        lattice += np.roll(lattice_old * Minv[:, :, :, 2, t], +1, axis=0)  # down
        lattice += np.roll(lattice_old * Minv[:, :, :, 3, t], -1, axis=0)  # up
        lattice += np.roll(lattice_old * Minv[:, :, :, 4, t], +1, axis=2)  # in
        lattice += np.roll(lattice_old * Minv[:, :, :, 5, t], -1, axis=2)  # out
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
    F = np.stack([Pspan,np.zeros(Pspan.shape[0])],dtype=float)
    max_workers = min(os.cpu_count() or 1,n_iter,8)
    for i,Pinv in enumerate(Pspan):
        if i & 10 == 0:
            print(f'DP: set#{i}')
        args = [(N, timesteps, Pdis, Pinv)] * n_iter
        success = 0
        if n_iter == 1:
            success = model(args[0])
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as exe:
                for res in exe.map(model,args):
                    success += res
        F[1,i] = success/n_iter            
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
    N, timesteps, par, n_iter, fill = params
    F = np.stack([Pspan,np.zeros(Pspan.shape[0])],dtype=float)
    max_workers = min(os.cpu_count() or 1,n_iter,8)
    for i,Pinv in enumerate(Pspan):
        if i % 10 == 0:
            print(f'ST: set#{i}')
        local_par = par.copy()
        local_par['Dx'],local_par['Dy'] = [Pinv,Pinv]
        args = [(N, timesteps, local_par, fill)] * n_iter
        success = 0
        if n_iter == 1:
            success = model(args[0])
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as exe:
                for res in exe.map(model,args):
                    success += res
        F[1,i] = success/n_iter            
    return F