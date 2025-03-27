import os
import sys
import time
import warnings
import math
import numpy as np
from jax import config, vmap, jit
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from omega_gw_jax import OmegaGWjax
from getdist import plots, MCSamples, loadMCSamples
from interpax import CubicSpline
from functools import partial
from nautilus import Sampler
from mpi4py.futures import MPIPoolExecutor

# Set matplotlib parameters
font = {'size': 16, 'family': 'serif'}
axislabelfontsize = 'large'
matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=True)
matplotlib.rc('legend', fontsize=16)

# Global variables (will be set in main)
free_nodes = None
left_node = None
right_node = None
y_min = None
y_max = None
y_mins = None
y_maxs = None
l_min = None
l_max = None
gwb_calculator = None
frequencies = None
Omegas = None
cov = None
p_arr = None
pz_amp = None

#############################
# Intermediate Functions
#############################

def split_vmap(func,input_arrays,batch_size=8):
    """
    Utility to split vmap over a function taking multiple arrays as input into multiple chunks, useful for reducing memory usage.
    """
    num_inputs = input_arrays[0].shape[0]
    num_batches = (num_inputs + batch_size - 1 ) // batch_size
    batch_idxs = [jnp.arange( i*batch_size, min( (i+1)*batch_size,num_inputs  )) for i in range(num_batches)]
    res = [vmap(func)(*tuple([arr[idx] for arr in input_arrays])) for idx in batch_idxs]
    nres = len(res[0])
    # now combine results across batches and function outputs to return a tuple (num_outputs, num_inputs, ...)
    results = tuple( jnp.concatenate([x[i] for x in res]) for i in range(nres))
    return results

def prior_1D(cube):
    # Order and transform nodes to be in the correct range, from Polychord SortedUniformPrior
    params = cube.copy()
    x = params[:free_nodes]
    N = free_nodes
    t_arr = np.zeros(N)
    t_arr[N - 1] = x[N - 1] ** (1.0 / N)
    for n in range(N - 2, -1, -1):
        t_arr[n] = x[n] ** (1.0 / (n + 1)) * t_arr[n + 1]
    xs = t_arr * (right_node - left_node) + left_node
    ys = params[free_nodes:]
    ys = ys * (y_max - y_min) + y_min
    return np.concatenate([xs, ys])

@jit
def prior(cube):
    """
    Transforms the input cube from [0,1] uniform parameters to the desired prior space.
    This vectorized version supports cube being either a 1D array of shape (nd,)
    or a 2D array of shape (Npoints, nd).
    """
    cube = jnp.atleast_2d(cube.copy())
    l = cube[:,0]*(l_max - l_min) + l_min
    l = jnp.reshape(l, (-1, 1)) 
    N = free_nodes  # Number of x parameters
    x = cube[:, 1:N+1]
    exponents = 1.0 / jnp.arange(1, N + 1)
    y_vals = x ** exponents  # shape (Npoints, free_nodes)
    t_arr = jnp.cumprod(y_vals[:, ::-1], axis=1)[:, ::-1]
    xs = t_arr * (right_node - left_node) + left_node
    ys = cube[:, N+1:]
    ys = ys * (y_maxs[None, :] - y_mins[None, :]) + y_mins[None, :]
    # shapes
    # print(f"xs: {xs.shape}, ys: {ys.shape}, l: {l.shape}")
    return jnp.concatenate([ l, xs, ys], axis=1)

def dist_sq(x, y):
    """
    Compute squared Euclidean distance between two points x, y. 
    If x is n1 x d and y is n2 x d returns a n1 x n2 matrix of distancess.
    """
    return jnp.sum(jnp.square(x[:,None,:] - y[None,:,:]),axis=-1) 

def rbf_kernel(xa,
               xb,
               lengthscales,
               outputscale,
               ): 
    """
    The RBF kernel
    """
    sq_dist = dist_sq(xa/lengthscales,xb/lengthscales) 
    sq_dist = jnp.exp(-0.5*sq_dist)
    k = outputscale*sq_dist
    return k

def get_mean_from_cho(k11_cho,k12,train_y):
    mu = jnp.matmul(jnp.transpose(k12),cho_solve((k11_cho,True),train_y)) # can also store alphas
    mean = mu #.squeeze(-1) #mu[:,0]  
    return mean

def interpolate(nodes, vals, lengthscale, x):
    # Create a GP interpolation of log10(Pζ) and then convert back to linear scale.
    
    # convert nodes to [0,1]
    nodes = (nodes - left_node) / (right_node - left_node)
    # convert vals to zero mean and unit variance
    vals_mean = jnp.mean(vals)
    vals_std = jnp.std(vals)
    vals = (vals - vals_mean) / vals_std

    nodes = jnp.reshape(nodes, (-1, 1))  # Ensure shape (n_nodes, 1)
    vals = jnp.reshape(vals, (-1, 1))  # Ensure shape (n_nodes, 1)
    x_flat = jnp.reshape(x, (-1, 1))  # Ensure shape (n_points, 1)
    x_flat = (x_flat - left_node) / (right_node - left_node)  # Normalize x_flat to [0,1]

    # print(f"nodes: {nodes.shape}, vals: {vals.shape}, x: {x.shape}, x_flat: {x_flat.shape}")
    k11 = rbf_kernel(nodes,nodes,10**lengthscale,outputscale=1.0) + 1e-12 * jnp.eye(len(nodes))
    k11_cho = jnp.linalg.cholesky(k11)
    k12 = rbf_kernel(nodes,x_flat,10**lengthscale,outputscale=1.0)
    res = get_mean_from_cho(k11_cho,k12,vals)
    res = res*vals_std + vals_mean
    res = jnp.power(10,res)
    # print(f"res shape {res.shape}")
    res = jnp.where(x_flat < 0., 0., res)
    res = jnp.where(x_flat > 1., 0., res)
    res = res.reshape(x.shape)
    return res

def get_gwb(nodes, vals, lengthscales):
    # Given nodes and values, create a function for Pζ and compute Ω_GW.
    pf = lambda k: interpolate(nodes, vals,lengthscales, jnp.log10(k))
    omegagw = gwb_calculator(pf, frequencies)
    return (omegagw,)

# JIT compile get_gwb for speed.
get_gwb_func = jit(get_gwb)

def likelihood(params):
    params = jnp.atleast_2d(params)
    lengthscales = params[:, 0]
    nodes = params[:, 1:free_nodes+1]
    # Pad nodes with fixed endpoints
    nodes = jnp.pad(nodes, ((0, 0), (1, 1)), 'constant',
                      constant_values=((0, 0), (left_node, right_node)))
    vals = params[:, free_nodes+1:]
    omegagw = split_vmap(get_gwb_func, (nodes, vals,lengthscales), batch_size=100)[0]
    diff = omegagw - Omegas
    sol = np.linalg.solve(cov, diff.T).T
    res = -0.5 * np.sum(diff * sol, axis=1)
    res = np.where(jnp.isnan(res), -1e10, res)
    res = np.where(res < -1e10, -1e10, res)
    return res

def resample_equal(samples, logl, logwt, rstate):
    # Resample samples to obtain equal weights.
    wt = np.exp(logwt)
    weights = wt / wt.sum()
    cumulative_sum = np.cumsum(weights)
    cumulative_sum /= cumulative_sum[-1]
    nsamples = len(weights)
    positions = (rstate.random() + np.arange(nsamples)) / nsamples
    idx = np.zeros(nsamples, dtype=int)
    i, j = 0, 0
    while i < nsamples:
        if positions[i] < cumulative_sum[j]:
            idx[i] = j
            i += 1
        else:
            j += 1
    perm = rstate.permutation(nsamples)
    resampled_samples = samples[idx][perm]
    resampled_logl = logl[idx][perm]
    return resampled_samples, resampled_logl

def plot_functional_posterior(vals=[], k_arr=[], intervals=[99.7, 95., 68.],
                              ylabels=[r'$P_{\zeta}$', r'$\Omega_{\rm GW}$'],
                              aspect_ratio=(6, 5),
                              interval_cols=[('#006FED', 0.2), ('#006FED', 0.4), ('#006FED', 0.6)]):
    # Plot the posterior of y = f(k|x) using symmetric credible intervals.
    nfuncs = len(vals)
    fig, ax = plt.subplots(1, nfuncs, figsize=(aspect_ratio[0] * nfuncs, aspect_ratio[1]), constrained_layout=True)
    if nfuncs == 1:
        ax = [ax]
    for i, val in enumerate(vals):
        for j, interval in enumerate(intervals):
            y_low, y_high = np.percentile(val, [50 - interval / 2, 50 + interval / 2], axis=0)
            ax[i].fill_between(k_arr[i], y_low, y_high, color=interval_cols[j][0], alpha=interval_cols[j][1])
        ax[i].plot(k_arr[i], np.median(val, axis=0), color='#006FED', lw=2.5)
        ax[i].set_ylabel(ylabels[i])
    for x in ax:
        x.set(xscale='log', yscale='log', xlabel=r'$f\,{\rm [Hz]}$')
    return fig, ax

def get_pz_omega(nodes, vals):
    # Given nodes and vals, compute Pζ and Ω_GW.
    pf = lambda k: interpolate(nodes, vals, jnp.log10(k))
    pz_amps = pf(p_arr)
    gwb_res = gwb_calculator(pf, frequencies)
    return (pz_amps, gwb_res)

#############################
# Main function: Sampling and Postprocessing
#############################

def main():
    global free_nodes, left_node, right_node, y_min, y_max, y_mins, y_maxs, l_min, l_max
    global gwb_calculator, frequencies, Omegas, cov, p_arr, pz_amp

    model = str(sys.argv[1])
    # Load the gravitational wave background data.
    data = np.load(f'./{model}_data.npz')
    frequencies = data['k']
    Omegas = data['gw']
    cov = data['cov']
    p_arr = data['p_arr']
    pz_amp = data['pz_amp']

    # Set up internal momenta for the OmegaGWjax calculator.
    s = jnp.linspace(0, 1, 15)  # rescaled internal momentum
    t = jnp.logspace(-5, 5, 200)  # rescaled internal momentum
    t_expanded = jnp.expand_dims(t, axis=-1)
    t = jnp.repeat(t_expanded, len(frequencies), axis=-1)

    # Create the gravitational wave background calculator.
    gwb_calculator = OmegaGWjax(s=s, t=t, f=frequencies, norm="RD", jit=True)

    # Parse the number of nodes from command line arguments.
    num_nodes = int(sys.argv[2])
    print(f"Running inference with number of nodes: {num_nodes}, free nodes: {num_nodes - 2}")
    free_nodes = num_nodes - 2

    # Set the GP lengthscale bounds
    l_min = -2.
    l_max = 1.

    # Set the range for the x (log10) nodes using the data.
    pk_min, pk_max = min(p_arr), max(p_arr)
    left_node = np.log10(pk_min)
    right_node = np.log10(pk_max)

    # Set the y range for the interpolation.
    y_max = -1.
    y_min = -8.
    y_mins = np.array(num_nodes * [y_min])
    y_maxs = np.array(num_nodes * [y_max])

    # TESTING
    # np.random.seed(100)
    # test_y = np.random.uniform(y_min, y_max, num_nodes)
    # test_x = np.linspace(left_node, right_node, 6)
    # print(f"test_x: {test_x}")
    # print(f"test_y: {test_y}")
    # l = 0.
    # pz_vals = interpolate(test_x, test_y, l, jnp.log10(p_arr))
    # print(f"pz_vals: {pz_vals}")    
    # plt.loglog(p_arr, pz_vals)
    # plt.scatter(10**test_x, 10**test_y, color='r', label='nodes')
    # plt.show()



    # Set up the sampler.
    ndim = free_nodes + num_nodes + 1  # +1 for the lengthscale
    sampler = Sampler(prior, likelihood, ndim, pass_dict=False, vectorized=True
                                            ,pool=(None,4),filepath=f'./results/nautilus_{model}_{num_nodes}_gp.h5') 

    start = time.time()
    sampler.run(verbose=True, f_live=0.01, n_like_max=5e6)#, n_eff=2000*ndim)
    end = time.time()
    print('Time taken: {:.2f} s'.format(end - start))
    print('log Z: {:.2f}'.format(sampler.log_z))

    # Retrieve posterior samples.
    samples, logl, logwt = sampler.posterior()
    np.savez(f'./results/nautilus_{model}_{num_nodes}_gp_nodes.npz', samples=samples, logl=logl, logwt=logwt, logz=sampler.log_z)
    print(samples.shape)
    print(logl.shape)
    print(logwt.shape)

if __name__ == '__main__':
    main()
