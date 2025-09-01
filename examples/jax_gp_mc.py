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
    mean = mu
    return mean

def interpolate(nodes, vals, lengthscale, x):
    # Create a GP interpolation of log10(Pζ) and then convert back to linear scale.
    nodes = (nodes - left_node) / (right_node - left_node)
    vals_mean = jnp.mean(vals)
    vals_std = jnp.std(vals)
    vals = (vals - vals_mean) / vals_std

    nodes = jnp.reshape(nodes, (-1, 1))
    vals = jnp.reshape(vals, (-1, 1))
    x_flat = jnp.reshape(x, (-1, 1))
    x_flat = (x_flat - left_node) / (right_node - left_node)

    k11 = rbf_kernel(nodes,nodes,10**lengthscale,outputscale=1.0) + 1e-12 * jnp.eye(len(nodes))
    k11_cho = jnp.linalg.cholesky(k11)
    k12 = rbf_kernel(nodes,x_flat,10**lengthscale,outputscale=1.0)
    res = get_mean_from_cho(k11_cho,k12,vals)
    res = res*vals_std + vals_mean
    res = jnp.power(10,res)
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

#############################
# Main function: Sampling and Postprocessing
#############################

def main():
    global free_nodes, left_node, right_node, y_min, y_max, y_mins, y_maxs, l_min, l_max
    global gwb_calculator, frequencies, Omegas, cov, p_arr, pz_amp

    model = str(sys.argv[1])
    num_nodes = int(sys.argv[2])
    realization_index = 100 * int(sys.argv[3]) if len(sys.argv) > 3 else None

    # Load the gravitational wave background data.
    data = np.load(f'./{model}_data.npz')
    frequencies = data['k']
    Omegas_mean = data['gw']
    cov = data['cov']
    p_arr = data['p_arr']
    pz_amp = data['pz_amp']

    if realization_index is not None:
        print(f"Generating realization {realization_index} of Omegas.")
        rstate = np.random.default_rng(realization_index)
        Omegas = rstate.multivariate_normal(Omegas_mean, cov)
    else:
        print("Using mean Omegas.")
        Omegas = Omegas_mean

    # Set up internal momenta for the OmegaGWjax calculator.
    s = jnp.linspace(0, 1, 15)
    t = jnp.logspace(-5, 5, 200)
    t_expanded = jnp.expand_dims(t, axis=-1)
    t = jnp.repeat(t_expanded, len(frequencies), axis=-1)

    # Create the gravitational wave background calculator.
    gwb_calculator = OmegaGWjax(s=s, t=t, f=frequencies, norm="RD", jit=True)

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

    # Set up the sampler.
    ndim = free_nodes + num_nodes + 1  # +1 for the lengthscale
    if realization_index is not None:
        filepath = f'./results/nautilus_{model}_{num_nodes}_gp_realization_{realization_index}.h5'
    else:
        filepath = f'./results/nautilus_{model}_{num_nodes}_gp.h5'
    
    sampler = Sampler(prior, likelihood, ndim, pass_dict=False, vectorized=True
                                            ,pool=None,filepath=filepath) 

    start = time.time()
    sampler.run(verbose=True, f_live=0.01, n_like_max=5e6)
    end = time.time()
    print('Time taken: {:.2f} s'.format(end - start))
    print('log Z: {:.2f}'.format(sampler.log_z))

    # Retrieve posterior samples.
    samples, logl, logwt = sampler.posterior()
    
    save_data = {'samples': samples, 'logl': logl, 'logwt': logwt, 'logz': sampler.log_z}
    
    if realization_index is not None:
        npz_filepath = f'./results/nautilus_{model}_{num_nodes}_gp_realization_{realization_index}.npz'
        save_data['omegas'] = Omegas
    else:
        npz_filepath = f'./results/nautilus_{model}_{num_nodes}_gp.npz'
        
    np.savez(npz_filepath, **save_data)
    print(samples.shape)
    print(logl.shape)
    print(logwt.shape)

if __name__ == '__main__':
    main()
