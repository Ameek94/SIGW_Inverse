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
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from omega_gw_jax import OmegaGWjax
from getdist import plots, MCSamples, loadMCSamples
from interpax import CubicSpline
from functools import partial
import dynesty

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

def prior_transform(cube):
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

def interpolate(nodes, vals, x):
    # Create a cubic spline interpolation of log10(Pζ) and then convert back to linear scale.
    res = jnp.power(10, jnp.interp(x, nodes, vals))
    res = jnp.where(x < left_node, 0, res)
    res = jnp.where(x > right_node, 0, res)
    return res

def get_gwb(nodes, vals):
    # Given nodes and values, create a function for Pζ and compute Ω_GW.
    pf = lambda k: interpolate(nodes, vals, jnp.log10(k))
    omegagw = gwb_calculator(pf, frequencies)
    return (omegagw,)

# JIT compile get_gwb for speed.
get_gwb_func = jit(get_gwb)

@jit
def log_likelihood_jax(params):
    params = jnp.atleast_2d(params)
    nodes = params[:, :free_nodes]
    # Pad nodes with fixed endpoints
    nodes = jnp.pad(nodes, ((0, 0), (1, 1)), 'constant',
                      constant_values=((0, 0), (left_node, right_node)))
    vals = params[:, free_nodes:]
    omegagw = split_vmap(get_gwb_func, (nodes, vals), batch_size=100)[0]
    diff = omegagw - Omegas
    sol =jnp.linalg.solve(cov, diff.T).T
    res = -0.5 * jnp.sum(diff * sol, axis=1)
    res = jnp.where(jnp.isnan(res), -1e10, res)
    res = jnp.where(res < -1e10, -1e10, res)
    return res

def log_likelihood(params):
    return log_likelihood_jax(params)[0]

#############################
# Main function: Sampling and Postprocessing
#############################

def main():
    global free_nodes, left_node, right_node, y_min, y_max, y_mins, y_maxs
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
    s = jnp.linspace(0, 1, 15)  # rescaled internal momentum
    t = jnp.logspace(-5, 5, 200)  # rescaled internal momentum
    t_expanded = jnp.expand_dims(t, axis=-1)
    t = jnp.repeat(t_expanded, len(frequencies), axis=-1)

    # Create the gravitational wave background calculator.
    gwb_calculator = OmegaGWjax(s=s, t=t, f=frequencies, norm="RD", jit=True)

    # Parse the number of nodes from command line arguments.
    print(f"Running inference with number of nodes: {num_nodes}, free nodes: {num_nodes - 2}")
    free_nodes = num_nodes - 2

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
    ndim = free_nodes + num_nodes
    nlive = 3000
    sampler = dynesty.DynamicNestedSampler(log_likelihood, prior_transform, ndim, nlive=nlive)
    start = time.time()
    sampler.run_nested(dlogz_init=0.01)
    end = time.time()

    results = sampler.results
    
    print('Time taken: {:.2f} s'.format(end - start))
    print('log Z: {:.2f}'.format(results.logz[-1]))

    # Retrieve posterior samples.
    samples = results.samples
    logl = results.logl
    logwt = results.logwt
    
    save_data = {'samples': samples, 'logl': logl, 'logwt': logwt, 'logz': results.logz[-1]}
    
    if realization_index is not None:
        npz_filepath = f'dynesty_{model}_{num_nodes}_linear_nodes_realization_{realization_index}.npz'
        save_data['omegas'] = Omegas
    else:
        npz_filepath = f'dynesty_{model}_{num_nodes}_linear_nodes.npz'
        
    np.savez(npz_filepath, **save_data)
    print(samples.shape)
    print(logl.shape)
    print(logwt.shape)

if __name__ == '__main__':
    main()
