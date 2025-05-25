import os
import sys
import time
import warnings
import math
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from omega_gw_jax import OmegaGWjax
from getdist import plots, MCSamples, loadMCSamples
from interpax import CubicSpline
from functools import partial
from nautilus import Sampler
from utils import split_vmap

def prior(cube, free_nodes, left_node, right_node, y_mins, y_maxs):
    """
    Transforms the input cube from [0,1] uniform parameters to the desired prior space.
    This vectorized version supports cube being either a 1D array of shape (nd,)
    or a 2D array of shape (Npoints, nd).
    """
    cube = jnp.atleast_2d(cube.copy())
    N = free_nodes  # Number of x parameters
    x = cube[:, :N]
    exponents = 1.0 / jnp.arange(1, N + 1)
    y_vals = x ** exponents  # shape (Npoints, free_nodes)
    t_arr = jnp.cumprod(y_vals[:, ::-1], axis=1)[:, ::-1]
    xs = t_arr * (right_node - left_node) + left_node
    ys = cube[:, N:]
    ys = ys * (y_maxs[None, :] - y_mins[None, :]) + y_mins[None, :]
    return jnp.concatenate([xs, ys], axis=1)

def interpolate(nodes, vals, x, left_node, right_node):
    # Create a cubic spline interpolation of log10(Pζ) and then convert back to linear scale.
    # spl = CubicSpline(nodes, vals, check=False)
    # Testing linear interpolation
    # spl = lambda x: 
    res = jnp.power(10, jnp.interp(x, nodes, vals))
    res = jnp.where(x < left_node, 0, res)
    res = jnp.where(x > right_node, 0, res)
    return res

def get_gwb(nodes, vals, left_node, right_node, gwb_calculator, frequencies):
    # Given nodes and values, create a function for Pζ and compute Ω_GW.
    pf = lambda k: interpolate(nodes, vals, jnp.log10(k),left_node,right_node)
    omegagw = gwb_calculator(pf, frequencies)
    return (omegagw,)

def likelihood(params,free_nodes, left_node, right_node, Omegas, cov, get_gwb_func):
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
    return res, omegagw


def main():
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

    # Set the range for the x (log10) nodes using the data.
    pk_min, pk_max = min(p_arr), max(p_arr)
    left_node = np.log10(pk_min)
    right_node = np.log10(pk_max)

    # Set the y range for the interpolation.
    y_max = -1.
    y_min = -8.
    y_mins = np.array(num_nodes * [y_min])
    y_maxs = np.array(num_nodes * [y_max])

    # set up gwb_calculator, prior and likelihood functions.
    get_gwb_func = jax.jit(partial(get_gwb, gwb_calculator=gwb_calculator, frequencies=frequencies,
                                   left_node=left_node, right_node=right_node))

    prior_func = jax.jit(partial(prior, free_nodes=free_nodes, left_node=left_node, 
                         right_node=right_node, y_mins=y_mins, y_maxs=y_maxs))
    likelihood_func = jax.jit(partial(likelihood, free_nodes=free_nodes, left_node=left_node, 
                              right_node=right_node, Omegas=Omegas, cov=cov, 
                              get_gwb_func=get_gwb_func))

    # Set up the sampler.
    ndim = free_nodes + num_nodes
    sampler = Sampler(prior_func, likelihood_func, ndim, 
                      pass_dict=False, 
                      vectorized=True,
                      resume=True,
                      pool=(None,4),
                      filepath=f'./nautilus_{model}_{num_nodes}_linear_nodes.h5') 
    
    start = time.time()
    sampler.run(verbose=True, f_live=0.005, n_like_max=2e6)
    end = time.time()
    print('Time taken: {:.2f} s'.format(end - start))
    print('log Z: {:.2f}'.format(sampler.log_z))

    # Retrieve posterior samples.
    samples, logl, logwt, omegagw = sampler.posterior(return_blobs=True)
    print(f"Shape of samples: {samples.shape}, logl: {logl.shape}, logwt: {logwt.shape}, omegagw: {omegagw.shape}")
    np.savez(f'nautilus_{model}_{num_nodes}_linear_nodes.npz', 
             samples=samples, logl=logl, logwt=logwt, logz=sampler.log_z, omegagw=omegagw)
    print(f"Logwts min: {np.min(logwt)}, max: {np.max(logwt)}")
    
if __name__ == "__main__":
    main()
