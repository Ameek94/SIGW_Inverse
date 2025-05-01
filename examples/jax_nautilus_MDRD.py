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
gwb_calculator = None
frequencies = None
Omegas = None
cov = None
p_arr = None
pz_amp = None
params_MDRD = None

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
    N = free_nodes  # Number of x parameters
    x = cube[:, :N]
    exponents = 1.0 / jnp.arange(1, N + 1)
    y_vals = x ** exponents  # shape (Npoints, free_nodes)
    t_arr = jnp.cumprod(y_vals[:, ::-1], axis=1)[:, ::-1]
    xs = t_arr * (right_node - left_node) + left_node
    ys = cube[:, N:]
    ys = ys * (y_maxs[None, :] - y_mins[None, :]) + y_mins[None, :]
    return jnp.concatenate([xs, ys], axis=1)


# def linear_interpolate(nodes, vals, x):
#     # Create a linear interpolation of log10(Pζ) and then convert back to linear scale.
    

#     res = jnp.where(x < left_node, 0, res)
#     res = jnp.where(x > right_node, 0, res)
#     return res

def interpolate(nodes, vals, x):
    # Create a cubic spline interpolation of log10(Pζ) and then convert back to linear scale.
    # spl = CubicSpline(nodes, vals, check=False)
    # Testing linear interpolation
    # spl = lambda x: 
    res = jnp.power(10, jnp.interp(x, nodes, vals))
    res = jnp.where(x < left_node, 0, res)
    res = jnp.where(x > right_node, 0, res)
    return res

# k_max = 8e-3
# eta_R = 20/k_max
# params_MDRD = [k_max, eta_R]

# def get_gwb(nodes, vals):
#     # Given nodes and values, create a function for Pζ and compute Ω_GW.
#     pf = lambda k,  *args: interpolate(nodes, vals, jnp.log10(k))
#     omegagw = gwb_calculator(pf, frequencies, *params_MDRD)
#     # for comparison with RD:
#     # pf = lambda k: interpolate(nodes, vals, jnp.log10(k))
#     # omegagw = gwb_calculator(pf, frequencies)
#     return (omegagw,)

def get_gwb(nodes, vals):
    pf = lambda k, *args: interpolate(nodes, vals, jnp.log10(k))
    if params_MDRD is not None:
        omegagw = gwb_calculator(pf, frequencies, *params_MDRD)
    else:
        omegagw = gwb_calculator(pf, frequencies)
    return (omegagw,)


# JIT compile get_gwb for speed.
get_gwb_func = jit(get_gwb)

def likelihood(params):
    params = jnp.atleast_2d(params)
    nodes = params[:, :free_nodes]
    # Pad nodes with fixed endpoints
    nodes = jnp.pad(nodes, ((0, 0), (1, 1)), 'constant',
                      constant_values=((0, 0), (left_node, right_node)))
    vals = params[:, free_nodes:]
    omegagw = split_vmap(get_gwb_func, (nodes, vals), batch_size=100)[0]
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
            # if j == nsamples:
            #     break
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

# def get_pz_omega(nodes, vals):
#     # Given nodes and vals, compute Pζ and Ω_GW.
#     pf = lambda k, *args: interpolate(nodes, vals, jnp.log10(k))
#     pz_amps = pf(p_arr)
#     gwb_res = gwb_calculator(pf, frequencies, *params_MDRD)
#     # for comparison with RD:
#     # pf = lambda k: interpolate(nodes, vals, jnp.log10(k))
#     # pz_amps = pf(p_arr)
#     # gwb_res = gwb_calculator(pf, frequencies)
#     return (pz_amps, gwb_res)

def get_pz_omega(nodes, vals):
    pf = lambda k, *args: interpolate(nodes, vals, jnp.log10(k))
    pz_amps = pf(p_arr_local)
    if params_MDRD is not None:
        gwb_res = gwb_calculator(pf, frequencies, *params_MDRD)
    else:
        gwb_res = gwb_calculator(pf, frequencies)
    return (pz_amps, gwb_res)


#############################
# Main function: Sampling and Postprocessing
#############################

def main():
    global free_nodes, left_node, right_node, y_min, y_max, y_mins, y_maxs
    global gwb_calculator, frequencies, Omegas, cov, p_arr, pz_amp, get_gwb_func, params_MDRD

    model = str(sys.argv[1])
    # Load the gravitational wave background data.
    data = np.load(f'./results/{model}_MDRD_data.npz')
    # data = np.load(f'./results/{model}_RDvsMDRD_data.npz')
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
    if '_RD' in model:
        params_MDRD = None
        gwb_calculator = OmegaGWjax(s=s, t=t, f=frequencies, norm="RD", kernel="RD", jit=True)
    else:
        k_max = 6e-3
        eta_R = 20 / k_max
        params_MDRD = [k_max, eta_R]
        gwb_calculator = OmegaGWjax(s=s, t=t, f=frequencies, norm="RD", kernel="I_MD_to_RD", jit=True)

    get_gwb_func = jit(get_gwb)

    # gwb_calculator = OmegaGWjax(s=s, t=t, f=frequencies, norm="RD",kernel="I_MD_to_RD", jit=True)
    # for comparison with RD:

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
    y_min = -12.
    y_mins = np.array(num_nodes * [y_min])
    y_maxs = np.array(num_nodes * [y_max])

    # Set up the sampler.
    ndim = free_nodes + num_nodes
    sampler = Sampler(prior, likelihood, ndim, pass_dict=False, vectorized=True
                                            ,pool=(None,4),filepath=f'./results/nautilus_{model}_MDRD_{num_nodes}_linear_nodes.h5')
    start = time.time()
    sampler.run(verbose=True, f_live=0.005,n_like_max=1e6) #f_live=0.005, n_like_max=5e6)#, n_eff=2000*ndim)
    end = time.time()
    print('Time taken: {:.2f} s'.format(end - start))
    print('log Z: {:.2f}'.format(sampler.log_z))

    # Retrieve posterior samples.
    samples, logl, logwt = sampler.posterior()
    np.savez(f'./results/nautilus_{model}_MDRD_{num_nodes}_linear_nodes.npz', samples=samples, logl=logl, logwt=logwt,
             logz=sampler.log_z)

    print(samples.shape)
    print(logl.shape)
    print(logwt.shape)

    # # Resample to obtain equally weighted samples.
    # rstate = np.random.default_rng(100000)
    # samples, lp = resample_equal(samples, logl, logwt, rstate=rstate)
    # print("Obtained equally weighted samples")
    # print(f"Max and min logprob: {np.max(lp)}, {np.min(lp)}")
    # print(len(lp))

    # # Postprocessing: Compute functional posteriors.
    # p_arr_local = jnp.logspace(left_node + 0.001, right_node - 0.001, 150)
    # thinning = samples.shape[0] // 512
    # xs = samples[:, :free_nodes][::thinning]
    # ys = samples[:, free_nodes:][::thinning]
    # xs = jnp.pad(xs, ((0, 0), (1, 1)), 'constant', constant_values=((0, 0), (left_node, right_node)))
    # ys = jnp.array(ys)
    # pz_amps, gwb_amps = split_vmap(get_pz_omega, (xs, ys), batch_size=32)

    # print(pz_amps.shape)
    # print(gwb_amps.shape)

    # fig, ax = plot_functional_posterior([pz_amps, gwb_amps],
    #                                     k_arr=[p_arr_local, frequencies],
    #                                     aspect_ratio=(6, 4))
    # ax[0].loglog(p_arr_local, pz_amp, color='k', lw=1.5)
    # ax[1].loglog(frequencies, Omegas, color='k', lw=1.5, label='Truth')

    # # Add secondary x-axis (e.g. converting f [Hz] to k [Mpc^{-1}]).
    # k_mpc_f_hz = 2 * np.pi * 1.03 * 10**14
    # for x in ax:
    #     secax = x.secondary_xaxis('top', functions=(lambda x: x * k_mpc_f_hz,
    #                                                    lambda x: x / k_mpc_f_hz))
    #     secax.set_xlabel(r"$k\,{\rm [Mpc^{-1}]}$", labelpad=10)

    # plt.savefig(f'{model}_{num_nodes}.pdf', bbox_inches='tight')
    # plt.show()

if __name__ == '__main__':
    main()
