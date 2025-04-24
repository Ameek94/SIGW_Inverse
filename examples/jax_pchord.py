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
from dynesty import DynamicNestedSampler
from mpi4py.futures import MPIPoolExecutor
import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior, SortedUniformPrior


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

def prior(params):
    """ Uniform prior from [-1,1]^D. """
    xs = SortedUniformPrior(left_node,right_node)(params[:free_nodes])
    ys = UniformPrior(y_min,y_max)(params[free_nodes:])
    return  np.concatenate([xs,ys]) # array


def interpolate(nodes, vals, x):
    # Create a cubic spline interpolation of log10(Pζ) and then convert back to linear scale.
    # spl = CubicSpline(nodes, vals, check=False)
    # Testing linear interpolation
    # spl = lambda x: 
    res = jnp.power(10, jnp.interp(x, nodes, vals))
    res = jnp.where(x < left_node, 0, res)
    res = jnp.where(x > right_node, 0, res)
    return res

def get_gwb(nodes, vals):
    # Given nodes and values, create a function for Pζ and compute Ω_GW.
    pf = lambda k: interpolate(nodes, vals, jnp.log10(k))
    omegagw = gwb_calculator(pf, frequencies)
    return omegagw

# JIT compile get_gwb for speed.
get_gwb_func = jit(get_gwb)

# @jit
def likelihood(params):
    print(f"params shape: {params.shape}")
    nodes = params[:free_nodes]
    # Pad nodes with fixed endpoints
    nodes = jnp.pad(nodes, ((1, 1),), 'constant', constant_values=(left_node, right_node))
    vals = params[free_nodes:]
    print(f"shapes: {nodes.shape}, {vals.shape}")
    omegagw = get_gwb(nodes,vals) #split_vmap(get_gwb_func, (nodes, vals), batch_size=100)[0]
    diff = omegagw - Omegas
    print(f"diff shape: {diff.shape}")
    sol = jnp.linalg.solve(cov, diff.T).T
    # print(f"sol shape: {sol.shape}")
    res = -0.5 * jnp.dot(diff, sol.T)
    res = jnp.where(jnp.isnan(res), -1e10, res)
    res = jnp.where(res < -1e10, -1e10, res)
    print(res)
    return res, omegagw


def get_pz_omega(nodes, vals):
    # Given nodes and vals, compute Pζ and Ω_GW.
    pf = lambda k: interpolate(nodes, vals, jnp.log10(k))
    pz_amps = pf(p_arr)
    gwb_res = gwb_calculator(pf, frequencies)
    return (pz_amps, gwb_res)

def dumper(live, dead, logweights, logZ, logZerr):
    print("Last dead point:", dead[-1])

#############################
# Main function: Sampling and Postprocessing
#############################

def main():
    global free_nodes, left_node, right_node, y_min, y_max, y_mins, y_maxs
    global gwb_calculator, frequencies, Omegas, cov, p_arr, pz_amp

    model = str(sys.argv[1])
    # Load the gravitational wave background data.
    data = np.load(f'./{model}_data.npz')
    frequencies = data['k']
    Omegas = jnp.array(data['gw'])
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

    nDims = free_nodes + num_nodes
    nDerived = len(frequencies)
    settings = PolyChordSettings(nDims, nDerived)
    settings.file_root = 'bpl_pchord_'+str(num_nodes)
    settings.nlive = 10 * nDims
    settings.do_clustering = True
    settings.read_resume = True
    settings.precision_criterion = 0.01


    # Set up the sampler.
    start = time.time()
    output = pypolychord.run_polychord(likelihood, nDims, nDerived, settings, prior, dumper)
    print("Nested sampling complete")
    end = time.time()
    print('Time taken: {:.2f} s'.format(end - start))
    # print('log Z: {:.2f}'.format(sampler.log_z))

    # # Retrieve posterior samples.
    # samples, logl, logwt = sampler.posterior()
    # np.savez(f'nautilus_{model}_{num_nodes}_linear_nodes.npz', samples=samples, logl=logl, logwt=logwt, logz=sampler.log_z)
    # print(samples.shape)
    # print(logl.shape)
    # print(logwt.shape)
    # plt.show()

if __name__ == '__main__':
    main()
