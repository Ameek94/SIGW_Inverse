import os
os.environ['XLA_FLAGS'] = f"--xla_force_host_platform_device_count={os.cpu_count()}"
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
import tensorflow_probability.substrates.jax as tfp
from jaxns import NestedSampler, TerminationCondition
from jaxns import Model, Prior
from jaxns.framework.special_priors import ForcedIdentifiability
tfpd = tfp.distributions


# Set matplotlib parameters
font = {'size': 16, 'family': 'serif'}
axislabelfontsize = 'large'
matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=True)
matplotlib.rc('legend', fontsize=16)

# Global variables (will be set in main)
num_nodes = None
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


def prior():
    x = yield ForcedIdentifiability(n=num_nodes, low=left_node, high=right_node, name='x',fix_left=True,fix_right=True)
    y = yield Prior(tfpd.Uniform(low=y_min*jnp.ones(num_nodes), high=y_max*jnp.ones(num_nodes)), name='y')
    return x, y


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

@jit
def likelihood(nodes,vals):
    omegagw = get_gwb(nodes,vals) 
    diff = omegagw - Omegas
    sol = jnp.linalg.solve(cov, diff.T).T
    res = -0.5 * jnp.dot(diff, sol.T)
    res = jnp.where(jnp.isnan(res), -1e10, res)
    res = jnp.where(res < -1e10, -1e10, res)
    return res

#############################
# Main function: Sampling and Postprocessing
#############################

def main():
    global num_nodes, free_nodes, left_node, right_node, y_min, y_max, y_mins, y_maxs
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

    # Set up the sampler.
    ndim = free_nodes + num_nodes

    model = Model(prior, likelihood)
    model.sanity_check(key=jax.random.PRNGKey(0),S=100) 
    
    start = time.time()
    exact_ns = NestedSampler(model=model,parameter_estimation=False,
                             verbose=True,difficult_model=True)

    termination_reason, state = exact_ns(jax.random.PRNGKey(42),
                                         term_cond=TerminationCondition(dlogZ=0.2,
                                                                        evidence_uncert=0.01,
                                                                        max_num_likelihood_evaluations=int(1e6)))
    results = exact_ns.to_results(termination_reason=termination_reason, state=state)
    exact_ns.summary(results)
    end = time.time()
    # print('log Z: {:.2f} +/- {:.2f}'.format(logz, logzerr))
    print('Time taken: {:.2f} s'.format(end - start))
    # print('log Z: {:.2f}'.format(sampler.log_z))

if __name__ == '__main__':
    main()
