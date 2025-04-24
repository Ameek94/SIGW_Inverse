import sys
import os 
import re
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from omega_gw_jax import OmegaGWjax
from getdist import plots, MCSamples, loadMCSamples
from interpax import CubicSpline
from jax import config, vmap
from scipy.special import logsumexp
config.update("jax_enable_x64", True)


# Set matplotlib parameters
font = {'size': 16, 'family': 'serif'}
axislabelfontsize = 'large'
matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=True)
matplotlib.rc('legend', fontsize=16)

def renormalise_log_weights(log_weights):
    log_total = logsumexp(log_weights)
    normalized_weights = np.exp(log_weights - log_total)
    return normalized_weights

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


def split_vmap(func,input_arrays,batch_size=32):
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

def weighted_median(data, weights):
    """
    Compute the weighted median of data.
    """
    # Sort the data and weights.
    s_data, s_weights = map(np.array, zip(*sorted(zip(data, weights))))
    # Compute the cumulative sum of the weights.
    cdf = np.cumsum(s_weights)
    # Find the median value.
    idx = np.searchsorted(cdf, 0.5)
    return s_data[idx]

def plot_functional_posterior(vals=[], k_arr=[], intervals=[99.7, 95., 68.],
                              weights = None,
                              ylabels=[r'$P_{\zeta}$', r'$\Omega_{\rm GW}$'],
                              aspect_ratio=(6, 5),
                              interval_cols=[('#006FED', 0.2), ('#006FED', 0.4), ('#006FED', 0.6)]):
    """
    Plot the posterior of y = f(k|x) using symmetric credible intervals.
    """
    nfuncs = len(vals)
    fig, ax = plt.subplots(1, nfuncs, figsize=(aspect_ratio[0] * nfuncs, aspect_ratio[1]), constrained_layout=True)
    if nfuncs == 1:
        ax = [ax]
    if weights is None:
        weights = np.ones(vals.shape[0])
    for i, val in enumerate(vals):
        # weights_i = weights[i]
        # print(weights_i.shape)
        print(val.shape)
        for j, interval in enumerate(intervals):
            y_low, y_high = np.percentile(val, [50 - interval / 2, 50 + interval / 2], axis=0
                                          ,weights=weights,method='inverted_cdf')
            ax[i].fill_between(k_arr[i], y_low, y_high, color=interval_cols[j][0], alpha=interval_cols[j][1])
        medians = np.apply_along_axis(weighted_median, 0, val, weights)
        ax[i].plot(k_arr[i], medians, color='#006FED', lw=2.5)
        # ax[i].plot(k_arr[i], np.median(val, axis=0), color='#006FED', lw=2.5)
        ax[i].set_ylabel(ylabels[i])
    return fig, ax

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

# Set the range for the x (log10) nodes using the data.
pk_min, pk_max = min(p_arr), max(p_arr)
left_node = np.log10(pk_min)
right_node = np.log10(pk_max)
y_min = -6.
y_max = -2.


def interpolate(nodes, vals, x):
    # Create a cubic spline interpolation of log10(Pζ) and then convert back to linear scale.
    # spl = CubicSpline(nodes, vals, check=False)
    # Testing linear interpolation
    spl = lambda x: jnp.interp(x, nodes, vals)
    res = jnp.power(10, spl(x))
    res = jnp.where(x < left_node, 0, res)
    res = jnp.where(x > right_node, 0, res)
    return res

# thinning the samples
num_samples = int(sys.argv[2])

p_arr_local = jnp.logspace(left_node+0.001, right_node-0.001, 200)

def get_pz_omega(nodes, vals):
    # Given nodes and vals, compute Pζ and Ω_GW.
    pf = lambda k: interpolate(nodes, vals, jnp.log10(k))
    pz_amps = pf(p_arr_local)
    gwb_res = gwb_calculator(pf, frequencies)
    return (pz_amps, gwb_res)

# Check current working directory for files matching the pattern
pattern = re.compile(rf'nautilus_{model}_(\d+)_linear_nodes\.npz')

# List all files in the current working directory
files_in_dir = os.listdir(os.getcwd())

# Filter files matching the pattern where n > 2
matching_files = [f for f in files_in_dir if pattern.match(f) and int(pattern.match(f).group(1)) > 2]
logz_list = []
logweights = []
gwb_samples = []
pz_samples = []
if matching_files:
    print("Matching files found:")
    for file in matching_files:
        n = int(pattern.match(file).group(1))
        free_nodes = n - 2
        print(f"Processing file: {file} with {n} nodes")
        data = np.load(file)
        logz = data['logz'].item()
        print(f"Logz: {logz:.4f}")
        samples, logl, logwt = data['samples'], data['logl'], data['logwt']
        equal_samples, equal_logl = resample_equal(samples, logl, logwt, np.random.default_rng())
        thinning = max(1,len(samples)//num_samples)
        xs = equal_samples[:, :free_nodes][::thinning]
        ys = equal_samples[:, free_nodes:][::thinning]
        xs = jnp.pad(xs, ((0, 0), (1, 1)), 'constant', constant_values=((0, 0), (left_node, right_node)))
        ys = jnp.array(ys)
        pz, gwb = split_vmap(get_pz_omega, (xs, ys), batch_size=32)
        pz_samples.append(pz)
        gwb_samples.append(gwb)
        logz_list.append(logz)
        logweight = np.ones(len(xs)) * logz
        print(f"pz_samples shape: {pz.shape}, gwb_samples shape: {gwb.shape}, logweight shape: {logweight.shape}")
        logweights.append(logweight)

# no renormalise the logweights
logweights = np.concatenate(logweights)
weights = renormalise_log_weights(logweights)
gwb_samples = np.concatenate(gwb_samples)
pz_samples = np.concatenate(pz_samples)

print(f"pz_samples shape: {pz_samples.shape}, gwb_samples shape: {gwb_samples.shape}, weights shape: {weights.shape}")

fig, ax = plot_functional_posterior([pz_samples,gwb_samples],
                                    k_arr=[p_arr_local, frequencies],
                                    weights = weights,
                                    aspect_ratio=(6,4.5))
ax[0].loglog(p_arr, pz_amp, color='k', lw=1.5)
ax[1].loglog(frequencies, Omegas, color='k', lw=1.5, label='Truth')
ax[1].errorbar(frequencies, Omegas, yerr=np.sqrt(np.diag(cov)), fmt='o', color='k', capsize=4.,alpha=0.5,markersize=2)
ax[1].legend()
k_mpc_f_hz = 2*np.pi * 1.03 * 10**14
for x in ax:
    x.set(xscale='log', yscale='log', xlabel=r'$f\,{\rm [Hz]}$')
    secax = x.secondary_xaxis('top', functions=(lambda x: x * k_mpc_f_hz, lambda x: x / k_mpc_f_hz))
    secax.set_xlabel(r"$k\,{\rm [Mpc^{-1}]}$",labelpad=10) 
plt.savefig(f'./results/nautilus_{model}_linear_posterior.pdf',bbox_inches='tight')