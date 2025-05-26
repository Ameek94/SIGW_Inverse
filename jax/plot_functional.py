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
from utils import renormalise_log_weights, resample_equal, split_vmap, plot_functional_posterior


# Set matplotlib parameters
font = {'size': 16, 'family': 'serif'}
axislabelfontsize = 'large'
matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=True)
matplotlib.rc('legend', fontsize=16)


if len(sys.argv) < 3:
    print("Usage: python plot_functional.py <model> <num_samples>")
    sys.exit(1)

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
y_min = -8.
y_max = 0.

def interpolate(nodes, vals, x):
    spl = lambda x: jnp.interp(x, nodes, vals)
    res = jnp.power(10, spl(x))
    res = jnp.where(x < left_node, 0, res)
    res = jnp.where(x > right_node, 0, res)
    return res

# thinning the samples, use ~10000 for decent plots
num_samples = int(sys.argv[2])

p_arr_local = jnp.logspace(left_node+0.001, right_node-0.001, 200)

def get_pz(nodes, vals):
    # Given nodes and vals, compute Pζ
    pf = lambda k: interpolate(nodes, vals, jnp.log10(k))
    pz_amps = pf(p_arr_local)
    # omegagw = gwb_calculator(pf, frequencies)
    return (pz_amps,)

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
Num_nodes = []
if matching_files:
    print("Matching files found:")
    for file in matching_files:
        n = int(pattern.match(file).group(1))
        free_nodes = n - 2
        print(f"Processing file: {file} with {n} nodes")
        data = np.load(file)
        logz = data['logz'].item()
        print(f"Logz: {logz:.4f}")
        samples, logl, logwt, omegagw = data['samples'], data['logl'], data['logwt'], data['omegagw']
        equal_samples, equal_omegagw = resample_equal(samples, omegagw, logwt, np.random.default_rng())
        thinning = max(1,len(samples)//num_samples)
        xs = equal_samples[:, :free_nodes][::thinning]
        ys = equal_samples[:, free_nodes:][::thinning]
        omegagw = equal_omegagw[::thinning]
        xs = jnp.pad(xs, ((0, 0), (1, 1)), 'constant', constant_values=((0, 0), (left_node, right_node)))
        ys = jnp.array(ys)
        pz = split_vmap(get_pz, (xs, ys), batch_size=100)[0]
        pz_samples.append(pz)
        gwb_samples.append(omegagw)
        Num_nodes.append(n)
        logz_list.append(logz)
        logweight = np.ones(len(xs)) * logz
        print(f"pz_samples shape: {pz.shape}, gwb_samples shape: {omegagw.shape}, logweight shape: {logweight.shape}")
        logweights.append(logweight)

# now renormalise the logweights
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
plt.savefig(f'./nautilus_{model}_linear_posterior.pdf',bbox_inches='tight')


# Plot the logZ values against the number of nodes
plt.figure(figsize=(6,4))
plt.plot()
plt.xlabel(r'Number of nodes')
plt.ylabel(r'$\log \mathcal{Z}$')
# Get data from matching files


Num_nodes, logZ = zip(*sorted(zip(Num_nodes, logz_list)))
Num_nodes = list(Num_nodes)
logZ = list(logZ)

print(logZ, Num_nodes)
plt.plot(Num_nodes, logZ, '-.',color='k',alpha=0.9)
plt.scatter(Num_nodes, logZ, color='k',marker='x',s=20)
# Annotate each point with its logZ value
ax = plt.gca()
y_min = min(logZ) - 5
y_max = max(logZ) + 5
ax.set_ylim(y_min, y_max)
ax.set_xlim(min(Num_nodes) - 0.5, max(Num_nodes) + 0.5)
y_mid = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2
for x, y in zip(Num_nodes, logZ):
    plt.text(x+0.1, y-2, f'({y:.2f})', fontsize=12, ha='center', va='bottom')
plt.savefig(f'./linear_logz_{model}.pdf',bbox_inches='tight')