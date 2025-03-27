import sys
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
from jax.scipy.linalg import cho_solve
config.update("jax_enable_x64", True)


# Set matplotlib parameters
font = {'size': 16, 'family': 'serif'}
axislabelfontsize = 'large'
matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=True)
matplotlib.rc('legend', fontsize=16)

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
    # Plot the posterior of y = f(k|x) using symmetric credible intervals.
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
num_nodes = int(sys.argv[2])
free_nodes = num_nodes - 2

# Set the range for the x (log10) nodes using the data.
pk_min, pk_max = min(p_arr), max(p_arr)
left_node = np.log10(pk_min)
right_node = np.log10(pk_max)
y_min = -8.
y_max = -1.

# get the samples
samples_data = np.load(f'./results/nautilus_{model}_{num_nodes}_gp_nodes.npz')
samples = samples_data['samples']
logl = samples_data['logl']
logz = samples_data['logz']
print(f"Logz: {logz}, max logl: {logl.max()}")

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

# thinning the samples
num_samples = int(sys.argv[3])
thinning = max(1,len(samples)//num_samples)
xs = samples[:, 1:free_nodes+1][::thinning]
ys = samples[:, free_nodes+1:][::thinning]
lengthscales = samples[:, 0][::thinning]
xs = jnp.pad(xs, ((0, 0), (1, 1)), 'constant', constant_values=((0, 0), (left_node, right_node)))
ys = jnp.array(ys)
logwt = samples_data['logwt'][::thinning]
print(xs.shape, ys.shape, logwt.shape)
from scipy.special import logsumexp
logwt_total = logsumexp(logwt)
thinned_weights = np.exp(logwt - logwt_total)
thinned_weights = thinned_weights / thinned_weights.sum()
# print(weights.shape)

p_arr_local = jnp.logspace(left_node+0.001, right_node-0.001, 200)

def get_pz_omega(nodes, vals,lengthscales):
    # Given nodes and vals, compute Pζ and Ω_GW.
    pf = lambda k: interpolate(nodes, vals, lengthscales, jnp.log10(k))
    pz_amps = pf(p_arr_local)
    gwb_res = gwb_calculator(pf, frequencies)
    return (pz_amps, gwb_res)

pz_amps, gwb_amps = split_vmap(get_pz_omega, (xs, ys, lengthscales), batch_size=64)


fig, ax = plot_functional_posterior([pz_amps, gwb_amps],
                                    k_arr=[p_arr_local, frequencies],
                                    weights = thinned_weights,
                                    aspect_ratio=(6,4.5))
ax[0].loglog(p_arr, pz_amp, color='k', lw=1.5)
ax[1].loglog(frequencies, Omegas, color='k', lw=1.5, label='Truth')
ax[1].errorbar(frequencies, Omegas, yerr=np.sqrt(np.diag(cov)), fmt='o', color='k', capsize=4.,alpha=0.5,markersize=2)
ax[1].legend()
ax[0].set_ylim(10**y_min, 10**y_max)
k_mpc_f_hz = 2*np.pi * 1.03 * 10**14
for x in ax:
    x.set(xscale='log', yscale='log', xlabel=r'$f\,{\rm [Hz]}$')
    secax = x.secondary_xaxis('top', functions=(lambda x: x * k_mpc_f_hz, lambda x: x / k_mpc_f_hz))
    secax.set_xlabel(r"$k\,{\rm [Mpc^{-1}]}$",labelpad=10) 
plt.savefig(f'./results/nautilus_{model}_{num_nodes}_gp_posterior.pdf',bbox_inches='tight')