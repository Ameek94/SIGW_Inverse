import numpy as np
from jax import config, vmap, jit
config.update("jax_enable_x64", True)
from omega_gw_jax import OmegaGWjax
from interpax import CubicSpline
import matplotlib
from matplotlib import cm, colors
import natpy as nat
import warnings
from getdist import plots, MCSamples, loadMCSamples
import sys
# Set matplotlib parameters
font = {'size': 16, 'family': 'serif'}
axislabelfontsize = 'large'
matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=True)
matplotlib.rc('legend', fontsize=16)
import matplotlib.pyplot as plt
import jax.numpy as jnp
from scipy.special import logsumexp


import math
SQRTEPS = math.sqrt(float(np.finfo(np.float64).eps))
def resample_equal(samples,logl, logwt, rstate):
    # Extract the weights and compute the cumulative sum.
    # wt = np.exp(logwt)
    # weights = wt / wt.sum()
    log_total = logsumexp(logwt)
    weights = np.exp(logwt - log_total) 
    cumulative_sum = np.cumsum(weights)

    # if abs(cumulative_sum[-1] - 1.) > SQRTEPS:
    #     # same tol as in numpy's random.choice.
    #     # Guarantee that the weights will sum to 1.
    #     warnings.warn("Weights do not sum to 1 and have been renormalized.")
    cumulative_sum /= cumulative_sum[-1]
    # this ensures that the last element is strictly == 1

    # Make N subdivisions and choose positions with a consistent random offset.
    nsamples = len(weights)
    print(nsamples)

    positions = (rstate.random() + np.arange(nsamples)) / nsamples

    # Resample the data.
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

def plot_functional_posterior(vals=[], k_arr=[], intervals=[99.7, 95., 68.],
                              weights = None,
                              fig = None, ax=None,
                              ylabels=[r'$P_{\zeta}$', r'$\Omega_{\rm GW}$'],
                              aspect_ratio=(6, 5),
                              interval_cols=[('#006FED', 0.2), ('#006FED', 0.4), ('#006FED', 0.6)]):
    # Plot the posterior of y = f(k|x) using symmetric credible intervals.
    nfuncs = len(vals)
    if fig is None:
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
        ax[i].plot(k_arr[i], np.median(val, axis=0), color='#006FED', lw=2.5)
        # ax[i].set_ylabel(ylabels[i])
    return fig, ax


# Command line argument for number of nodes
if len(sys.argv) != 3:
    print("Usage: python script.py <num_nodes>")
    sys.exit(1)

num_nodes = int(sys.argv[1])
free_nodes = num_nodes - 2
# Load data files
datadir = './NG15_Ceffyl/30f_fs{hd}_ceffyl/'
violin_data = np.load('./violins_data.npz')
OmegaGW_data = violin_data['OmegaGW_data']
freqs = violin_data['freqs']

# v1 = ax.violinplot(list(OmegaGW_data), np.log10(freqs), widths=0.05)

# frequencies = np.load(f'{datadir}/freqs.npy')
# frequencies =fre
frequencies = freqs
left_node = -9.
right_node = -7.
y_max = -1.
y_min = -6.
s = jnp.linspace(0, 1, 15)  # First rescaled internal momentum
t = jnp.logspace(-5,5, 200)  # Second rescaled internal momentum
t_expanded = jnp.expand_dims(t, axis=-1)
t = jnp.repeat(t_expanded, len(frequencies), axis=-1)
gwb_calculator = OmegaGWjax(s=s,t=t,f=frequencies,norm="RD",jit=True,to_numpy=False)
p_arr = jnp.logspace(left_node+0.001, right_node-0.001, 150)


def interpolate(nodes, vals, x):
    # Create a cubic spline interpolation of log10(Pζ) and then convert back to linear scale.
    # spl = CubicSpline(nodes, vals, check=False)
    # Testing linear interpolation
    # spl = lambda x: 
    res = jnp.power(10, jnp.interp(x, nodes, vals))
    res = jnp.where(x < left_node, 0, res)
    res = jnp.where(x > right_node, 0, res)
    return res


def get_pz_gwb(nodes,vals):
    pf = lambda k: interpolate(nodes=nodes,vals=vals,x=jnp.log10(k))
    pz = pf(p_arr)
    omegagw = gwb_calculator(pf,frequencies)
    return (pz, omegagw)

# run data
run_data = np.load(f'ptarcade_{num_nodes}_linear.npz')
samples = run_data['samples']
logl = run_data['logl']
logwt = run_data['logwt']

# thinning the samples
num_samples = int(sys.argv[2])
thinning = max(1,len(samples)//num_samples)
xs = samples[:, :free_nodes][::thinning]
ys = samples[:, free_nodes:][::thinning]
xs = jnp.pad(xs, ((0, 0), (1, 1)), 'constant', constant_values=((0, 0), (left_node, right_node)))
ys = jnp.array(ys)
# logwt = run_data['logwt'][::thinning]
# print(xs.shape, ys.shape, logwt.shape)
# logwt_total = logsumexp(logwt)
# thinned_weights = np.exp(logwt - logwt_total)
# thinned_weights = thinned_weights / thinned_weights.sum()

rstate = np.random.RandomState(42)
resampled_samples, resampled_logl = resample_equal(samples, logl, logwt, rstate)
thinning = max(1,len(resampled_samples) // num_samples)
ys = resampled_samples[:,free_nodes:][::thinning]
ys = resampled_samples[:,free_nodes:][::thinning]
thinned_weights = np.ones(ys.shape[0])
print(f"shapes xs = {xs.shape}, ys = {ys.shape}, thinned_weights = {thinned_weights.shape}")
ys = jnp.array(ys)
if free_nodes>=1:
    xs = resampled_samples[:,:free_nodes][::thinning]
    xs = jnp.pad(xs, ((0,0),(1,1)), 'constant', constant_values=((0,0),(left_node, right_node)))
else:
    xs = jnp.array([[left_node,right_node] for _ in range(len(ys))])

pz_amps,gwb_amps = split_vmap(get_pz_gwb,(xs,ys),batch_size=32)

# print(pz_amps[0])
# print(gwb_amps[0])

log_pz_amps = np.log10(pz_amps)
log_gwb_amps = np.log10(gwb_amps)



fig, (ax1,ax2) = plot_functional_posterior(vals=[pz_amps,log_gwb_amps],
                                    k_arr=[p_arr,np.log10(frequencies)],#intervals=[95.,68.],
                                    weights = thinned_weights,
                                    aspect_ratio=(6,4.5))
                                    #aspect_ratio=(8,6))
ax2.set(xscale='linear',yscale='linear',xlabel=r'$\log_{10} f\,{\rm [Hz]}$')
ax1.set(xscale='log',yscale='log',xlabel=r'$f\,{\rm [Hz]}$')
k_mpc_f_hz = 2*np.pi * 1.03 * 10**14
for x in [ax1]:#,ax2]:
    secax = x.secondary_xaxis('top', functions=(lambda x: x * k_mpc_f_hz, lambda x: x / k_mpc_f_hz))
    secax.set_xlabel(r"$k\,{\rm [Mpc^{-1}]}$",labelpad=10) 
# ax[0].set_xlim(min(p_arr),max(p_arr))
from matplotlib import colors
# OmegaGW_data = 10**OmegaGW_data
v1 = ax2.violinplot(list(OmegaGW_data), positions=np.log10(freqs), widths=0.05)
for pc in v1['bodies']:
    pc.set_facecolor(('#E03424', 0.25))
    # pc.set_facecolor(('blue',0.25))
    pc.set_edgecolor(('#E03424',0.75))
    pc.set_linestyle('solid')
    # pc.set_alpha(0.8)
    pc.set_linewidth(1.5)
v1['cmins'].set_color(('#E03424',0.5))
v1['cmaxes'].set_color(('#E03424',0.5))
v1['cbars'].set_color(('#E03424',0.5))
ax2.set_ylim(-12,-4)
# ax2.set_ylim(1e-12,1e-4)
ax2.set_xlim(-8.8, -7.68)# plt.show()
    # for x in ax:
    #     x.set(xscale='log',yscale='linear',xlabel=r'$f\,{\rm [Hz]}$')
plt.savefig(f'NG15_recon_{num_nodes}_ptarcade_linear_nodes.pdf', bbox_inches='tight')


# Results:

# N = 2
# log Z: -53.6508

# N = 3 
# log Z: -54.5899

# N = 4
# log Z: -51.0853

# N = 5
# log Z: -51.3488

n_nodes = [2,3,4,5]

def plot_grid():
    num_nodes = len(n_nodes)
    fig, ax = plt.subplots(num_nodes , 2, figsize=(6*num_nodes,4.5),sharex='col',constrained_layout=True)
    for i,n in enumerate(n_nodes):
        run_data = np.load(f'ptarcade_{n_nodes}_linear.npz')
        samples = run_data['samples']
        logl = run_data['logl']
        logwt = run_data['logwt']
        rstate = np.random.RandomState(42)
        resampled_samples, resampled_logl = resample_equal(samples, logl, logwt, rstate)
        thinning = max(1,len(resampled_samples) // num_samples)
        ys = resampled_samples[:,free_nodes:][::thinning]
        ys = resampled_samples[:,free_nodes:][::thinning]
        thinned_weights = np.ones(ys.shape[0])
        print(f"shapes xs = {xs.shape}, ys = {ys.shape}, thinned_weights = {thinned_weights.shape}")
        ys = jnp.array(ys)
        if free_nodes>=1:
            xs = resampled_samples[:,:free_nodes][::thinning]
            xs = jnp.pad(xs, ((0,0),(1,1)), 'constant', constant_values=((0,0),(left_node, right_node)))
        else:
            xs = jnp.array([[left_node,right_node] for _ in range(len(ys))])

        pz_amps,gwb_amps = split_vmap(get_pz_gwb,(xs,ys),batch_size=32)

        log_pz_amps = np.log10(pz_amps)
        log_gwb_amps = np.log10(gwb_amps)

        plot_functional_posterior(vals=[pz_amps,log_gwb_amps],
                                    fig = fig, ax = ax[i],
                                    k_arr=[p_arr,np.log10(frequencies)],#intervals=[95.,68.],
                                    weights = thinned_weights,
                                    aspect_ratio=(6,4.5))

        ax[i][1].set(xscale='linear',yscale='linear')
        ax[i][0].set(xscale='log',yscale='log')
    k_mpc_f_hz = 2*np.pi * 1.03 * 10**14
    secax = ax[0,0].secondary_xaxis('top', functions=(lambda x: x * k_mpc_f_hz, lambda x: x / k_mpc_f_hz))
    secax.set_xlabel(r"$k\,{\rm [Mpc^{-1}]}$",labelpad=10)
    ax[i][0].set_xlabel(r'$f\,{\rm [Hz]}$')
    ax[i][1].set_xlabel(r'$\log_{10} f\,{\rm [Hz]}$')
    ax[i][0].set_ylabel(r'$P_{\zeta}$')
    ax[i][1].set_ylabel(r'$\log_{10} \Omega_{\rm GW}$')
    plt.show()

def main():
    plot_grid()
    plt.savefig(f'NG15_recon_ptarcade_linear_nodes.pdf', bbox_inches='tight')

if __name__ == "__main__":  
    main()