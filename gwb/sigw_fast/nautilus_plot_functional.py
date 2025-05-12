from getdist import plots, MCSamples, loadMCSamples
import sys
sys.path.append('../')
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from getdist import plots, MCSamples
from scipy.special import logsumexp
from sigw_fast.sigwfast import sigwfast_mod as gw
from math import sqrt
import tqdm
try:
    from sigw_fast.libraries import sdintegral_numba as sd
except ImportError:
    from sigw_fast.libraries import sdintegral as sd

# Set matplotlib parameters
font = {'size': 16, 'family': 'serif'}
matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=True)
matplotlib.rc('legend', fontsize=16)
blue = '#006FED'

if len(sys.argv) < 4:
    print("Usage: python nautilus_plot_functional.py <w> <num_nodes> <gwb_model>")
    print("w: 0.5, 0.66, 0.99")
    print("num_nodes: number of nodes in the spline")
    print("gwb_model: 'bpl' or 'peaked' or 'osc' ")
    sys.exit(1)

def weighted_median(data, weights):
    """Compute the weighted median of data."""
    s_data, s_weights = map(np.array, zip(*sorted(zip(data, weights))))
    cdf = np.cumsum(s_weights)
    idx = np.searchsorted(cdf, 0.5)
    return s_data[idx]

def resample_equal(samples, logl, logwt, rstate):
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

def plot_functional_posterior(vals,samples,k_arr = [], intervals=[99.7, 95., 68.],weights=None,
                              ylabels=[r'$P_{\zeta}$', r'$\Omega_{\rm GW}$'],
                              aspect_ratio=(6, 4.5),
                              interval_cols=[('#006FED', 0.2), ('#006FED', 0.4), ('#006FED', 0.6)]):
    # given a function y = f(k|x) with x~Posterior samples, plot the posterior of y at k_arr, with symmetric credible intervals

    nfuncs = len(vals)

    if weights is None:
        weights = np.ones(len(samples.shape[0]))

    fig, ax = plt.subplots(1,nfuncs,figsize=(aspect_ratio[0]*nfuncs,aspect_ratio[1]),constrained_layout=True)
    if nfuncs == 1:
        ax = [ax]
    for i,val in enumerate(vals):
        y = val # so y should have shape (nsamples, nk)
        for j,interval in enumerate(intervals):
            y_low, y_high = np.percentile(y,[50-interval/2,50+interval/2],axis=0,weights=weights,
                                          method='inverted_cdf')
            ax[i].fill_between(k_arr[i],y_low,y_high,color=interval_cols[j])
        medians = np.apply_along_axis(weighted_median, 0, val, weights)
        ax[i].plot(k_arr[i], medians, color='#006FED', lw=2.5)
        # ax[i].plot(k_arr[i],np.median(y,axis=0),color=blue,lw=2)
        ax[i].set_ylabel(ylabels[i])
    for x in ax:
        x.set(xscale='log', yscale='log', xlabel=r'$f\,{\rm [Hz]}$')
    return fig, ax

def compute_pz(k, samples, num_nodes, left_node, right_node):
    """Compute Pz for given k and samples."""
    Pz = []
    for sample in tqdm.tqdm(samples,desc='Pz'):
        free_nodes = sample[1:num_nodes - 1]
        nodes = np.pad(free_nodes, (1, 1), 'constant', constant_values=(left_node, right_node))
        vals = sample[num_nodes - 1:]
        res = gw.power_spectrum_k_array(nodes, vals, k)
        Pz.append(res)
    return np.array(Pz)  # Pz has shape (nsamples, nk)

if __name__ == "__main__":
    # Load data
    w = float(sys.argv[1])

    if w == 0.5:
        data = np.load('./spectra_0p5_interp.npz')
        save_file = '0p5'
    elif w == 0.99:
        data = np.load('./spectra_0p99_interp.npz')
        save_file = '0p99'
    elif w==0.66:
        data = np.load('./spectra_0p66_interp.npz')
        save_file = '0p66'

    frequencies = data['frequencies']
    gwb_model = str(sys.argv[3])
    Omegas = data[f'gw_{gwb_model}']
    kstar = 1e-3
    omks_sigma = Omegas * (0.05 * (np.log(frequencies / kstar))**2 + 0.1)
    cov = np.diag(omks_sigma**2)
    pk_arr = data['pk_arr']
    pk_min, pk_max = min(pk_arr), max(pk_arr)
    left_node = np.log10(pk_min)
    right_node = np.log10(pk_max)
    p_arr = np.logspace(left_node + 0.001, right_node - 0.001, 100)
    pz_amp = data[f'pk_{gwb_model}']
    num_nodes = int(sys.argv[2])

    # Load samples
    sample_data = np.load(f'{gwb_model}_{save_file}_interp_free_{num_nodes}.npz')
    samples = sample_data['samples']
    logwt = sample_data['logwt']
    logl = sample_data['logl']
    omegagw = sample_data['omegagw']
    weights_total = logsumexp(logwt)
    weights = np.exp(logwt - weights_total)
    weights = weights / np.sum(weights)

    print(f"Shapes: {samples.shape}, {logwt.shape}, {logl.shape}, {omegagw.shape}")

    # Compute Pz
    pz_amps = compute_pz(p_arr, samples, num_nodes, left_node, right_node)

    # Plot posterior
    fig, ax = plot_functional_posterior([pz_amps, omegagw], samples, [p_arr, frequencies], weights=weights)
    ax[0].loglog(pk_arr, pz_amp, color='k', lw=1.5)
    ax[1].loglog(frequencies, Omegas, color='k', lw=1.5, label='Truth')
    ax[1].errorbar(frequencies, Omegas, yerr=np.sqrt(np.diag(cov)), fmt='o', color='k', capsize=4., alpha=0.5, markersize=2)
    ax[1].legend()
    k_mpc_f_hz = 2 * np.pi * 1.03 * 10**14
    for x in ax:
        secax = x.secondary_xaxis('top', functions=(lambda x: x * k_mpc_f_hz, lambda x: x / k_mpc_f_hz))
        secax.set_xlabel(r"$k\,{\rm [Mpc^{-1}]}$", labelpad=10)
    plt.savefig(f'{gwb_model}_{save_file}_{num_nodes}_posterior.pdf', bbox_inches='tight')
    plt.show()

    # Plot 1D posterior
    names = ['w']
    labels = ['w']
    bounds = [[0.1,0.99]]
    ranges = dict(zip(names,bounds))
    print(ranges)
    gd_samples = MCSamples(samples=samples[:,0], names=names, labels=labels,ranges=ranges,weights=weights)
    g = plots.get_subplot_plotter(subplot_size=6,subplot_size_ratio=2/3)
    blue = '#006FED'
    g.settings.title_limit_fontsize = 14
    g.settings.axes_fontsize=16
    g.settings.axes_labelsize=16
    g.plot_1d(gd_samples, 'w', marker=w, marker_color=blue, colors=[blue],title_limit=2)
    # plt.show()
    g.export(f'{gwb_model}_{save_file}_{num_nodes}_1D_w.pdf')
    ax = g.subplots[0,0]
    ax.set_xlim(w - 0.2, 1.0)
    print(gd_samples.getMargeStats())
