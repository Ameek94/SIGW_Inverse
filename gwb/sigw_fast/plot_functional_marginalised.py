import sys
sys.path.append('../')
import os 
import re
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from getdist import plots, MCSamples, loadMCSamples
from scipy.special import logsumexp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sigw_fast.sigwfast import sigwfast_mod as gw
import tqdm
try:
    from sigw_fast.libraries import sdintegral_numba as sd
except ImportError:
    from sigw_fast.libraries import sdintegral as sd
# Set matplotlib parameters
font = {'size': 16, 'family': 'serif'}
axislabelfontsize = 'large'
matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=True)
matplotlib.rc('legend', fontsize=16)
blue = '#006FED'


def renormalise_log_weights(log_weights):
    log_total = logsumexp(log_weights)
    normalized_weights = np.exp(log_weights - log_total)
    return normalized_weights

# resample to get equal weights
def resample_equal(samples, aux, logwt, rstate):
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
    resampled_aux = aux[idx][perm]
    return resampled_samples, resampled_aux

def compute_pz(k,samples,num_nodes,left_node,right_node):
    Pz = []
    num_free_nodes = num_nodes - 2
    for sample in tqdm.tqdm(samples,desc='Pz'):
        free_nodes = sample[2:num_free_nodes+2]
        lengthscale = sample[1]
        nodes = np.pad(free_nodes, (1,1), 'constant', constant_values=(left_node, right_node))
        vals = sample[num_free_nodes+2:]
        gpkernel = 1 * RBF(length_scale=lengthscale, length_scale_bounds="fixed") #+ np.eye(len(nodes)) * 1e-10
        gaussian_process = GaussianProcessRegressor(kernel=gpkernel, optimizer=None, normalize_y=True)
        gaussian_process.fit(nodes.reshape(-1, 1),vals)
        interp_nodes = np.linspace(nodes[0], nodes[-1], 100)
        interp_vals = gaussian_process.predict(interp_nodes.reshape(-1, 1))
        res = gw.power_spectrum_k_array(interp_nodes, interp_vals, k)
        Pz.append(res)
    return np.array(Pz)

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

def plot_functional_posterior(vals = [],k_arr = [], intervals=[99.7, 95., 68.],weights=None,
                              ylabels=[r'$P_{\zeta}$', r'$\Omega_{\rm GW}$'],
                              aspect_ratio=(6, 4.5),
                              interval_cols=[('#006FED', 0.2), ('#006FED', 0.4), ('#006FED', 0.6)]):
    # given a function y = f(k|x) with x~Posterior samples, plot the posterior of y at k_arr, with symmetric credible intervals

    nfuncs = len(vals)

    if weights is None:
        weights = np.ones(len(vals[0].shape[0]))

    fig, ax = plt.subplots(1,nfuncs,figsize=(aspect_ratio[0]*nfuncs,aspect_ratio[1]),constrained_layout=True)
    if nfuncs == 1:
        ax = [ax]
    for i,val in enumerate(vals):
        y = val # so y should have shape (nsamples, nk)
        for j,interval in enumerate(intervals):
            y_low, y_high = np.percentile(y,[50-interval/2,50+interval/2],axis=0,weights=weights,
                                          method='inverted_cdf')
            ax[i].fill_between(k_arr[i],y_low,y_high,color=interval_cols[j])
        # medians = np.apply_along_axis(weighted_median, 0, val, weights)
        # ax[i].plot(k_arr[i], medians, color='#006FED', lw=2.5)
        ax[i].plot(k_arr[i],np.median(y,axis=0),color=blue,lw=2)
        ax[i].set_ylabel(ylabels[i])
    for x in ax:
        x.set(xscale='log', yscale='log', xlabel=r'$f\,{\rm [Hz]}$')
    return fig, ax

# model = str(sys.argv[1])
# Load the gravitational wave background data.
data = np.load('./spectra_0p66_interp.npz')
frequencies = data['frequencies']
gwb_model =  str(sys.argv[1])
Omegas = data[f'gw_{gwb_model}'] 
kstar = 1e-3
omks_sigma = Omegas * (0.05 * (np.log(frequencies / kstar))**2 + 0.1)
cov = np.diag(omks_sigma**2)
pk_arr = data['pk_arr']
pk_min, pk_max = min(pk_arr), max(pk_arr)
left_node = np.log10(pk_min)
right_node = np.log10(pk_max)
p_arr = np.logspace(left_node+0.001, right_node-0.001, 100)
pz_amp = data[f'pk_{gwb_model}']


num_thinned_samples = int(sys.argv[2])

p_arr_local = jnp.logspace(left_node+0.001, right_node-0.001, 200)

# Check current working directory for files matching the pattern
pattern = re.compile(rf'{gwb_model}_w0p66_gp_(\d+)\.npz')

# List all files in the current working directory
files_in_dir = os.listdir(os.getcwd())

# Filter files matching the pattern where n > 2
matching_files = [f for f in files_in_dir if pattern.match(f) and int(pattern.match(f).group(1)) > 2]
logz_list = []
logweights = []
gwb_samples = []
pz_samples = []
ws = []
N_nodes = []
if matching_files:
    print("Matching files found:")
    for file in matching_files:
        num_nodes = int(pattern.match(file).group(1))
        N_nodes.append(num_nodes)
        free_nodes = num_nodes - 2
        print(f"Processing file: {file} with {num_nodes} nodes")
        data = np.load(file)
        logz = data['logz'].item()
        print(f"Logz: {logz:.4f}")
        samples, logl, logwt, omegagw = data['samples'], data['logl'], data['logwt'], data['omegagw']
        equal_samples, equal_omegagw = resample_equal(samples, omegagw, logwt, np.random.RandomState(42))

        print(f"Shapes: {equal_samples.shape}, {equal_omegagw.shape}")
        idxs = np.arange(equal_samples.shape[0])
        thinned_samples_idxs = np.random.choice(idxs, num_thinned_samples, replace=False)
        thinned_samples = equal_samples[thinned_samples_idxs]
        thinned_weights = np.ones(num_thinned_samples)
        thinned_omegagw = equal_omegagw[thinned_samples_idxs]
        thinned_ws = thinned_samples[:,0]
        ws.append(thinned_ws)
        print(f"Shapes: {thinned_samples.shape}, {thinned_weights.shape}, {thinned_omegagw.shape}")
        pz = compute_pz(p_arr_local, thinned_samples, num_nodes , left_node, right_node)
        pz_samples.append(pz)
        gwb_samples.append(thinned_omegagw)
        logz_list.append(logz)
        logweight = np.ones(len(thinned_samples)) * logz
        print(f"pz_samples shape: {pz.shape}, gwb_samples shape: {omegagw.shape}, logweight shape: {logweight.shape}")
        logweights.append(logweight)

# no renormalise the logweights
logweights = np.concatenate(logweights)
weights = renormalise_log_weights(logweights)
gwb_samples = np.concatenate(gwb_samples)
pz_samples = np.concatenate(pz_samples)
ws = np.concatenate(ws)

print(f"pz_samples shape: {pz_samples.shape}, gwb_samples shape: {gwb_samples.shape}, weights shape: {weights.shape}")

fig, ax = plot_functional_posterior([pz_samples,gwb_samples],
                                    k_arr=[p_arr_local, frequencies],
                                    weights = weights,
                                    aspect_ratio=(6,4.5))
ax[0].loglog(pk_arr, pz_amp, color='k', lw=1.5)
ax[1].loglog(frequencies, Omegas, color='k', lw=1.5, label='Truth')
ax[1].errorbar(frequencies, Omegas, yerr=np.sqrt(np.diag(cov)), fmt='o', color='k', capsize=4.,alpha=0.5,markersize=2)
ax[1].legend()
k_mpc_f_hz = 2*np.pi * 1.03 * 10**14
for x in ax:
    x.set(xscale='log', yscale='log', xlabel=r'$f\,{\rm [Hz]}$')
    secax = x.secondary_xaxis('top', functions=(lambda x: x * k_mpc_f_hz, lambda x: x / k_mpc_f_hz))
    secax.set_xlabel(r"$k\,{\rm [Mpc^{-1}]}$",labelpad=10) 
plt.savefig(f'./{gwb_model}_0p66_gp_posterior.pdf',bbox_inches='tight')
plt.show()

w = 0.66
names = ['w']
labels = ['w']
bounds = [[0.2,0.99]]
ranges = dict(zip(names,bounds))
print(ranges)
# ws = samples[:,0]
# ws = ws[samples[:,0] < 0.9]
# w_weights = weights[samples[:,0] < 0.9]
# print(f"Shapes: {ws.shape}, {weights.shape}")
# # gd_samples = MCSamples(samples=samples[:,0], names=names, labels=labels,ranges=ranges,weights=normalized_weights,loglikes=logl)
# gd_samples = MCSamples(samples=ws, names=names, labels=labels,ranges=ranges,weights=w_weights)
gd_samples = MCSamples(samples=ws, names=names, labels=labels,ranges=ranges,weights=weights)
g = plots.get_subplot_plotter(subplot_size=6,subplot_size_ratio=2/3)
blue = '#006FED'
g.settings.title_limit_fontsize = 14
g.settings.axes_fontsize=16
g.settings.axes_labelsize=16
g.plot_1d(gd_samples, 'w', marker=w, marker_color=blue, colors=[blue],title_limit=2)
g.export(f'{gwb_model}_0p66_gp_1D_w.pdf')
ax = g.subplots[0,0]
ax.set_xlim(0.2, 1.0)
print(gd_samples.getMargeStats())


plt.figure(figsize=(6,4))
plt.plot()
plt.xlabel(r'Number of nodes')
plt.ylabel(r'$\log \mathcal{Z}$')
plt.plot(N_nodes, logz_list, '-.',color='k',alpha=0.9)
plt.scatter(N_nodes, logz_list, color='k',marker='x',s=20)
plt.xticks(N_nodes)
# Annotate each point with its logZ value
ax = plt.gca()
y_min = min(logz_list) - 5
y_max = max(logz_list) + 5
ax.set_ylim(y_min, y_max)
ax.set_xlim(min(N_nodes) - 0.5, max(N_nodes) + 0.5)
for x, y in zip(N_nodes, logz_list):
    plt.text(x+0.1, y-2, f'({y:.2f})', fontsize=12, ha='center', va='bottom')
plt.savefig(f'{gwb_model}_gp_logz.pdf',bbox_inches='tight')
plt.show()
