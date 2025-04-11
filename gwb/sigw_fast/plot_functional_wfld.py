import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from getdist import plots, MCSamples, loadMCSamples
import sys
sys.path.append('../')
import tqdm

# Set matplotlib parameters
font = {'size': 16, 'family': 'serif'}
axislabelfontsize = 'large'
matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=True)
matplotlib.rc('legend', fontsize=16)

# Load data
data = np.load('./spectra_0p8.npz')
frequencies = data['frequencies']
gwb_model = str(sys.argv[1])
Omegas = data[f'gw_{gwb_model}'] 
kstar = 1e-3
omks_sigma = Omegas * (0.1 * (np.log(frequencies / kstar))**2 + 0.1)
cov = np.diag(omks_sigma**2)
fac = 5
pk_min, pk_max = np.array(min(frequencies) / fac), np.array(max(frequencies) * fac)
left_node = np.log10(pk_min)
right_node = np.log10(pk_max)
p_arr = np.logspace(left_node+0.001, right_node-0.001, 100)

blue = '#006FED'
def plot_functional_posterior(funcs,samples,k_arr = [], intervals = [95,68]
                              ,ylabels=[r'$P_{\zeta}$',r'$\Omega_{\rm GW}$']
                              ,aspect_ratio = (6,4), 
                              interval_cols = [(blue,0.25),(blue,0.6)]):
    # given a function y = f(k|x) with x~Posterior samples, plot the posterior of y at k_arr, with symmetric credible intervals

    nfuncs = len(funcs)

    fig, ax = plt.subplots(1,nfuncs,figsize=(aspect_ratio[0]*nfuncs,aspect_ratio[1]),constrained_layout=True)
    if nfuncs == 1:
        ax = [ax]
    for i,func in enumerate(funcs):
        y = func(k_arr[i],samples) # so y should have shape (nsamples, nk)
        for j,interval in enumerate(intervals):
            y_low, y_high = np.percentile(y,[50-interval/2,50+interval/2],axis=0)
            ax[i].fill_between(k_arr[i],y_low,y_high,color=interval_cols[j])
        ax[i].plot(k_arr[i],np.median(y,axis=0),color=blue,lw=2)
        ax[i].set_ylabel(ylabels[i])
    for x in ax:
        x.set(xscale='log',yscale='log',xlabel=r'$k$')
    return fig, ax 

# Define the functions to be plotted
OMEGA_R = 4.2 * 10**(-5)
CG = 0.39
rd_norm = CG*OMEGA_R 
nd = 150
from sigw_fast.sigwfast import sigwfast_mod as gw
from sigw_fast.libraries import sdintegral_numba as sd

num_nodes = int(sys.argv[2])

def compute_w(frequencies,samples,use_mp=False,nd=150,fref=1.):
    OmegaGW = []
    for sample in tqdm.tqdm(samples,desc='OmegaGW'):
        w, log10_f_rh = sample[:2]
        free_nodes = sample[2:num_nodes-2]
        nodes = np.pad(free_nodes, (1,1), 'constant', constant_values=(left_node, right_node))
        vals = sample[num_nodes:]
        nd,ns1,ns2, darray,d1array,d2array, s1array,s2array = sd.arrays_w(w,frequencies,nd=nd)
        b = sd.beta(w)
        kernel1 = sd.kernel1_w(d1array, s1array, b)
        kernel2 = sd.kernel2_w(d2array, s2array, b)
        nk = len(frequencies)
        Integral = np.empty_like(frequencies)
        Integral = gw.compute_w_k_array(nodes = nodes, vals = vals, nk = nk,komega = frequencies, 
                                            kernel1 = kernel1, kernel2 = kernel2, d1array=d1array,
                                            s1array=s1array, d2array=d2array, s2array=s2array,
                                            darray=darray, nd = nd, ns1 = ns1, ns2 = ns2)
        f_rh = 10**log10_f_rh
        two_b = 2*b
        norm = rd_norm * (frequencies)**(-2*b) *  (f_rh/fref)**(two_b)   
        OmegaGW.append(norm * Integral)
    return np.array(OmegaGW) # OmegaGW has shape (nsamples, nk)

def compute_pz(k,samples):
    Pz = []
    for sample in tqdm.tqdm(samples,desc='Pz'):
        free_nodes = sample[2:num_nodes-2]
        nodes = np.pad(free_nodes, (1,1), 'constant', constant_values=(left_node, right_node))
        vals = sample[num_nodes:]
        res = gw.power_spectrum_k_array(nodes, vals, k)
        Pz.append(res)
    return np.array(Pz) # Pz has shape (nsamples, nk)

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

sample_data = np.load(f'{gwb_model}_wfld_free_{num_nodes}.npz')
samples = sample_data['samples']
logwt = sample_data['logwt']
logl = sample_data['logl']
from scipy.special import logsumexp
log_total = logsumexp(logwt)
# Subtract the normalization constant and exponentiate to obtain normalized weights
normalized_weights = np.exp(logwt - log_total)
equal_samples, _ = resample_equal(samples, logl, logwt, np.random.RandomState(0))

# use 256 samples for plotting 
thinning = max(1,len(equal_samples) // 512)
equal_samples = equal_samples[::thinning]

fig, ax  = plot_functional_posterior([compute_pz,compute_w],equal_samples
                                     ,k_arr = [p_arr,frequencies],intervals=[95,68],aspect_ratio=(7,5))
plt.savefig(f'{gwb_model}_wfld_{num_nodes}_posterior.pdf',bbox_inches='tight')

# getdist plots
names = ['w','log10_f_rh']
labels = ['w','\\log_{10} f_{rh}']
names+= [f'x_{i}' for i in range(num_nodes-2)]
labels+= [f'x_{i}' for i in range(num_nodes-2)]
names+= [f'y_{i}' for i in range(num_nodes)]
labels+= [f'y_{i}' for i in range(num_nodes)]
bounds = [[0.6,0.9],[-5.5,-4.5]]
bounds+= [[left_node, right_node] for i in range(num_nodes-2)]
bounds+=[[-6,-2] for i in range(num_nodes)]
ranges = dict(zip(names,bounds))
print(ranges)
gd_sample = MCSamples(samples=samples, names=names, labels=labels,ranges=ranges,weights=normalized_weights,loglikes=logl)
g = plots.get_subplot_plotter(subplot_size=2.5)
markers = {'w': 0.8, 'log10_f_rh': -5.0}
g.triangle_plot(gd_sample,filled=True,markers=markers,title_limit=1)
g.export(f'{gwb_model}_wfld_{num_nodes}_triangle.pdf')