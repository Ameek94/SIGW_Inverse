import jax.numpy as jnp
from jax import vmap, jit
import matplotlib.pyplot as plt
from jaxns.internals.random import resample_indicies

# minor modifications of https://github.com/martinjankowiak/saasbo/blob/main/util.py
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

def plot_mcmc_spectra(samples,):
    pass
@jit
def unit_transform(x,bounds):
    """
    Transform array from original domain given by bounds to [0,1]. 
    """
    ux = (x - bounds[0])/(bounds[1]-bounds[0])
    return ux
@jit
def unit_untransform(x,bounds):
    """
    Transform array to original domain given by bounds from [0,1]
    """
    xu = x*(bounds[1]-bounds[0]) + bounds[0]
    return xu


def plot_functional_posterior(funcs,samples,k_arr = [], intervals = [95,68],ylabels=[r'$P_{\zeta}$',r'$\Omega_{\rm GW}$'],aspect_ratio = (6,4), interval_cols = [('r',0.2),('b',0.5)]):
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
        ax[i].plot(k_arr[i],np.median(y,axis=0),color='b',lw=2)
        ax[i].set_ylabel(ylabels[i])
    for x in ax:
        x.set(xscale='log',yscale='log',xlabel=r'$k$')
    return fig, ax 

# OMEGA_R = 4.2 * 10**(-5)
# CG = 0.39
# rd_norm = CG*OMEGA_R 
# nd = 150
# from sigw_fast.libraries import sdintegral_numba as sd
# left_node,right_node = -5.0, -1.30103
# def compute_w(frequencies,samples,use_mp=False,nd=150,fref=1.):
#     OmegaGW = []
#     for sample in samples:
#         w, log10_f_rh = sample[:2]
#         free_nodes = sample[2:num_nodes-2]
#         nodes = np.pad(free_nodes, (1,1), 'constant', constant_values=(left_node, right_node))
#         vals = sample[num_nodes:]
#         nd,ns1,ns2, darray,d1array,d2array, s1array,s2array = sd.arrays_w(w,frequencies,nd=nd)
#         b = sd.beta(w)
#         kernel1 = sd.kernel1_w(d1array, s1array, b)
#         kernel2 = sd.kernel2_w(d2array, s2array, b)
#         nk = len(frequencies)
#         Integral = np.empty_like(frequencies)
#         Integral = gw.compute_w_k_array(nodes = nodes, vals = vals, nk = nk,komega = frequencies, 
#                                             kernel1 = kernel1, kernel2 = kernel2, d1array=d1array,
#                                             s1array=s1array, d2array=d2array, s2array=s2array,
#                                             darray=darray, nd = nd, ns1 = ns1, ns2 = ns2)
#         f_rh = 10**log10_f_rh
#         two_b = 2*b
#         norm = rd_norm * (frequencies)**(-2*b) *  (f_rh/fref)**(two_b)   
#         OmegaGW.append(norm * Integral)
#     return np.array(OmegaGW) # OmegaGW has shape (nsamples, nk)

# def compute_pz(k,samples):
#     Pz = []
#     for sample in samples:
#         free_nodes = sample[2:num_nodes-2]
#         nodes = np.pad(free_nodes, (1,1), 'constant', constant_values=(left_node, right_node))
#         vals = sample[num_nodes:]
#         res = gw.power_spectrum_k_array(nodes, vals, k)
#         Pz.append(res)
#     return np.array(Pz) # Pz has shape (nsamples, nk)

# print(len(samples_4['samples']))
# thinning = len(samples_4['samples']) // 32
# thinned_samples = samples_4['samples'][::thinning]
# print(len(thinned_samples))

# data = np.load('./spectra_0p8.npz')
# frequencies = data['frequencies']
# gwb_model = 'bpl'
# Omegas = data[f'gw_{gwb_model}'] 
# kstar = 1e-3
# omks_sigma = Omegas*( 0.1*(np.log(frequencies/kstar))**2 + 0.1) # 2% error at kstar + more towards edges
# cov = np.diag(omks_sigma**2)
# kpz = np.logspace(left_node,right_node,50)
# k_arr = [kpz,frequencies]
# fig, ax = plot_functional_posterior([compute_pz,compute_w],thinned_samples,k_arr = k_arr,intervals=[95,68],aspect_ratio=(6,4))
# ax[1].errorbar(frequencies, Omegas, yerr=np.sqrt(np.diag(cov)), fmt="", color='k', label='data',capsize=2,ecolor='k');
