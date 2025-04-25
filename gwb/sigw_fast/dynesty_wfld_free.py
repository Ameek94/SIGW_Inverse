from functools import partial
import os
import sys
import time
import numpy as np
from sigwfast import sigwfast_mod as gw
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from nautilus import Sampler
import math
from scipy.special import logsumexp
from dynesty import DynamicNestedSampler

OMEGA_R = 4.2 * 10**(-5)
CG = 0.39
rd_norm = CG * OMEGA_R 
nd = 150
SQRTEPS = math.sqrt(float(np.finfo(np.float64).eps))

from libraries import sdintegral_numba as sd

from collections import OrderedDict

# Global cache for storing kernels keyed by rounded w
kernel_cache = OrderedDict()

cache_counter = 0

def get_kernels(w, d1array, s1array, d2array, s2array, tolerance=3):
    global cache_counter

    # Round w to the desired tolerance (number of decimals)
    key = round(w, tolerance)
    # If already cached, update the order and return
    if key in kernel_cache:
        cache_counter += 1
        kernel_cache.move_to_end(key)
        return kernel_cache[key]
    
    # Otherwise compute the kernels
    b = sd.beta(w)
    kernel1 = sd.kernel1_w(d1array, s1array, b)
    kernel2 = sd.kernel2_w(d2array, s2array, b)
    
    # If cache size is 4, remove the least recently used entry
    if len(kernel_cache) >= 500:
        kernel_cache.popitem(last=False)
    
    # Store and return the result
    kernel_cache[key] = (kernel1, kernel2)
    return kernel_cache[key]

def compute_w(w,log10_f_rh,nodes,vals,frequencies,use_mp=False,nd=150,fref=1.):
    nd,ns1,ns2, darray,d1array,d2array, s1array,s2array = sd.arrays_w(w,frequencies,nd=nd)
    b = sd.beta(w)
    kernel1, kernel2 = get_kernels(w, d1array, s1array, d2array, s2array)
    # kernel1 = sd.kernel1_w(d1array, s1array, b)
    # kernel2 = sd.kernel2_w(d2array, s2array, b)
    nk = len(frequencies)
    Integral = np.empty_like(frequencies)
    Integral = gw.compute_w_k_array(nodes = nodes, vals = vals, nk = nk,komega = frequencies, 
                                            kernel1 = kernel1, kernel2 = kernel2, d1array=d1array,
                                            s1array=s1array, d2array=d2array, s2array=s2array,
                                            darray=darray, nd = nd, ns1 = ns1, ns2 = ns2)
    f_rh = 10**log10_f_rh
    two_b = 2*b
    norm = rd_norm * (frequencies)**(-2*b) *  (f_rh/fref)**(two_b)   
    OmegaGW = norm * Integral
    return OmegaGW

def prior(cube, w_min, w_max,free_nodes, left_node,right_node, y_min, y_max):
    params = cube.copy()
    w = params[0]
    w = w * (w_max - w_min) + w_min
    xs = params[1:free_nodes+1]
    xs = xs * (right_node - left_node) + left_node
    ys = params[free_nodes+1:]
    ys = ys * (y_max - y_min) + y_min
    return np.concatenate([[w],xs, ys])

def likelihood(params, log10_f_rh,free_nodes, left_node,right_node, frequencies, Omegas, cov):
    w = params[0]
    # log10_f_rh = params[1]
    nodes = params[1:free_nodes+1]
    nodes = np.pad(nodes, (1,1), 'constant', constant_values=(left_node, right_node))
    vals = params[free_nodes+1:]    
    omegagw = compute_w(w, log10_f_rh, nodes, vals, frequencies, use_mp=False, nd=nd)
    diff = omegagw - Omegas
    return -0.5 * np.dot(diff, np.linalg.solve(cov, diff)), omegagw


def renormalise_log_weights(log_weights):
    log_total = logsumexp(log_weights)
    normalized_weights = np.exp(log_weights - log_total)
    return normalized_weights

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

def main():
    # Load the gwb data from file
    data = np.load('./spectra_0p66_interp.npz')
    frequencies = data['frequencies']
    gwb_model = str(sys.argv[1])
    Omegas = data[f'gw_{gwb_model}'] 
    kstar = 1e-3
    omks_sigma = Omegas * (0.05 * (np.log(frequencies / kstar))**2 + 0.1)
    cov = np.diag(omks_sigma**2)

    num_nodes = int(sys.argv[2])
    free_nodes = num_nodes - 2
    pk_arr = data['pk_arr']
    pk_min, pk_max = min(pk_arr), max(pk_arr)
    # pk_min, pk_max = np.array(min(frequencies) / fac), np.array(max(frequencies) * fac)
    left_node = np.log10(pk_min)
    right_node = np.log10(pk_max)
    y_max = 0.
    y_min = -8.

    w_min = 0.33
    w_max = 0.99
    log10_f_rh = -5.

    ndim = 1 + free_nodes + num_nodes

    prior_transform = partial(prior,w_min=w_min, w_max=w_max,  
                              free_nodes=free_nodes,
                              left_node=left_node, right_node=right_node,
                              y_min=y_min, y_max=y_max)
    
    loglikelihood = partial(likelihood,log10_f_rh=log10_f_rh, 
                            free_nodes=free_nodes, left_node=left_node, right_node=right_node,
                            frequencies=frequencies, Omegas=Omegas, cov=cov)

    # sampler = Sampler(prior_transform, loglikelihood, ndim, pass_dict=False,filepath=f'{gwb_model}_w0p66_free_{num_nodes}.h5',pool=(None,4))
    sampler = DynamicNestedSampler(loglikelihood,prior_transform,ndim=ndim,
                                       sample='rwalk',blob=True) #(likelihood, prior, ndim, nlive=nlive, sample='rslice')
    sampler.run_nested(dlogz_init=0.05, print_progress=True,maxcall=int(2e6),checkpoint_file='{gwb_model}_w0p66_free_dynesty_{num_nodes}.save')

    res = sampler.results  # type: ignore # grab our results
    mean = res['logz'][-1]
    logz_err = res['logzerr'][-1]

    print(f"Mean logz: {mean:.4f} +/- {logz_err:.4f}")
    # sampler.run(verbose=True, f_live=0.001,n_like_max=int(2e6),max)
    # print('log Z: {:.4f}'.format(sampler.log_z))
    samples = res['samples']
    logwt = res['logwt']
    logl = res['logl']
    omegagw = res['blob']
    print(f"Max and min loglike: {np.max(logl)}, {np.min(logl)}")
    np.savez(f'{gwb_model}_w0p66_free_dynesty_{num_nodes}.npz', samples=samples, logl=logl, logwt=logwt,logz=mean,omegagw=omegagw)
    print("Nested sampling complete")
    print(f"Cached kernel was used {cache_counter} times")

if __name__ == "__main__":
    main()