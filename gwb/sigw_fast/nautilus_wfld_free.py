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
import h5py

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

# open once, reuse for all likelihood calls
h5 = h5py.File("precomputed_kernels.h5", "r")
w_vals = h5["w_vals"][:]      # load in memory once; it's small
# kernel datasets stay on disk until you index them
dset1 = h5["kernel1"]
dset2 = h5["kernel2"]
tol = 1e-3

def get_kernels_from_file(w):
    # find index of the closest precomputed w
    idx = np.abs(w_vals - w).argmin()
    # print(f"Closest precomputed w: {w_vals[idx]:.6f} (target: {w:.6f})")
    if abs(w_vals[idx] - w) > tol:
        raise ValueError(f"No precomputed w within ±{tol}: nearest is {w_vals[idx]:.6f}")
    # pull the two rows
    k1 = dset1[idx, :]
    k2 = dset2[idx, :]
    return k1, k2

def compute_w(w,log10_f_rh,nodes,vals,frequencies,use_mp=False,nd=150,fref=1.,kernels_from_file=True):
    global cache_counter
    nd,ns1,ns2, darray,d1array,d2array, s1array,s2array = sd.arrays_w(w,frequencies,nd=nd)
    b = sd.beta(w)

    if kernels_from_file:
        try:
            kernel1, kernel2 = get_kernels_from_file(w,)
            cache_counter += 1
        except:
            # If the kernel is not found in the file, compute it
            kernel1, kernel2 = sd.kernel1_w(d1array, s1array, b), sd.kernel2_w(d2array, s2array, b)
    else:
        kernel1, kernel2 = sd.kernel1_w(d1array, s1array, b), sd.kernel2_w(d2array, s2array, b)

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
    N = len(xs)
    t = np.zeros(N)
    t[N-1] = xs[N-1]**(1./N)
    for n in range(N-2, -1, -1):
        t[n] = xs[n]**(1./(n+1)) * t[n+1]
    xs = t*(right_node - left_node) + left_node
    ys = params[free_nodes+1:]
    ys = ys * (y_max - y_min) + y_min
    return np.concatenate([[w],xs, ys])


def likelihood(params, log10_f_rh,free_nodes, left_node,right_node, frequencies, Omegas, omgw_sigma):
    # start = time.time()
    w = params[0]
    # log10_f_rh = params[1]
    nodes = params[1:free_nodes+1]
    nodes = np.pad(nodes, (1,1), 'constant', constant_values=(left_node, right_node))
    vals = params[free_nodes+1:]    
    omegagw = compute_w(w, log10_f_rh, nodes, vals, frequencies, use_mp=False, nd=nd,kernels_from_file=True)
    diff = (omegagw - Omegas)
    ll = -0.5 * np.sum(diff**2 / omgw_sigma**2)
    # print(f"likelihood took {time.time()-start:.2f} seconds")
    return ll, omegagw

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

    w_min = 0.1
    w_max = 0.99
    log10_f_rh = -5.

    ndim = 1 + free_nodes + num_nodes

    prior_transform = partial(prior,w_min=w_min, w_max=w_max,  
                              free_nodes=free_nodes,
                              left_node=left_node, right_node=right_node,
                              y_min=y_min, y_max=y_max)
    
    loglikelihood = partial(likelihood,log10_f_rh=log10_f_rh, 
                            free_nodes=free_nodes, left_node=left_node, right_node=right_node,
                            frequencies=frequencies, Omegas=Omegas, omgw_sigma=omks_sigma)

    sampler = Sampler(prior_transform, loglikelihood, ndim, pass_dict=False,filepath=f'{gwb_model}_w0p66_free_{num_nodes}.h5',pool=(None,4))

    sampler.run(verbose=True, f_live=0.01,n_like_max=int(2e6))
    print('log Z: {:.4f}'.format(sampler.log_z))

    samples, logl, logwt, blobs = sampler.posterior(return_blobs=True)
    print(f"Max and min loglike: {np.max(logl)}, {np.min(logl)}")
    np.savez(f'{gwb_model}_w0p66_free_{num_nodes}.npz', samples=samples, logl=logl, logwt=logwt,logz=sampler.log_z,omegagw=blobs)
    print("Nested sampling complete")
    print(f"Cached kernel was used {cache_counter} times")

if __name__ == "__main__":
    main()