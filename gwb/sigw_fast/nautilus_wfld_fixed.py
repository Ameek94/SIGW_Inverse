from functools import partial
import os
import sys
sys.path.append('../')
import time
from sigw_fast.RD import compute
import numpy as np
from scipy.interpolate import interp1d
from sigw_fast.sigwfast import sigwfast_mod as gw
from sigw_fast.sigwfast_fortran import compute_rd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from nautilus import Sampler
import math

OMEGA_R = 4.2 * 10**(-5)
CG = 0.39
rd_norm = CG * OMEGA_R 
nd = 150
SQRTEPS = math.sqrt(float(np.finfo(np.float64).eps))

from sigw_fast.libraries import sdintegral_numba as sd

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
    if len(kernel_cache) >= 100:
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

def prior(cube, w_min, w_max, y_min, y_max):
    params = cube.copy()
    w = params[0]
    w = w * (w_max - w_min) + w_min
    ys = params[1:]
    ys = ys * (y_max - y_min) + y_min
    return np.concatenate([[w], ys])

def likelihood(params, log10_f_rh, nodes, frequencies, Omegas, cov):
    w = params[0]
    # log10_f_rh = params[1]
    vals = params[1:]    
    omegagw = compute_w(w, log10_f_rh, nodes, vals, frequencies, use_mp=False, nd=nd)
    diff = omegagw - Omegas
    return -0.5 * np.dot(diff, np.linalg.solve(cov, diff))

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
    data = np.load('./spectra_0p8_interp.npz')
    frequencies = data['frequencies']
    gwb_model = str(sys.argv[1])
    Omegas = data[f'gw_{gwb_model}'] 
    kstar = 1e-3
    omks_sigma = Omegas * (0.05 * (np.log(frequencies / kstar))**2 + 0.1)
    cov = np.diag(omks_sigma**2)

    num_nodes = int(sys.argv[2])
    # free_nodes = num_nodes - 2
    fac = 5
    pk_min, pk_max = np.array(min(frequencies) / fac), np.array(max(frequencies) * fac)
    left_node = np.log10(pk_min)
    right_node = np.log10(pk_max)
    nodes = np.linspace(left_node, right_node, num_nodes)
    free_nodes = 0
    y_max = -2
    y_min = -6

    w_min = 0.01
    w_max = 0.99
    log10_f_rh = -5.

    ndim = 1 + free_nodes + num_nodes

    prior_transform = partial(prior,w_min=w_min, w_max=w_max,  
                              y_min=y_min, y_max=y_max)
    
    loglikelihood = partial(likelihood,log10_f_rh=log10_f_rh, nodes=nodes,
                            frequencies=frequencies, Omegas=Omegas, cov=cov)

    sampler = Sampler(prior_transform, loglikelihood, ndim, pass_dict=False,filepath=f'{gwb_model}_wfld_fixed_{num_nodes}.h5')

    sampler.run(verbose=True, f_live=0.002,n_like_max=int(5e6))
    print('log Z: {:.2f}'.format(sampler.log_z))

    samples, logl, logwt = sampler.posterior()

    np.savez(f'{gwb_model}_wfld_fixed_{num_nodes}.npz', samples=samples, logl=logl, logwt=logwt,logz=sampler.log_z)
    print("Nested sampling complete")

    print(f"Cached kernel was used {cache_counter} times")

    rstate = np.random.default_rng(100000)
    samples, lp = resample_equal(samples, logl, logwt, rstate=rstate)
    print("Obtained equally weighted samples")
    print(f"Max and min logprob: {np.max(lp)}, {np.min(lp)}")
    # np.savez(f'{gwb_model}_wfld_{num_nodes}.npz', samples=samples, logl=lp)

    p_arr = np.geomspace(pk_min * 1.001, pk_max * 0.999, 100, endpoint=True)
    ws = samples[:, 0]
    xs = nodes 
    ys = samples[:, 1:]
    thinning = len(samples) // 128
    cmap = matplotlib.colormaps['Reds']
    ys = ys[::thinning]
    xs = xs[::thinning]
    lp = lp[::thinning] 
    lp_min, lp_max = np.min(lp), np.max(lp)
    cols = (lp - lp_min) / (lp_max - lp_min)
    norm = colors.Normalize(lp_min, lp_max)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')

    def get_pz_omega(w, log10_f_rh, x, y):
        pz_amps = gw.power_spectrum_k_array(x, y, p_arr)
        gwb_res = compute_w(w, log10_f_rh, x, y, frequencies, use_mp=False, nd=150)
        return pz_amps, gwb_res

    for i, y in enumerate(ys):
        w = ws[i]
        x = nodes
        pz_amps, gwb_amps = get_pz_omega(w, log10_f_rh, x, y)
        ax1.loglog(p_arr, pz_amps, alpha=0.25, color=cmap(cols[i]))
        ax1.scatter(10**(x), 10**(ys[i]), s=16, alpha=0.5, color=cmap(cols[i]))
        ax2.loglog(frequencies, gwb_amps, alpha=0.25, color=cmap(cols[i]))

    pz_amp = data[f'pk_{gwb_model}']
    p_arr = data['pk_arr']
    ax1.loglog(p_arr, pz_amp, color='k', lw=1.5)
    ax2.loglog(frequencies, Omegas, color='k', lw=1.5, label='Truth')
    ax2.legend()
    ax1.set_ylabel(r'$P_{\zeta}(k)$')
    ax1.set_xlabel(r'$k$')
    ax2.errorbar(frequencies, Omegas, yerr=np.sqrt(np.diag(cov)), fmt="", color='k', label='data', capsize=2, ecolor='k')
    ax2.set_ylabel(r'$\Omega_{\mathrm{GW}}(k)$')
    ax2.set_xlabel(r'$k$')
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=[ax1, ax2], label='Logprob')
    plt.savefig(f"{gwb_model}_wfld_nautilus_fixed_{num_nodes}.pdf")


if __name__ == "__main__":
    main()