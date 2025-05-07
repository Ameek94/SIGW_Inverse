from functools import partial
import os
# print OMP NUM Threads
print(f'NUM THREADS = {os.environ.get("OMP_NUM_THREADS")}')
import sys
import time
import numpy as np
from sigwfast import sigwfast_mod as gw
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from nautilus import Sampler
import math
import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior, SortedUniformPrior

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

def get_kernels(w, d1array, s1array, d2array, s2array, tolerance=8):
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
    
    # If cache size exceeds, remove the least recently used entry
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

def prior(cube, w_min, w_max,n_min,n_max, logA_min, logA_max,pstar_min, pstar_max,sigma_min, sigma_max):
    params = cube.copy()
    w = params[0]
    w = w * (w_max - w_min) + w_min
    n1 = params[1]
    n1 = n1 * (n_max - n_min) + n_min
    n2 = params[2]
    n2 = n2 * (n_max - n_min) + n_min
    logA = params[3]
    logA = logA * (logA_max - logA_min) + logA_min
    pstar = params[4]
    pstar = pstar * (pstar_max - pstar_min) + pstar_min
    sigma = params[5]
    sigma = sigma * (sigma_max - sigma_min) + sigma_min
    return np.array([w, n1, n2, logA, pstar, sigma])


def pz(p,logA = -2., pstar=np.log10(5e-4), n1=2, n2=-1, sigma=2):
    pstar = 10**pstar
    nir = n1
    pl1 = (p / pstar) ** nir
    nuv = (n2 - n1) / sigma
    pl2 = (1 + (p / pstar) ** sigma) ** nuv
    return 10**(logA) * pl1 * pl2

def likelihood(params, log10_f_rh, nodes, frequencies, Omegas, cov):
    w = params[0]
    # log10_f_rh = params[1]
    vals = np.log10(pz(p=10**nodes,pstar=params[4],n1=params[1],n2=params[2],sigma=params[5], logA = params[3]))
    omegagw = compute_w(w, log10_f_rh, nodes, vals, frequencies, use_mp=False, nd=nd)
    diff = omegagw - Omegas
    return -0.5 * np.dot(diff, np.linalg.solve(cov, diff)), omegagw

def dumper(live, dead, logweights, logZ, logZerr):
    print("Last dead point:", dead[-1])


def main():
    # Load the gwb data from file
    data = np.load('./spectra_0p66_interp.npz')
    frequencies = data['frequencies']
    gwb_model = str(sys.argv[1])
    Omegas = data[f'gw_{gwb_model}'] 
    kstar = 1e-3
    omks_sigma = Omegas * (0.05 * (np.log(frequencies / kstar))**2 + 0.1)
    cov = np.diag(omks_sigma**2)

    num_nodes = 20
    pk_arr = data['pk_arr']
    pk_min, pk_max = min(pk_arr), max(pk_arr)
    # pk_min, pk_max = np.array(min(frequencies) / fac), np.array(max(frequencies) * fac)
    left_node = np.log10(pk_min)
    right_node = np.log10(pk_max)
    nodes = np.linspace(left_node,right_node,num_nodes)

    w_min = 0.1
    w_max = 0.99
    n_min = -4.
    n_max = 4.
    logA_min = -5.
    logA_max = 1.
    log10_f_rh = -5.
    pstar_min = -6.
    pstar_max = -2.
    sigma_min = 0.5
    sigma_max = 3.


    prior_transform = partial(prior,w_min=w_min, w_max=w_max,  
                              n_min=n_min, n_max=n_max,
                              logA_max=logA_max, logA_min=logA_min,
                              pstar_min=pstar_min, pstar_max=pstar_max,
                              sigma_min=sigma_min, sigma_max=sigma_max)

    
    loglikelihood = partial(likelihood,log10_f_rh=log10_f_rh, 
                            nodes=nodes,
                            frequencies=frequencies, Omegas=Omegas, cov=cov)

    nDims = 6
    nDerived = len(frequencies)
    settings = PolyChordSettings(nDims, nDerived)
    settings.file_root = f'{gwb_model}_pchord_full'
    settings.nlive = 15 * nDims
    settings.do_clustering = True
    settings.read_resume = True
    settings.precision_criterion = 0.005

    start = time.time()
    output = pypolychord.run_polychord(loglikelihood, nDims, nDerived, settings
                                       , prior_transform, dumper)

    end = time.time()
    print(f"Time taken: {end - start:.4f}")
    paramnames = [('w', r'w')]
    paramnames += [('n1', r'n1'), ('n2', r'n2'), ('logA', r'logA'), ('pstar', r'pstar'), ('sigma', r'sigma')]
    paramnames += [(f'gw_{i}', f'gw_{i}') for i in range(len(frequencies))]
    output.make_paramnames_files(paramnames)

    posterior = output.posterior

    import getdist.plots
    g = getdist.plots.getSubplotPlotter(subplot_size=3.5)
    blue = '#006FED'
    g.settings.title_limit_fontsize = 14
    g.settings.axes_fontsize=16
    g.settings.axes_labelsize=18
    g.triangle_plot(posterior, ['w', 'n1', 'n2','logA','pstar','sigma'], filled=True, 
                   title_limit=1,markers={'w': 2/3, 'n1': 2, 'n2':-1, 'logA':-2, 'pstar': np.log10(5e-4), 'sigma': 2. })
    # g.plot_1d(posterior, param = 'w', marker=2/3, marker_color=blue, colors=[blue],title_limit=1)
    g.export(settings.file_root + '_.pdf')

    print("Nested sampling complete")
    print(f"Cached kernel was used {cache_counter} times")

if __name__ == "__main__":
    main()