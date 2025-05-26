import numpy as np
import h5py
from libraries import sdintegral_numba as sd
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import h5py
from sigwfast import sigwfast_mod as gw
import time 
import sys

tol = 1e-3

cache_counter = 0

# open once, reuse for all likelihood calls
h5 = h5py.File("precomputed_kernels.h5", "r")
w_vals = h5["w_vals"][:]      # load in memory once; it's small
# kernel datasets stay on disk until you index them
dset1 = h5["kernel1"]
dset2 = h5["kernel2"]

def get_kernels_from_file(w):
    # find index of the closest precomputed w
    idx = np.abs(w_vals - w).argmin()
    print(f"Closest precomputed w: {w_vals[idx]:.6f} (target: {w:.6f})")
    if abs(w_vals[idx] - w) > tol:
        raise ValueError(f"No precomputed w within ±{tol}: nearest is {w_vals[idx]:.6f}")
    # pull the two rows
    k1 = dset1[idx, :]
    k2 = dset2[idx, :]
    return k1, k2

# Example usage inside your likelihood:
test_w = 0.5


OMEGA_R = 4.2 * 10**(-5)
CG = 0.39
rd_norm = CG * OMEGA_R 
nd = 150

def compute_w(w,log10_f_rh,nodes,vals,frequencies,use_mp=False,nd=150,fref=1.,compute_kernels=True):
    global cache_counter
    nd,ns1,ns2, darray,d1array,d2array, s1array,s2array = sd.arrays_w(w,frequencies,nd=nd)
    b = sd.beta(w)
    if compute_kernels:
        kernel1, kernel2 = sd.kernel1_w(d1array, s1array, b), sd.kernel2_w(d2array, s2array, b)
    else:
        kernel1, kernel2 = get_kernels_from_file(w,)
        cache_counter+= 1


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

def bpl(p, pstar=5e-4, n1=2, n2=-1, sigma=2):
    nir = n1
    pl1 = (p / pstar) ** nir
    nuv = (n2 - n1) / sigma
    pl2 = (1 + (p / pstar) ** sigma) ** nuv
    return 1e-2 * pl1 * pl2

w = 0.5
log10_f_rh = -5.
nodes_lr = np.array([ -5., 0.]) 
nodes = np.linspace(nodes_lr[0],nodes_lr[1],100) 
vals = np.log10(bpl(10**nodes)) 

import matplotlib.pyplot as plt

data = np.load('./spectra_0p5_interp.npz')
frequencies = data['frequencies']
gwb_model = 'bpl'
Omegas = data[f'gw_{gwb_model}'] 

niter = int(sys.argv[1]) if len(sys.argv) > 1 else 10

ws = np.linspace(0.1,0.9,niter-1)
ws = np.concatenate((ws,[w]))

start = time.time()
for i in tqdm(range(niter)):
    gw1 = compute_w(ws[i],log10_f_rh,nodes,vals,frequencies,use_mp=False,nd=150,fref=1.,compute_kernels=True)
print(f"Caclulate kernel: timing for {niter} iterations took {time.time()-start:.4f}s with {-(start-time.time())/niter:.4f}s average ")

start = time.time()
for i in tqdm(range(niter)):
    gw2 = compute_w(ws[i],log10_f_rh,nodes,vals,frequencies,use_mp=False,nd=150,fref=1.,compute_kernels=False)
print(f"Precomputed kernel: timing for {niter} iterations took {time.time()-start:.4f}s with {-(start-time.time())/niter:.4f}s average ")

print(f"Cache hits: {cache_counter}")

# plt.figure(figsize=(10, 6))
# plt.plot(frequencies, 1.02 * gw1, label='Precomputed Kernels')
# plt.plot(frequencies, 1.05 * gw2, label='Calculated Kernels')
# plt.plot(frequencies, Omegas, label='Original')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Omega GW')
# plt.legend()
# # plt.show()
