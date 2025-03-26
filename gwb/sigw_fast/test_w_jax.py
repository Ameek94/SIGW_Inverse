import os
os.environ["OMP_NUM_THREADS"] = "8"
import sys
import time
from EOS import compute_w
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from libraries import sdintegral_numba as sd
from sigwfast import sigwfast_mod as gw

def test_pz(p,pstar=5e-4,n1=2,n2=-1,sigma=2):
    nir = n1
    pl1 = (p/pstar)**nir
    nuv = (n2 - n1)/sigma
    pl2 = (1+(p/pstar)**sigma)**nuv
    return 1e-2 * pl1 * pl2

frequencies = np.logspace(-5,-2, 50)

niter = int(sys.argv[1])

start = time.time()
for i in range(niter):
    omega_gw1 = compute_w(test_pz, frequencies,Use_Cpp=False,w = 0.6,nd=200)# compute(test_pz, frequencies,Use_Cpp=True)
end = time.time()
print(f"Using python and CPP code, {niter} iterations took {end-start:.4f} seconds, average {(end-start)/niter:.4f} seconds, {(end-start)/niter / len(frequencies):.4f} per frequency")

nodes = jnp.linspace(-5,-1,10)
vals = jnp.log10(test_pz(10**nodes))
from libraries.sigwfast_jax import compute_w as compute_w_jax
start = time.time()
for i in range(niter):
    omega_gw2 = compute_w_jax(w = 0.6,log10_f_rh = 0.,nodes=nodes,vals=vals,frequencies=frequencies,use_mp=False,nd=200)
end = time.time()
print(f"Using JAX code, {niter} iterations took {end-start:.4f} seconds, average {(end-start)/niter:.4f} seconds, {(end-start)/niter / len(frequencies):.4f} per frequency")

OMEGA_R = 4.2 * 10**(-5)
CG = 0.39
rd_norm = CG * OMEGA_R 
def compute_w(w,log10_f_rh,nodes,vals,frequencies,use_mp=False,nd=150,fref=1.):
    nd,ns1,ns2, darray,d1array,d2array, s1array,s2array = sd.arrays_w(w,frequencies,nd=nd)
    b = sd.beta(w)
    # kernel1, kernel2 = get_kernels(w, d1array, s1array, d2array, s2array)
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
    OmegaGW = norm * Integral
    return OmegaGW

start = time.time()
for i in range(niter):
    omega_gw3 = compute_w(0.6,0.,nodes,vals,frequencies,use_mp=False,nd=200)
end = time.time()
print(f"Using Fortran code, {niter} iterations took {end-start:.4f} seconds, average {(end-start)/niter:.4f} seconds, {(end-start)/niter / len(frequencies):.4f} per frequency")



# from sigw_fast.libraries.sigwfast import sigwfast_mod as gw
# from sigw_fast.libraries.kernel import kernel_mod as kernel
# from sigw_fast.libraries import sdintegral as sd
# OMEGA_R = 4.2 * 10**(-5)
# CG = 0.39
# norm = CG*OMEGA_R 

# def wfld_full_fortran(komega,nd=250,w=0.001):
#     beta = sd.beta(w)
#     nd,ns1,ns2, darray,d1array,d2array, s1array,s2array = sd.arrays_w(w,komega,nd=nd)
#     print(f" nd: {nd}, ns1: {ns1}, ns2: {ns2}")
#     # print array shapes
#     print(f" darray: {darray.shape}, d1array: {d1array.shape}, d2array: {d2array.shape}, s1array: {s1array.shape}, s2array: {s2array.shape}")
#     nd = int(nd)
#     ns1 = int(ns1)
#     ns2 = int(ns2)
#     nd1 = len(d1array)
#     nd2 = len(d2array)
#     kernel1 = np.empty_like(d1array)
#     kernel2 = np.empty_like(d2array)
#     kernel1 = kernel.kernel1_w(d=d1array,s=s1array,b=beta,ny=nd1) # sd.kernel1_r(d1array, s1array)
#     print("computed kernel1")
#     kernel2 = kernel.kernel2_w(d=d2array,s=s2array,b=beta,ny=nd2) #sd.kernel2_r(d2array, s2array)
#     print("computed kernel2")
#     komega = np.array(komega, dtype=np.float64).ravel()
#     nk = len(komega)
#     # print(f"nk: {nk}, len(komega): {len(komega)}")
#     Integral = np.empty(nk)
#     Integral = gw.compute_w_k_array(nk = nk,komega = komega, 
#                                             kernel1 = kernel1, kernel2 = kernel2, d1array=d1array,
#                                             s1array=s1array, d2array=d2array, s2array=s2array,
#                                             darray=darray, nd = nd, ns1 = ns1, ns2 = ns2)
#     print("computed Integral")
#     OmegaGW = norm*Integral
#     return OmegaGW

# start = time.time()
# for i in range(niter):
#     omega_gw4 = wfld_full_fortran(frequencies)
# end = time.time()
# print(f"Using Fortran kernels with full Fortran integrator, {niter} iterations took {end-start:.4f} seconds, average {(end-start)/niter:.4f} seconds, {(end-start)/niter / len(frequencies):.4f} per frequency")



plt.loglog(frequencies, omega_gw1, label="Full python version",lw=2.5)
# plt.loglog(frequencies, omega_gw2, label="Python and CPP version", linestyle="-.")
plt.loglog(frequencies, omega_gw2, label="Numba kernel, jax integrator", linestyle="--")
plt.loglog(frequencies, omega_gw3, label="Numba kernels with full Fortran integrator", linestyle=":")
plt.legend()
plt.suptitle("Custom w")
plt.show()
