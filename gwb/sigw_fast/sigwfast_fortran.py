### SIGWfast - python interface to the SIGWfast fortran version
# This file includes code derived from [SIGWfast](https://github.com/Lukas-T-W/SIGWfast),
# originally written by Lukas T. Witkowski.
# MIT License
# Copyright (c) 2022 Lukas T. Witkowski
# Modifications have been made by Ameek Malhotra.

import sys
sys.path.append('libraries/')
# from sigw_fast.libraries import sdintegral as sd    
from sigw_fast.sigwfast import sigwfast_mod as gw
import numpy as np
import multiprocessing as mp

try:
    from sigw_fast.libraries import sdintegral_numba as sd
except ModuleNotFoundError:
    print("Numba not installed. Please install numba and numba-scipy to use the numba kernel (recommended for non-standard kernels). Using default scipy kernel.")
    from sigw_fast.libraries import sdintegral as sd


# print(compute_fortran.__doc__)

OMEGA_R = 4.2 * 10**(-5)
CG = 0.39
norm = CG*OMEGA_R 

def mp_kernel1_r(args):
    d_i, s_i = args
    return sd.kernel1_r(d_i,s_i)

def mp_kernel2_r(args):
    d_i, s_i = args
    return sd.kernel2_r(d_i,s_i)

def compute_rd(nodes, vals, komega,norm = norm,nd = 250, use_mp = False):
    nd,ns1,ns2, darray,d1array,d2array, s1array,s2array = sd.arrays_r(komega,nd=nd)
    # print(f" nd: {nd}, ns1: {ns1}, ns2: {ns2}")
    nd = int(nd)
    ns1 = int(ns1)
    ns2 = int(ns2)
    args_list1 = [(d_i, s_i) for d_i, s_i in zip(d1array, s1array)]
    args_list2 = [(d_i, s_i) for d_i, s_i in zip(d2array, s2array)]
    if use_mp: 
        with mp.Pool(mp.cpu_count()) as pool:
            kernel1 = pool.map(mp_kernel1_r, args_list1)
            kernel2 = pool.map(mp_kernel2_r, args_list2)
    else:
        kernel1 = sd.kernel1_r(d1array, s1array)
        kernel2 = sd.kernel2_r(d2array, s2array)
    komega = np.array(komega, dtype=np.float64).ravel()
    nk = len(komega)
    # print(f"nk: {nk}, len(komega): {len(komega)}")
    Integral = np.zeros(nk)
    Integral = gw.compute_w_k_array(nodes = nodes, vals = vals, nk = nk,komega = komega, 
                                            kernel1 = kernel1, kernel2 = kernel2, d1array=d1array,
                                            s1array=s1array, d2array=d2array, s2array=s2array,
                                            darray=darray, nd = nd, ns1 = ns1, ns2 = ns2)
    OmegaGW = norm*Integral
    return OmegaGW

def compute_w(komega,w=1/3,cs_equal_one=False,fref=1e-3,f_rh=1.,norm = norm,nd = 250):

    if cs_equal_one:
        OmegaGW = compute_w_cs1(komega,w,fref=fref,f_rh=f_rh,norm = norm,nd = nd)
    else:
        OmegaGW = compute_w_fld(komega,w,fref=fref,f_rh=f_rh,norm = norm,nd = nd)

    return OmegaGW


def mp_kernel1_w(args):
    d_i, s_i, beta = args
    return sd.kernel1_w(d_i, s_i, beta)

def mp_kernel2_w(args):
    d_i, s_i, beta = args
    return sd.kernel2_w(d_i, s_i, beta)

def compute_w_fld(komega,w=1/3,fref=1e-5,f_rh=1.,norm = norm,nd = 250,use_mp = False):

    beta = sd.beta(w)
    nd,ns1,ns2, darray,d1array,d2array, s1array,s2array = sd.arrays_w(w,komega,nd=nd)
    nd = int(nd)
    ns1 = int(ns1)
    ns2 = int(ns2)
    if use_mp: # Use multiprocessing if not using numba kernel
        args_list1 = [(d_i, s_i, beta) for d_i, s_i in zip(d1array, s1array)]
        args_list2 = [(d_i, s_i, beta) for d_i, s_i in zip(d2array, s2array)]
        with mp.Pool(mp.cpu_count()) as pool:
            kernel1 = pool.map(mp_kernel1_w, args_list1)
            kernel2 = pool.map(mp_kernel2_w, args_list2)
    else: # Use numba kernel
        kernel1 = sd.kernel1_w(d1array, s1array, beta)
        kernel2 = sd.kernel2_w(d2array, s2array, beta)
    komega = np.array(komega, dtype=np.float64).ravel()
    nk = len(komega)
    Integral  = gw.compute_w_k_array(nk = nk,komega = komega,
                                            kernel1 = kernel1, kernel2 = kernel2, d1array=d1array,
                                            s1array=s1array, d2array=d2array, s2array=s2array,
                                            darray=darray, nd = nd, ns1 = ns1, ns2 = ns2)
    OmegaGW = norm*Integral
    return OmegaGW #* (f_rh/fref)**(2*beta)

def compute_w_cs1(komega,w=1/3,fref=1e-3,f_rh=1.,norm = norm,nd = 250):
    beta = sd.beta(w)
    nd, ns, darray, ddarray, ssarray = sd.arrays_1(w,komega,nd=nd)
    nd = int(nd)
    kernel = sd.kernel_1(ddarray, ssarray, beta)
    komega = np.array(komega, dtype=np.float64).ravel()
    nk = len(komega)
    Integral  = gw.compute_1_k_array(nk = nk,komega = komega,
                                            kernel = kernel, ddarray=ddarray,
                                            ssarray=ssarray,
                                            darray=darray, nd = nd, ns=ns)
    OmegaGW = norm*Integral
    return OmegaGW #* (f_rh/fref)**(2*beta)
