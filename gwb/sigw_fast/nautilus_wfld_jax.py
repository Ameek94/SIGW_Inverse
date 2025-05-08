from functools import partial
import os
import sys
import time
import numpy as np
from scipy.interpolate import interp1d
import libraries.sdintegral_numba as sd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from nautilus import Sampler
import math
import jax.numpy as jnp
from collections import OrderedDict
import jax
jax.config.update("jax_enable_x64", True)
from jax.scipy.linalg import cho_solve, solve_triangular
import numpyro.distributions as dist
import h5py
from numpyro.util import enable_x64
enable_x64()
from math import sqrt
sqrt2 = sqrt(2.)
sqrt3 = sqrt(3.)


OMEGA_R = 4.2 * 10**(-5)
CG = 0.39
rd_norm = CG * OMEGA_R 
nd = 150
SQRTEPS = math.sqrt(float(np.finfo(np.float64).eps))

# Global cache for storing kernels keyed by rounded w
kernel_cache = OrderedDict()

cache_counter = 0

@partial(jax.jit,static_argnames='include_noise')
def rbf_kernel(x1,x2,lengthscales,noise=1e-8, include_noise=True):
    sq_dist = jnp.sum(jnp.square(x1[:,None,:] - x2),axis=-1) 
    sq_dist = sq_dist / lengthscales
    k = jnp.exp(-0.5*sq_dist) 
    if include_noise:
        k+= noise * jnp.eye(len(x1))
    return k

cache_counter = 0

# open once, reuse for all likelihood calls
h5 = h5py.File("precomputed_kernels.h5", "r")
w_vals = h5["w_vals"][:]      # load in memory once; it's small
# kernel datasets stay on disk until you index them
dset1 = h5["kernel1"]
dset2 = h5["kernel2"]
tol = 1e-2

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

def get_kernels(w, d1array, s1array, d2array, s2array, tolerance=5):
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
    if len(kernel_cache) >= 200:
        kernel_cache.popitem(last=False)
    
    # Store and return the result
    kernel_cache[key] = (jnp.array(kernel1), jnp.array(kernel2))
    return kernel_cache[key]

@jax.jit
def get_mean_from_cho(k11_cho,k12,train_y):
    mu = jnp.matmul(jnp.transpose(k12),cho_solve((k11_cho,True),train_y)) # can also store alphas
    mean = mu.squeeze(-1)
    return mean

@jax.jit
def interpolate(nodes, vals, x, left_node,right_node, k11_cho, lengthscale,y_mean,y_std):
    x = jnp.reshape(x,(-1,1))
    # standardise x as well
    x = (x - left_node) / (right_node - left_node)
    k12 = rbf_kernel(nodes,x,lengthscale,include_noise=False)
    # print(f"x shape: {x.shape} k12 shape: {k12.shape}")
    res = get_mean_from_cho(k11_cho,k12,vals)
    # print(f"res shape: {res.shape}")
    res = res * y_std + y_mean
    # print(f"res shape after rescale: {res.shape}")
    res = jnp.power(10, res)
    res = jnp.where(x[0] < left_node, 0, res)
    res = jnp.where(x[0] > right_node, 0, res)
    # print(f"res shape at end: {res.shape}")
    return res

# @jax.jit
# def Pz(f):
#     res = interpolate(nodes=nodes,vals=vals,x=jnp.log10(f),
#                           y_mean=vals_mean,y_std=vals_std,lengthscale=lengthscale,
#                           left_node=nodes[0],right_node=nodes[-1],k11_cho = k11_cho)
#     return res

@jax.jit
def compute_single_f(f,
                    nodes,vals,
                    y_mean,y_std,lengthscale,
                    left_node,right_node,k11_cho,
                    darray,s1,s2,
                    s1array,d1array,
                    s2array,d2array,
                    K1,K2):
    
    nd, ns1, ns2 = darray.shape[0], s1.shape[1], s2.shape[1]

    Pz = lambda f: 10 ** interpolate(nodes=nodes,vals=vals,x=jnp.log10(f),
                           y_mean=y_mean,y_std=y_std,lengthscale=lengthscale,
                           left_node=left_node,right_node=right_node,k11_cho = k11_cho)
    psq1 = Pz(f/2 * (s1array + d1array)) * Pz(f/2 * (s1array - d1array))  #psquared_jax(d1array, s1array, Pz, f).reshape((nd, ns1))
    psq1 = psq1.reshape((nd, ns1))
    psq2 = Pz(f/2 * (s2array + d2array)) * Pz(f/2 * (s2array - d2array))  #psquared_jax(d2array, s2array, Pz, f).reshape((nd, ns2))
    psq2 = psq2.reshape((nd, ns2))
    Int_ds1 = K1 * psq1
    Int_ds2 = K2 * psq2
    int_s1 = jnp.trapezoid(Int_ds1,x=s1,axis=1)
    int_s2 = jnp.trapezoid(Int_ds2,x=s2,axis=1)
    int_d = int_s1+int_s2
    res = jnp.trapezoid(int_d,x=darray)
    return res


def compute_w(w,log10_f_rh,nodes,vals,lengthscale,frequencies,nd=150,fref=1.,kernels_from_file=True):
    # global kernel_cache
    global cache_counter
    # standardise the input data
    vals_mean = jnp.mean(vals)
    vals_std = jnp.std(vals)
    vals = (vals-vals_mean)/vals_std
    # print(f"vals shape: {vals.shape}")
    left_node, right_node = nodes[0], nodes[-1]
    nodes = (nodes-left_node)/(right_node - left_node)
    n = len(nodes)
    nodes = np.reshape(nodes,(n,1))
    vals = np.reshape(vals,(n,1))
    # print(f"nodes shape = {nodes.shape}, vals shape = {vals.shape}")
    k11 = rbf_kernel(nodes,nodes,lengthscale)
    # print(f"k11 shape = {k11.shape}")
    k11_cho = jnp.linalg.cholesky(k11)

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



    # nd,ns1,ns2, darray,d1array,d2array, s1array,s2array = sd.arrays_w(w,frequencies,nd=nd)
    # b = sd.beta(w)
    # kernel1, kernel2 = get_kernels(w, d1array, s1array, d2array, s2array)

    # convert to jax arrays
    darray = jnp.array(darray)
    d1array = jnp.array(d1array)
    d2array = jnp.array(d2array)
    s1array = jnp.array(s1array)
    s2array = jnp.array(s2array)
    s1 = jnp.array(s1array.reshape((nd, ns1)))
    s2 = jnp.array(s2array.reshape((nd, ns2)))
    K1 = kernel1.reshape((nd, ns1))
    K2 = kernel2.reshape((nd, ns2))

    # print(f"shapes: darray {darray.shape}, d1array {d1array.shape}, d2array {d2array.shape}")
    # print(f"shapes: s1array {s1array.shape}, s2array {s2array.shape}")
    # print(f"shapes: s1 {s1.shape}, s2 {s2.shape}")
    # print(f"shapes: K1 {K1.shape}, K2 {K2.shape}")

    # @jax.jit
    # def compute_single_f(f):
    #     psq1 = Pz(f/2 * (s1array + d1array)) * Pz(f/2 * (s1array - d1array))  #psquared_jax(d1array, s1array, Pz, f).reshape((nd, ns1))
    #     psq1 = psq1.reshape((nd, ns1))
    #     psq2 = Pz(f/2 * (s2array + d2array)) * Pz(f/2 * (s2array - d2array))  #psquared_jax(d2array, s2array, Pz, f).reshape((nd, ns2))
    #     psq2 = psq2.reshape((nd, ns2))
    #     Int_ds1 = K1 * psq1
    #     Int_ds2 = K2 * psq2
    #     int_s1 = jnp.trapezoid(Int_ds1,x=s1,axis=1)
    #     int_s2 = jnp.trapezoid(Int_ds2,x=s2,axis=1)
    #     int_d = int_s1+int_s2
    #     res = jnp.trapezoid(int_d,x=darray)
    #     return res

    Integral = jax.vmap(lambda f: compute_single_f(f,
                                                   nodes,vals,
                                                    vals_mean,vals_std,lengthscale,
                                                    left_node,right_node,k11_cho,
                                                    darray,s1,s2,
                                                    s1array,d1array,
                                                    s2array,d2array,
                                                    K1,K2)
                        )(frequencies)
    f_rh = 10**log10_f_rh
    two_b = 2*b
    norm = rd_norm * (frequencies)**(-two_b) *  (f_rh/fref)**(two_b)   
    OmegaGW = norm * Integral
    return OmegaGW

def jax_prior(cube,
          w_min, w_max,
          free_nodes,
          left_node, right_node,
          y_min, y_max,
          l_min, l_max):
    # unpack & rescale w and l
    w = cube[0] * (w_max - w_min) + w_min
    l = cube[1] * (l_max - l_min) + l_min

    # extract raw xs and ys
    xs_raw = cube[2:2 + free_nodes]         # shape (N,)
    ys     = cube[2 + free_nodes:] \
               * (y_max - y_min) + y_min

    # vectorized computation of t[n] = prod_{k=n..N-1} xs_raw[k]**(1/(k+1))
    # 1) build exponents 1/(i+1)
    exponents = 1.0 / jnp.arange(1, free_nodes + 1)   # [1/1, 1/2, …, 1/N]
    # 2) form factors xs_raw**exponents
    factors = xs_raw ** exponents                     # shape (N,)
    # 3) reverse‐cumprod and then reverse back
    t = jnp.cumprod(factors[::-1])[::-1]              # shape (N,)

    # scale into [left_node, right_node]
    xs = t * (right_node - left_node) + left_node

    # pack everything and return
    return jnp.concatenate([jnp.array([w, l]), xs, ys])

def prior(cube, w_min, w_max,free_nodes, left_node,right_node, y_min, y_max, l_min,l_max):
    params = cube.copy()
    w = params[0]
    w = w * (w_max - w_min) + w_min
    l = params[1]
    l = l * (l_max - l_min) + l_min
    xs = params[2:free_nodes+2]
    N = len(xs)
    t = np.zeros(N)
    t[N-1] = xs[N-1]**(1./N)
    for n in range(N-2, -1, -1):
        t[n] = xs[n]**(1./(n+1)) * t[n+1]
    xs = t*(right_node - left_node) + left_node
    ys = params[free_nodes+2:]
    ys = ys * (y_max - y_min) + y_min
    return np.concatenate([[w],[l],xs, ys])

def likelihood(params, log10_f_rh,free_nodes, left_node,right_node, frequencies, Omegas, omks_sigma):
    w = params[0]
    lengthscale = 10**(params[1])
    nodes = params[2:free_nodes+2]
    nodes = np.pad(nodes, (1,1), 'constant', constant_values=(left_node, right_node))
    vals = params[free_nodes+2:]    
    # print(f"vals shape ll: {vals.shape}")
    omegagw = compute_w(w, log10_f_rh, nodes, vals, lengthscale, frequencies, nd=nd)
    diff = omegagw - Omegas
    ll = -0.5 * np.sum(diff**2 / omks_sigma**2)
    # lengthscale_prior = dist.LogNormal(loc=sqrt2,scale=sqrt3).log_prob(lengthscale)
    return ll, omegagw


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
    y_max = -1.
    y_min = -7.
    l_min = np.log10(1 / num_nodes)
    l_max = np.log10(10)
    w_min = 0.4
    w_max = 0.99
    log10_f_rh = -5.

    ndim = 2 + free_nodes + num_nodes

    prior_transform_jax = lambda cube: jax_prior(cube,
        w_min=w_min, w_max=w_max,
        free_nodes=free_nodes,
        left_node=left_node, right_node=right_node,
        y_min=y_min, y_max=y_max,
        l_min=l_min, l_max=l_max)
    
    # prior_transform = jax.jit(prior_transform_jax)

    prior_transform = partial(prior,w_min=w_min, w_max=w_max,  
                              free_nodes=free_nodes,
                              left_node=left_node, right_node=right_node,
                              y_min=y_min, y_max=y_max,
                              l_min = l_min, l_max = l_max)
    
    loglikelihood = partial(likelihood,log10_f_rh=log10_f_rh, 
                            free_nodes=free_nodes, left_node=left_node, right_node=right_node,
                            frequencies=frequencies, Omegas=Omegas, omks_sigma=omks_sigma)

    sampler = Sampler(prior_transform, loglikelihood, ndim, pass_dict=False,
                      filepath=f'{gwb_model}_w0p66_gp_{num_nodes}.h5',pool=(None,4))

    print(f"Running inference for {num_nodes} nodes")
    sampler.run(verbose=True, f_live=0.01,n_like_max=int(1e6))
    print('log Z: {:.4f}'.format(sampler.log_z))

    samples, logl, logwt, blobs = sampler.posterior(return_blobs=True)
    print(f"Max and min loglike: {np.max(logl)}, {np.min(logl)}")
    np.savez(f'{gwb_model}_w0p66_gp_{num_nodes}.npz', samples=samples, logl=logl, logwt=logwt,logz=sampler.log_z,omegagw=blobs)
    print("Nested sampling complete")
    print(f"Cached kernel was used {cache_counter} times")

if __name__ == "__main__":
    main()