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


#------------------------------------------------------------Data------------------------------------------------------------#
def test_pz(p, pstar=5e-4, n1=2, n2=-1, sigma=2):
    nir = n1
    pl1 = (p / pstar) ** nir
    nuv = (n2 - n1) / sigma
    pl2 = (1 + (p / pstar) ** sigma) ** nuv
    return 1e-2 * pl1 * pl2


# load the gwb data from file
data = np.load('./spectra.npz')
frequencies = data['frequencies']
gwb_model = str(sys.argv[1])
Omegas = data[f'gw_{gwb_model}'] 
kstar = 1e-3
omks_sigma = Omegas*( 0.1*(np.log(frequencies/kstar))**2 + 0.05) # 2% error at kstar + more towards edges
cov = np.diag(omks_sigma**2)

#------------------------------------------------------------SIGWFAST------------------------------------------------------------#
OMEGA_R = 4.2 * 10**(-5)
CG = 0.39
rd_norm = CG*OMEGA_R 
nd = 150
from sigw_fast.libraries import sdintegral as sd
nd,ns1,ns2, darray,d1array,d2array, s1array,s2array = sd.arrays_r(frequencies,nd=nd)
kernel1 = sd.kernel1_r(d1array, s1array)
kernel2 = sd.kernel2_r(d2array, s2array)
nk = len(frequencies)
def compute_rd(nodes,vals,frequencies,use_mp=False,nd=150):
    Integral = np.empty_like(frequencies)
    Integral = gw.compute_w_k_array(nodes = nodes, vals = vals, nk = nk,komega = frequencies, 
                                            kernel1 = kernel1, kernel2 = kernel2, d1array=d1array,
                                            s1array=s1array, d2array=d2array, s2array=s2array,
                                            darray=darray, nd = nd, ns1 = ns1, ns2 = ns2)
    OmegaGW = rd_norm*Integral
    return OmegaGW

#------------------------------------------------------------Interpolation------------------------------------------------------------#


num_nodes = int(sys.argv[2])
free_nodes = num_nodes - 2
fac = 10
pk_min, pk_max = np.array(min(frequencies)/fac), np.array(max(frequencies)*fac)
# nodes = np.log10(np.geomspace(pk_min, pk_max, num_nodes))
left_node = np.log10(pk_min)
right_node = np.log10(pk_max)
y_max = -2
y_min = -6




#------------------------------------------------------------Nautilus------------------------------------------------------------#
def prior(cube):
    # Order and transform nodes to be in the correct range, from Polychord SortedUniformPrior
    params = cube.copy()
    x = params[:free_nodes]
    N = len(x)
    t = np.zeros(N)
    t[N-1] = x[N-1]**(1./N)
    for n in range(N-2, -1, -1):
        t[n] = x[n]**(1./(n+1)) * t[n+1]
    xs = t*(right_node - left_node) + left_node
    ys = params[free_nodes:]
    ys = ys*(y_max - y_min) + y_min
    return np.concatenate([xs,ys]) # array

def likelihood(params):
    nodes = params[:free_nodes]
    nodes = np.pad(nodes, (1,1), 'constant', constant_values=(left_node, right_node))
    vals = params[free_nodes:]
    omegagw = compute_rd(nodes, vals, frequencies,use_mp=False,nd=nd)
    diff = omegagw - Omegas
    return -0.5 * np.dot(diff, np.linalg.solve(cov,diff))

from nautilus import Sampler

ndim = free_nodes + num_nodes
sampler = Sampler(prior, likelihood, ndim, pass_dict=False,)
sampler.run(verbose=True,f_live=0.01)
print('log Z: {:.2f}'.format(sampler.log_z))

samples, logl, logwt = sampler.posterior()

np.savez(f'{gwb_model}_{num_nodes}.npz',samples=samples,logl=logl,logwt=logwt)

print("Nested sampling complete")

import math
SQRTEPS = math.sqrt(float(np.finfo(np.float64).eps))
def resample_equal(samples,logl, logwt, rstate):
    # Extract the weights and compute the cumulative sum.
    wt = np.exp(logwt)
    weights = wt / wt.sum()
    cumulative_sum = np.cumsum(weights)

    # if abs(cumulative_sum[-1] - 1.) > SQRTEPS:
    #     # same tol as in numpy's random.choice.
    #     # Guarantee that the weights will sum to 1.
    #     warnings.warn("Weights do not sum to 1 and have been renormalized.")
    cumulative_sum /= cumulative_sum[-1]
    # this ensures that the last element is strictly == 1

    # Make N subdivisions and choose positions with a consistent random offset.
    nsamples = len(weights)
    positions = (rstate.random() + np.arange(nsamples)) / nsamples

    # Resample the data.
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

rstate = np.random.default_rng(100000)
samples, lp = resample_equal(samples,logl,logwt, rstate=rstate)

print("Obtained equally weighted samples")
print(f"Max and min logprob: {np.max(lp)}, {np.min(lp)}")

#------------------------------------------------------------Plotting------------------------------------------------------------#
p_arr = np.geomspace(pk_min*1.001,pk_max*0.999,100,endpoint=True)

ys = samples[:,free_nodes:]
xs = samples[:,:free_nodes]
print(len(samples))
thinning = len(samples) // 32
cmap = matplotlib.colormaps['Reds']
ys = ys[::thinning]
xs = xs[::thinning]
lp = lp[::thinning] 
lp_min, lp_max = np.min(lp), np.max(lp)
cols = (lp-lp_min)/(lp_max - lp_min) # normalise the logprob to a colour
norm = colors.Normalize(lp_min,lp_max)

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,4),layout='constrained')

def get_pz_omega(x,y):
    pz_amps = gw.power_spectrum_k_array(x, y , p_arr)
    gwb_res = compute_rd(x, y,  frequencies,use_mp=False,nd=150)
    return pz_amps, gwb_res

for i,y in enumerate(ys):
    x = np.pad(xs[i], (1,1), 'constant', constant_values=(left_node, right_node) )
    pz_amps, gwb_amps = get_pz_omega(x,y)
    ax1.loglog(p_arr,pz_amps,alpha=0.25,color=cmap(cols[i]))
    ax1.scatter(10**(x),10**(ys[i]),s=16,alpha=0.5,color=cmap(cols[i]))
    ax2.loglog(frequencies,gwb_amps,alpha=0.25,color=cmap(cols[i]))

pz_amp = test_pz(p_arr)
ax1.loglog(p_arr,pz_amp,color='k',lw=1.5)

ax2.loglog(frequencies,Omegas,color='k',lw=1.5,label='Truth')

ax2.legend()
ax1.set_ylabel(r'$P_{\zeta}(k)$')
ax1.set_xlabel(r'$k$')
ax2.errorbar(frequencies, Omegas, yerr=np.sqrt(np.diag(cov)), fmt="", color='k', label='data',capsize=2,ecolor='k')

ax2.set_ylabel(r'$\Omega_{\mathrm{GW}}(k)$')
ax2.set_xlabel(r'$k$')
fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),ax=[ax1,ax2],label='Logprob')
plt.savefig(f"{gwb_model}_nautilus_sigwfast_{num_nodes}.pdf")

# BPL
# 4 -  log Z: -22.13      
# 5 -  log Z: -21.85
