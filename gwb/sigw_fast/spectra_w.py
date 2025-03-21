import os
import sys
sys.path.append('../')
import time
from sigw_fast.EOS import compute
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors


data = np.load('./bpl_data.npz')
frequencies = data['k']

w = 0.8
log10_f_rh = -5.

from sigw_fast.libraries import sdintegral_numba as sd
from sigw_fast.sigwfast import sigwfast_mod as gw

OMEGA_R = 4.2 * 10**(-5)
CG = 0.39
rd_norm = CG * OMEGA_R 
def compute_w(w,log10_f_rh,func,num_nodes,pk_arr,frequencies,use_mp=False,nd=150,fref=1.):
    nodes = np.log10(np.geomspace(min(pk_arr),max(pk_arr),num_nodes))
    vals = np.log10(func(10**nodes))
    nd,ns1,ns2, darray,d1array,d2array, s1array,s2array = sd.arrays_w(w,frequencies,nd=nd)
    b = sd.beta(w)
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


def bpl(p, pstar=5e-4, n1=2, n2=-1, sigma=2):
    nir = n1
    pl1 = (p / pstar) ** nir
    nuv = (n2 - n1) / sigma
    pl2 = (1 + (p / pstar) ** sigma) ** nuv
    return 1e-2 * pl1 * pl2

def osc(p,pstar=5e-4,n1=3,n2=-2,sigma=2):
    nir = n1
    pl1 = (p/pstar)**nir
    nuv = (n2 - n1)/sigma
    pl2 = (1+(p/pstar)**sigma)**nuv
    osc = (1 + 16.4*np.cos(1.4*np.log(p/1.))**2)
    return 2e-3*pl1 * pl2 *osc

def peaked(p,pstar=9e-4,sigma=0.1,amp=1e-1,floor=1e-3):
    res = amp *(floor+ np.exp(-0.5*((np.log(p/pstar)/sigma)**2)))
    return 3e-1*res

# plot spectra and corresponding OmegaGW

# frequencies = np.geomspace(min(frequencies),max(frequencies),500)

pk_min, pk_max = np.array(min(frequencies)/5), np.array(max(frequencies)*5)
pk_arr = np.geomspace(pk_min, pk_max, 250)
num_nodes = 150

omgw_bpl = compute_w(w,log10_f_rh,bpl,num_nodes,pk_arr,frequencies,nd=150)
omgw_peaked = compute_w(w,log10_f_rh,peaked,num_nodes,pk_arr,frequencies,nd=200)
omgw_osc = compute_w(w,log10_f_rh,osc,num_nodes,pk_arr,frequencies,nd=200)

# omgw_bpl = compute(bpl,frequencies,nd=150,w=w,f_rh=f_rh,Use_Cpp=False)
# omgw_peaked = compute(peaked,frequencies,nd = 500,w=w,f_rh=f_rh,Use_Cpp=False)
# omgw_osc = compute(osc,frequencies,nd = 500,w=w,f_rh=f_rh,Use_Cpp=False)

np.savez(f'spectra_0p8_interp.npz'
         ,frequencies=frequencies,gw_bpl=omgw_bpl,gw_peaked=omgw_peaked,
         gw_osc=omgw_osc, pk_arr=pk_arr, pk_bpl = bpl(pk_arr),
         pk_peaked = peaked(pk_arr), pk_osc = osc(pk_arr), w=w, log10_f_rh=log10_f_rh)

fig, ax  = plt.subplots(1,2,figsize=(12,6),layout='tight')

ax[1].loglog(frequencies,omgw_bpl,label='BPL')
ax[1].loglog(frequencies,omgw_peaked,label='Peaked')
ax[1].loglog(frequencies,omgw_osc,label='Oscillatory')
ax[1].set_xlabel(r'f [Hz]')
ax[1].set_ylabel(r'$\Omega_{GW}$')

def interp_pk(func,pk_arr):
    nodes = np.log10(np.geomspace(min(pk_arr),max(pk_arr),num_nodes))
    vals = np.log10(func(10**nodes))
    res = gw.power_spectrum_k_array(nodes,vals, pk_arr)
    return res
ax[0].loglog(pk_arr,bpl(pk_arr),label='BPL')
ax[0].loglog(pk_arr,peaked(pk_arr),label='Peaked')
ax[0].loglog(pk_arr,osc(pk_arr),label='Oscillatory')
ax[0].loglog(pk_arr,interp_pk(bpl,pk_arr),label='Interpolated BPL',ls='-.')
ax[0].loglog(pk_arr,interp_pk(peaked,pk_arr),label='Interpolated Peaked',ls='-.')
ax[0].loglog(pk_arr,interp_pk(osc,pk_arr),label='Interpolated Oscillatory',ls='-.')
ax[0].legend()
ax[0].set_ylabel(r'$P(k)$')
ax[0].set_xlabel(r'f [Hz]')
ax[0].set_ylabel(r'$P(f)$')

ax[1].legend()
plt.savefig('spectra_0p8.pdf',bbox_inches='tight')
plt.show()