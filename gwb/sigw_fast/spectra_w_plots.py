import os
import sys
sys.path.append('../')
import time
from sigw_fast.EOS import compute
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
font = {'size': 16, 'family': 'serif'}
axislabelfontsize = 'large'
matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=True)
matplotlib.rc('legend', fontsize=16)


data = np.load('./bpl_data.npz')
frequencies = data['k']


w = 1/3
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


def bpl(p, pstar=5e-4, n1=2, n2=-1, sigma=2.):
    nir = n1
    pl1 = (p / pstar) ** nir
    nuv = (n2 - n1) / sigma
    pl2 = (1 + (p / pstar) ** sigma) ** nuv
    return 1e-2 * pl1 * pl2

# frequencies = np.geomspace(min(frequencies),max(frequencies),500)
fac = 5.
pk_min, pk_max = np.array(min(frequencies)/fac), np.array(max(frequencies)*fac)
pk_arr = np.geomspace(pk_min, pk_max, 250)
num_nodes = 150

omgw_bpl_RD = compute_w(0.2,log10_f_rh,bpl,num_nodes,pk_arr,frequencies,nd=150)
omgw_bpl_w0p8 = compute_w(0.9,log10_f_rh,bpl,num_nodes,pk_arr,frequencies,nd=150)


# plot
fig, ax = plt.subplots(1,2,figsize=(12,4),constrained_layout=True)

ax[0].loglog(pk_arr,bpl(pk_arr)/max(bpl(pk_arr)),color='k')

w=0.2
ax[1].loglog(frequencies,omgw_bpl_RD / max(omgw_bpl_RD),label=rf'$w={{w}}$',color='C0')
f_at_max = frequencies[np.argmax(bpl(frequencies))]
f_cut = frequencies[frequencies < f_at_max]
gw_at_max = omgw_bpl_RD[np.argmax(bpl(frequencies))] / max(omgw_bpl_RD)
b = (1-3*w)/(1+3*w)
# b = (1-3*0.5)/(1+3*0.5)
print(f'ir = {3-2*abs(b)}')
ax[1].loglog(frequencies, (frequencies/frequencies[0])**(3 -2*abs(b)) * omgw_bpl_RD[0]/ max(omgw_bpl_RD),color='C0',ls='--')
# ax[1].loglog(f_cut,gw_at_max  * (f_cut/f_at_max)**(3 - 2*abs(b)),label=r'$IR$',color='C0',ls='--') # * np.log(f_cut/f_at_max)**2
f_at_max = frequencies[np.argmax(bpl(frequencies))]
f_cut = frequencies[frequencies < f_at_max]
gw_at_max = omgw_bpl_w0p8[np.argmax(bpl(frequencies))] / max(omgw_bpl_w0p8) #omgw_bpl_RD[np.argmax(bpl(frequencies))] / max(omgw_bpl_w0p8)
w = 0.9
b = (1-3*w)/(1+3*w)
print(f'ir = {3-2*abs(b)}')
ax[1].loglog(frequencies,omgw_bpl_w0p8/ max(omgw_bpl_w0p8),label=rf'$w={{w}}$',color='C1')
# ax[1].loglog(f_cut,gw_at_max * (f_cut/f_at_max)**(3 -2*abs(b)),label=r'$IR$',color='C1',ls='--') #* np.log(f_cut/f_at_max)**2 * 
ax[1].loglog(frequencies, (frequencies/frequencies[0])**(3 -2*abs(b)) * omgw_bpl_w0p8[0]/ max(omgw_bpl_w0p8),color='C1',ls='--')
ax[1].legend()
ylabels = [r'$P_{\zeta}$', r'$\Omega_{\rm GW}$']
for x in ax:
    x.set(xscale='log', yscale='log', xlabel=r'$f\,{\rm [Hz]}$',ylabel=ylabels[ax.tolist().index(x)])
k_mpc_f_hz = 2*np.pi * 1.03 * 10**14
for x in ax:
    secax = x.secondary_xaxis('top', functions=(lambda x: x * k_mpc_f_hz, lambda x: x / k_mpc_f_hz))
    secax.set_xlabel(r"$k\,{\rm [Mpc^{-1}]}$",labelpad=10) 
# plt.savefig('bpl_w_dep.pdf',bbox_inches='tight')
plt.show()



omgw_bpl_RD = compute_w(1/3,log10_f_rh,bpl,num_nodes,pk_arr,frequencies,nd=150)
bpl2 = lambda p: bpl(p, pstar=5e-4, n1=3, n2=-2, sigma=2)
omgw_bpl_RD_2 = compute_w(1/3,log10_f_rh,bpl2,num_nodes,pk_arr,frequencies,nd=150)

# plot
fig, ax = plt.subplots(1,2,figsize=(12,4),constrained_layout=True)

ax[0].loglog(pk_arr,bpl(pk_arr)/max(bpl(pk_arr)),color='C0')
ax[0].loglog(pk_arr,bpl2(pk_arr)/max(bpl2(pk_arr)),color='C1')

ax[1].loglog(frequencies,omgw_bpl_RD / max(omgw_bpl_RD),color='C0')
ax[1].loglog(frequencies,omgw_bpl_RD_2/ max(omgw_bpl_RD_2),color='C1')
# ax[1].legend()
ylabels = [r'$P_{\zeta}$', r'$\Omega_{\rm GW}$']
for x in ax:
    x.set(xscale='log', yscale='log', xlabel=r'$f\,{\rm [Hz]}$',ylabel=ylabels[ax.tolist().index(x)])
k_mpc_f_hz = 2*np.pi * 1.03 * 10**14
for x in ax:
    secax = x.secondary_xaxis('top', functions=(lambda x: x * k_mpc_f_hz, lambda x: x / k_mpc_f_hz))
    secax.set_xlabel(r"$k\,{\rm [Mpc^{-1}]}$",labelpad=10) 
plt.savefig('bpl_pz_dep.pdf',bbox_inches='tight')
plt.show()




# # omgw_bpl = compute(bpl,frequencies,nd=150,w=w,f_rh=f_rh,Use_Cpp=False)
# # omgw_peaked = compute(peaked,frequencies,nd = 500,w=w,f_rh=f_rh,Use_Cpp=False)
# # omgw_osc = compute(osc,frequencies,nd = 500,w=w,f_rh=f_rh,Use_Cpp=False)

# np.savez(f'spectra_0p66_interp.npz'
#          ,frequencies=frequencies,gw_bpl=omgw_bpl,gw_peaked=omgw_peaked,
#          gw_osc=omgw_osc, pk_arr=pk_arr, pk_bpl = bpl(pk_arr),
#          pk_peaked = peaked(pk_arr), pk_osc = osc(pk_arr), w=w, log10_f_rh=log10_f_rh)

# fig, ax  = plt.subplots(1,2,figsize=(12,6),layout='tight')

# ax[1].loglog(frequencies,omgw_bpl,label='BPL')
# ax[1].loglog(frequencies,omgw_peaked,label='Peaked')
# ax[1].loglog(frequencies,omgw_osc,label='Oscillatory')
# ax[1].set_xlabel(r'f [Hz]')
# ax[1].set_ylabel(r'$\Omega_{GW}$')

# def interp_pk(func,pk_arr):
#     nodes = np.log10(np.geomspace(min(pk_arr),max(pk_arr),num_nodes))
#     vals = np.log10(func(10**nodes))
#     res = gw.power_spectrum_k_array(nodes,vals, pk_arr)
#     return res
# ax[0].loglog(pk_arr,bpl(pk_arr),label='BPL')
# ax[0].loglog(pk_arr,peaked(pk_arr),label='Peaked')
# ax[0].loglog(pk_arr,osc(pk_arr),label='Oscillatory')
# ax[0].loglog(pk_arr,interp_pk(bpl,pk_arr),label='Interpolated BPL',ls='-.')
# ax[0].loglog(pk_arr,interp_pk(peaked,pk_arr),label='Interpolated Peaked',ls='-.')
# ax[0].loglog(pk_arr,interp_pk(osc,pk_arr),label='Interpolated Oscillatory',ls='-.')
# ax[0].legend()
# ax[0].set_ylabel(r'$P(k)$')
# ax[0].set_xlabel(r'f [Hz]')
# ax[0].set_ylabel(r'$P(f)$')

# ax[1].legend()
# plt.savefig('spectra_0p66.pdf',bbox_inches='tight')
# plt.show()