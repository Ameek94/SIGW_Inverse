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


data = np.load('./bpl_data.npz')
frequencies = data['k']

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

omgw_bpl = compute(bpl,frequencies,nd=150)
omgw_peaked = compute(peaked,frequencies,nd = 500)
omgw_osc = compute(osc,frequencies,nd = 500)

pk_min, pk_max = np.array(min(frequencies)/5), np.array(max(frequencies)*5)
pk_arr = np.geomspace(pk_min, pk_max, 250)

np.savez('spectra.npz'
         ,frequencies=frequencies,gw_bpl=omgw_bpl,gw_peaked=omgw_peaked,
         gw_osc=omgw_osc, pk_arr=pk_arr, pk_bpl = bpl(pk_arr),
         pk_peaked = peaked(pk_arr), pk_osc = osc(pk_arr))

fig, ax  = plt.subplots(1,2,figsize=(12,6),layout='tight')

ax[1].loglog(frequencies,omgw_bpl,label='BPL')
ax[1].loglog(frequencies,omgw_peaked,label='Peaked')
ax[1].loglog(frequencies,omgw_osc,label='Oscillatory')
ax[1].set_xlabel(r'f [Hz]')
ax[1].set_ylabel(r'$\Omega_{GW}$')

ax[0].loglog(pk_arr,bpl(pk_arr),label='BPL')
ax[0].loglog(pk_arr,peaked(pk_arr),label='Peaked')
ax[0].loglog(pk_arr,osc(pk_arr),label='Oscillatory')
ax[0].set_xlabel(r'f [Hz]')
ax[0].set_ylabel(r'$P(f)$')

ax[1].legend()
plt.savefig('spectra.pdf',bbox_inches='tight')
plt.show()