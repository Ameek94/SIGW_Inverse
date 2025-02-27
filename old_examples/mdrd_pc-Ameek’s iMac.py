import sys
sys.path.append('../')
from numpy import pi, log
import numpy as np
import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior, SortedUniformPrior
try:
    from mpi4py import MPI
except ImportError:
    pass
from interpax import CubicSpline
from gwb.omega_gw_jax import OmegaGWjax
import jax.numpy as jnp
from jax import jit
import matplotlib.pyplot as plt

# set up the power spectrum and omgw
psize = 50
ksize = 40
k_arr = jnp.geomspace(5e-5,1e-2,ksize)
p_arr = jnp.geomspace(1e-5,1e1,psize)


f = k_arr #jnp.geomspace(5e-5, 1e-2, ksize)  # The frequencies to calculate Omega_GW
s = jnp.linspace(0, 1, 10)  # First rescaled internal momentum
t = jnp.logspace(-3,3, 100)  # Second rescaled internal momentum

## Expand t to add a new axis
t_expanded = jnp.expand_dims(t, axis=-1)
## Repeat t along the new axis to match the shape (100, 1000)
t = jnp.repeat(t_expanded, len(f), axis=-1)

pstar=5e-4
n1=3
n2=-2
sigma=2

@jit
def pz(p,kmax,etaR):
    nir = n1
    pl1 = (p/pstar)**nir
    nuv = (n2 - n1)/sigma
    pl2 = (1+(p/pstar)**sigma)**nuv
    # osc = (1 + 16.4*jnp.cos(1.4*jnp.log(p/1.))**2)
    return pl1 * pl2 #*osc

gwb_calculator =  OmegaGWjax(s, t, f=f, kernel="I_MD_to_RD", upsample=False,norm="CT")

kmax = 5e-3
etaR = 1./kmax
transition_params = [kmax,etaR]

pz_amp = pz(p_arr,*transition_params)
gwb_amp = gwb_calculator(pz,f,*transition_params)
kstar = 1e-3
omks_sigma = gwb_amp*( 0.1*(np.log(k_arr/kstar))**2 + 0.05) # 2% error at kstar + more towards edges
gwb_cov = jnp.diag(omks_sigma**2)

print(gwb_amp.shape)

# fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,4))
# ax1.scatter(p_arr,pz_amp,s=2)
# ax1.set_ylabel(r'$P_{\zeta}(k)$')
# ax2.set_ylabel(r'$\Omega_{\mathrm{GW}}(k)$')
# ax2.scatter(f,gwb_amp,s=2)
# ax2.fill_between(k_arr,gwb_amp+1.96*omks_sigma,gwb_amp-1.96*omks_sigma,alpha=0.2,color='C0')
# for ax in [ax1,ax2]:
#     ax.set(yscale='log',xscale='log',xlabel=r'$k$')
# fig.tight_layout();

from interpolation.model import spline_predict
fac = 2.5
kmin, kmax = min(k_arr)/fac, max(k_arr)*fac
n = 4
nodes = jnp.log10(jnp.geomspace(kmin, kmax, n))
ymin, ymax = -10, 1
k1, k2 = -3, 0

def prior(x):
    ndim = len(x)
    x[:ndim-1] = x[:ndim-1]*(ymax-ymin) + ymin
    x[ndim-1] = x[ndim-1]*(k2-k1) + k1
    return x

def loglikelihood(params):
    ys = params[:-1]
    kmax = params[-1]
    kmax = 10**kmax
    etaR = 1./kmax
    pz_interp = lambda k, kmax, etaR: spline_predict(nodes, ys, k)
    gwb =  np.array(gwb_calculator(pz_interp,f,kmax,etaR))
    return float(-0.5*np.dot(gwb - gwb_amp, np.linalg.solve(gwb_cov, gwb - gwb_amp)), [0.]

def dumper(live, dead, logweights, logZ, logZerr):
    print("Last dead point:", dead[-1])
     
nDims = n + 1
nDerived = 1
settings = PolyChordSettings(nDims, nDerived)
settings.file_root = 'mdrd'
settings.nlive = 200
settings.do_clustering = True
settings.read_resume = False

output = pypolychord.run_polychord(loglikelihood, nDims, nDerived, settings, prior, dumper)
     

paramnames = [('p%i' % i, r'\theta_%i' % i) for i in range(nDims)]
paramnames += [('r*', 'r')]
output.make_paramnames_files(paramnames)
     
