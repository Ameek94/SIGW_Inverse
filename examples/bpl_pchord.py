import os
import sys
sys.path.append('../')
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import warnings
from gwb.omega_gw_sigwfast import SIGW_FAST
from gwb.omega_gw_grid import OmegaGWGrid
from interpolation.spline import Spline
from getdist import plots,MCSamples,loadMCSamples
font = {'size'   : 16, 'family':'serif'}
axislabelfontsize='large'
matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=True) # if using latex in plots
matplotlib.rc('legend', fontsize=16)


# load the gwb data from file and plot it
data = np.load('../bpl_data.npz')
frequencies = data['k']
Omegas = data['gw']
cov = data['cov']

# plt.figure(figsize=(6,4))
# plt.errorbar(frequencies, Omegas, yerr=np.sqrt(np.diag(cov)), fmt="", color='k', label='data',capsize=2,ecolor='k')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel(r'$f$ [Hz]')
# plt.ylabel(r'$\Omega_{\rm GW}(f)$')
# plt.show()

# set up the interpolator
fac = 5
pk_min, pk_max = np.array(min(frequencies)/fac), np.array(max(frequencies)*fac)
p_arr = np.geomspace(pk_min, pk_max, 400)

num_nodes = 5
interpolator = Spline(k_min=pk_min, k_max= pk_max,sample_nodes=False,fixed_nodes=None,num_nodes=num_nodes)
fixed_nodes = interpolator.fixed_nodes


# currently using grid, could replace with SIGWFast
omgw_calculator = SIGW_FAST(frequencies=frequencies,Use_Cpp=True) #OmegaGWGrid(omgw_karr=frequencies,pz_karr=p_arr)
ys = np.random.uniform(-4,-2,num_nodes)
pz_func = interpolator.interpolate(fixed_nodes,ys)
omgw_test = omgw_calculator(pz_func,)

# plt.figure(figsize=(6,4))
# plt.loglog(frequencies, omgw_test, label='interpolated', color='r')
# plt.loglog(frequencies, Omegas, label='data', color='k')
# plt.show()

# set up the log-likelihood and polychord settings

import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior

def prior(hypercube):
    """ Uniform prior from [-1,1]^D. """
    return UniformPrior(-6,-1)(hypercube) # array

def loglikelihood(params):
    # calculate the model
    pz = interpolator.interpolate(fixed_nodes,params)
    model = omgw_calculator(pz,)
    diff = model - Omegas
    loglike = -0.5 * np.dot(diff, np.linalg.solve(cov, diff))
    return loglike

def dumper(live, dead, logweights, logZ, logZerr):
    print("Last dead point:", dead[-1])

nDims = num_nodes
nDerived = 0
settings = PolyChordSettings(nDims, nDerived)
settings.file_root = 'bpl_fixed_sigwfast'
settings.nlive = 10 * num_nodes
settings.do_clustering = True
settings.read_resume = True
settings.precision_criterion = 0.5

output = pypolychord.run_polychord(loglikelihood, nDims, nDerived, settings, prior, dumper)

paramnames = [('p%i' % i, r'y_%i' % i) for i in range(nDims)]
output.make_paramnames_files(paramnames)

try:
    import anesthetic as ac
    samples = ac.read_chains(settings.base_dir + '/' + settings.file_root)
    fig, axes = ac.make_2d_axes(['p0', 'p1', 'p2', 'p3', 'r'])
    samples.plot_2d(axes)
    fig.savefig('posterior.pdf')

except ImportError:
    try:
        import getdist.plots
        posterior = output.posterior
        g = getdist.plots.getSubplotPlotter()
        g.triangle_plot(posterior, filled=True)
        g.export('bpl_pchord_fixed_nodes.pdf')
        plt.show()
    except ImportError:
        print("Install matplotlib and getdist for plotting examples")

samples = np.loadtxt('chains/bpl_fixed_sigwfast_equal_weights.txt')
print(samples.shape)
ys = samples[:,2:]

### Plot the MC realisations with their logprob
p_arr = np.geomspace(pk_min*1.001,pk_max*0.999,100,endpoint=True)

thinning = 16
cmap = matplotlib.colormaps['Reds']
ys =ys[::thinning]
xs = fixed_nodes
lp = -0.5 * samples[:,1] # col has -2*logprob
lp = lp[::thinning] 
lp_min, lp_max = np.min(lp), np.max(lp)
cols = (lp-lp_min)/(lp_max - lp_min) # normalise the logprob to a colour
norm = colors.Normalize(lp_min,lp_max)

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,4),layout='constrained')

def get_pz_omega(y):
    pf = interpolator.interpolate(xs, y)
    pz_amps = pf(p_arr)
    gwb_res = omgw_calculator(pf,)
    return pz_amps, gwb_res

for i,y in enumerate(ys):
    pz_amps, gwb_amps = get_pz_omega(y)
    ax1.loglog(p_arr,pz_amps,alpha=0.25,color=cmap(cols[i]))
    ax1.scatter(10**(xs),10**(ys[i]),s=16,alpha=0.5,color=cmap(cols[i]))
    ax2.loglog(frequencies,gwb_amps,alpha=0.25,color=cmap(cols[i]))

# # True pz
# def pz(p,pstar=5e-4,n1=2,n2=-1,sigma=2):
#     nir = n1
#     pl1 = (p/pstar)**nir
#     nuv = (n2 - n1)/sigma
#     pl2 = (1+(p/pstar)**sigma)**nuv
#     return 1e-2 * pl1 * pl2
# pz_amp = pz(p_arr)
# ax1.loglog(p_arr,pz_amp,color='k',lw=1.5)
ax2.loglog(frequencies,Omegas,color='k',lw=1.5,label='Truth')

ax2.legend()
ax1.set_ylabel(r'$P_{\zeta}(k)$')
ax1.set_xlabel(r'$k$')
ax2.errorbar(frequencies, Omegas, yerr=np.sqrt(np.diag(cov)), fmt="", color='k', label='data',capsize=2,ecolor='k')

ax2.set_ylabel(r'$\Omega_{\mathrm{GW}}(k)$')
ax2.set_xlabel(r'$k$')
fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),ax=[ax1,ax2],label='Logprob')
fig.savefig('bpl_pchord_fixed_sigwfast.pdf')