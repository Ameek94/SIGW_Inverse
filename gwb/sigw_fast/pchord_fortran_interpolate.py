from functools import partial
import os
import sys
os.environ["OMP_NUM_THREADS"] = str(sys.argv[2])
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

# load the gwb data from file
data = np.load('./bpl_data.npz')
frequencies = data['k']


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


num_nodes = int(sys.argv[1])
free_nodes = num_nodes - 2
print(f"Total number of nodes: {num_nodes}, Free nodes: {free_nodes}")
fac = 5
pk_min, pk_max = np.array(min(frequencies)/fac), np.array(max(frequencies)*fac)
# nodes = np.log10(np.geomspace(pk_min, pk_max, num_nodes))
left_node = np.log10(pk_min)
right_node = np.log10(pk_max)
y_max = -2
y_min = -7


#------------------------------------------------------------Data------------------------------------------------------------#
def test_pz(p,pstar=5e-4,n1=3,n2=-2,sigma=2):
    nir = n1
    pl1 = (p/pstar)**nir
    nuv = (n2 - n1)/sigma
    pl2 = (1+(p/pstar)**sigma)**nuv
    osc = (1 + 16.4*np.cos(1.4*np.log(p/1.))**2)
    return 2e-3*pl1 * pl2 *osc

nodes = np.log10(np.geomspace(pk_min, pk_max, 200))
vals =  np.log10(test_pz(10**nodes))
Omegas = compute_rd(nodes, vals, frequencies,use_mp=False,nd=nd)
kstar = 1e-3
omks_sigma = Omegas*( 0.1*(np.log(frequencies/kstar))**2 + 0.1) 
cov = np.diag(omks_sigma**2)

#------------------------------------------------------------Polychord------------------------------------------------------------#
import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior, SortedUniformPrior


def prior(params):
    """ Uniform prior from [-1,1]^D. """
    xs = SortedUniformPrior(left_node,right_node)(params[:free_nodes])
    ys = UniformPrior(y_min,y_max)(params[free_nodes:])
    return  np.concatenate([xs,ys]) # array

def likelihood(params):
    nodes = params[:free_nodes]
    nodes = np.pad(nodes, (1,1), 'constant', constant_values=(left_node, right_node))
    vals = params[free_nodes:]
    omegagw = compute_rd(nodes, vals, frequencies,use_mp=False,nd=nd)
    diff = omegagw - Omegas
    return -0.5 * np.dot(diff, np.linalg.solve(cov,diff))

def dumper(live, dead, logweights, logZ, logZerr):
    print("Last dead point:", dead[-1])

nDims = free_nodes + num_nodes
nDerived = 0
settings = PolyChordSettings(nDims, nDerived)
settings.file_root = 'osc_pchord_free_'+str(num_nodes)
settings.nlive = 10 * nDims
settings.do_clustering = True
settings.read_resume = True
settings.precision_criterion = 0.5

output = pypolychord.run_polychord(likelihood, nDims, nDerived, settings, prior, dumper)
print("Nested sampling complete")

#------------------------------------------------------------Plotting------------------------------------------------------------#
paramnames = [('x%i' % i, r'x_%i' % i) for i in range(free_nodes)]
paramnames += [('y%i' % i, r'y_%i' % i) for i in range(num_nodes)]
output.make_paramnames_files(paramnames)
try:
    import anesthetic as ac
    samples = ac.read_chains(settings.base_dir + '/' + settings.file_root)
    fig, axes = ac.make_2d_axes(['p0', 'p1', 'p2', 'p3', 'r'])
    samples.plot_2d(axes)
    fig.savefig('osc_free_nodes.pdf')

except ImportError:
    try:
        import getdist.plots
        posterior = output.posterior
        g = getdist.plots.getSuoscotPlotter()
        g.triangle_plot(posterior, filled=True)
        g.export(f'./plots/osc_pchord_free_{num_nodes}.pdf')
    except ImportError:
        print("Install matplotlib and getdist for plotting examples")

p_arr = np.geomspace(pk_min*1.001,pk_max*0.999,100,endpoint=True)

samples = np.loadtxt(f"chains/osc_pchord_free_{num_nodes}_equal_weights.txt")
xs = samples[:,2:free_nodes+2]
ys = samples[:,free_nodes+2:]
lp = -0.5 * samples[:,1] # col has -2*logprob

thinning = samples.shape[0] // 32
cmap = matplotlib.colormaps['Reds']
ys = ys[::thinning]
xs = xs[::thinning]
lp = lp[::thinning] 
lp_min, lp_max = np.min(lp), np.max(lp)
cols = (lp-lp_min)/(lp_max - lp_min) # normalise the logprob to a colour
norm = colors.Normalize(lp_min,lp_max)

fig, (ax1,ax2) = plt.suoscots(1,2,figsize=(12,4),layout='constrained')

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
plt.savefig(f"./plots/osc_pchord_sigwfast_{num_nodes}.pdf")