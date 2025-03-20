from functools import partial
import os
import sys
# os.environ["OMP_NUM_THREADS"] = str(sys.argv[2])
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

# load the gwb data from file
data = np.load('./spectra_0p8.npz')
frequencies = data['frequencies']

from sigw_fast.sigwfast_fortran import compute_w_fld

gwb_model = str(sys.argv[1])
Omegas = data[f'gw_{gwb_model}'] 
kstar = 1e-3
omks_sigma = Omegas*( 0.1*(np.log(frequencies/kstar))**2 + 0.1) # 2% error at kstar + more towards edges
cov = np.diag(omks_sigma**2)


#------------------------------------------------------------SIGWFAST------------------------------------------------------------#
OMEGA_R = 4.2 * 10**(-5)
CG = 0.39
rd_norm = CG*OMEGA_R 
nd = 150
from sigw_fast.libraries import sdintegral_numba as sd

from collections import OrderedDict

# Global cache for storing kernels keyed by rounded w
kernel_cache = OrderedDict()

def get_kernels(w, d1array, s1array, d2array, s2array, tolerance=3):
    # Round w to the desired tolerance (number of decimals)
    key = round(w, tolerance)
    
    # If already cached, update the order and return
    if key in kernel_cache:
        kernel_cache.move_to_end(key)
        return kernel_cache[key]
    
    # Otherwise compute the kernels
    b = sd.beta(w)
    kernel1 = sd.kernel1_w(d1array, s1array, b)
    kernel2 = sd.kernel2_w(d2array, s2array, b)
    
    # If cache size is 4, remove the least recently used entry
    if len(kernel_cache) >= 50:
        kernel_cache.popitem(last=False)
    
    # Store and return the result
    kernel_cache[key] = (kernel1, kernel2)
    return kernel_cache[key]


def compute_w(w,log10_f_rh,nodes,vals,frequencies,use_mp=False,nd=150,fref=1.):
    nd,ns1,ns2, darray,d1array,d2array, s1array,s2array = sd.arrays_w(w,frequencies,nd=nd)
    b = sd.beta(w)
    kernel1, kernel2 = get_kernels(w, d1array, s1array, d2array, s2array)
    # kernel1 = sd.kernel1_w(d1array, s1array, b)
    # kernel2 = sd.kernel2_w(d2array, s2array, b)
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

#------------------------------------------------------------Interpolation------------------------------------------------------------#

w_min = 0.6
w_max = 0.9
log10_f_rh_min = -5.5
log10_f_rh_max = -4.5
num_nodes = int(sys.argv[2])
free_nodes =  0
# free_nodes = num_nodes - 2
# print(f"Total number of nodes: {num_nodes}, Free nodes: {free_nodes}")
fac = 5
pk_min, pk_max = np.array(min(frequencies)/fac), np.array(max(frequencies)*fac)
# nodes = np.log10(np.geomspace(pk_min, pk_max, num_nodes))
left_node = np.log10(pk_min)
right_node = np.log10(pk_max)
nodes = np.linspace(left_node,right_node,num_nodes)
y_max = -2.
y_min = -6.


#------------------------------------------------------------Polychord------------------------------------------------------------#
import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior, SortedUniformPrior


def prior(cube):
    params = cube.copy()
    w = UniformPrior(w_min,w_max)(params[0])
    log10_f_rh = UniformPrior(log10_f_rh_min,log10_f_rh_max)(params[1])
    # xs = SortedUniformPrior(left_node,right_node)(params[1:free_nodes+1])
    # ys = UniformPrior(y_min,y_max)(params[free_nodes+1:])
    ys =  UniformPrior(y_min,y_max)(params[2:])
    # return  np.concatenate([[w],xs,ys,[log10_f_rh]]) # array
    return  np.concatenate([[w,log10_f_rh],ys])


def likelihood(params):
    w = params[0]
    log10_f_rh = params[1]
    # nodes = params[2:free_nodes+2]
    # nodes = np.pad(nodes, (1,1), 'constant', constant_values=(left_node, right_node))
    # vals = params[free_nodes+2:]    
    vals = params[2:]
    omegagw = compute_w(w,log10_f_rh,nodes, vals, frequencies,use_mp=False,nd=nd)
    diff = omegagw - Omegas
    return -0.5 * np.dot(diff, np.linalg.solve(cov,diff))

def dumper(live, dead, logweights, logZ, logZerr):
    print("Last dead point:", dead[-1])

nDims = 2 + free_nodes + num_nodes
nDerived = 0
settings = PolyChordSettings(nDims, nDerived)
settings.file_root = f'{gwb_model}_pchord_fixed_'+str(num_nodes)
# settings.nlive = 10 * nDims
settings.do_clustering = True
settings.read_resume = False
settings.precision_criterion = 0.01
settings.feedback = 2
# settings.grade_frac = [0.4,0.6] #[0.2,0.3,0.5] #[0.5,1.] + [1.]*free_nodes + [1.]*num_nodes
# settings.grade_dims = [1,num_nodes+1] #[1,num_nodes,1]
# settings.grade_dims = [1,num_nodes+free_nodes,1]

output = pypolychord.run_polychord(likelihood, nDims, nDerived, settings, prior, dumper)
print("Nested sampling complete")

del kernel_cache

# #------------------------------------------------------------Plotting------------------------------------------------------------#
# paramnames = [('x%i' % i, r'x_%i' % i) for i in range(free_nodes)]
paramnames = [('w', r'w'), ('log10_f_rh', r'\log_{10} f_{\mathrm{rh}}')]
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
        g.export(f'./plots/{gwb_model}_pchord_fixed_{num_nodes}.pdf')
    except ImportError:
        print("Install matplotlib and getdist for plotting examples")

# p_arr = np.geomspace(pk_min*1.001,pk_max*0.999,100,endpoint=True)

# samples = np.loadtxt(f"chains/osc_pchord_free_{num_nodes}_equal_weights.txt")
# xs = samples[:,2:free_nodes+2]
# ys = samples[:,free_nodes+2:]
# lp = -0.5 * samples[:,1] # col has -2*logprob

# thinning = samples.shape[0] // 32
# cmap = matplotlib.colormaps['Reds']
# ys = ys[::thinning]
# xs = xs[::thinning]
# lp = lp[::thinning] 
# lp_min, lp_max = np.min(lp), np.max(lp)
# cols = (lp-lp_min)/(lp_max - lp_min) # normalise the logprob to a colour
# norm = colors.Normalize(lp_min,lp_max)

# fig, (ax1,ax2) = plt.suoscots(1,2,figsize=(12,4),layout='constrained')

# def get_pz_omega(x,y):
#     pz_amps = gw.power_spectrum_k_array(x, y , p_arr)
#     gwb_res = compute_rd(x, y,  frequencies,use_mp=False,nd=150)
#     return pz_amps, gwb_res

# for i,y in enumerate(ys):
#     x = np.pad(xs[i], (1,1), 'constant', constant_values=(left_node, right_node) )
#     pz_amps, gwb_amps = get_pz_omega(x,y)
#     ax1.loglog(p_arr,pz_amps,alpha=0.25,color=cmap(cols[i]))
#     ax1.scatter(10**(x),10**(ys[i]),s=16,alpha=0.5,color=cmap(cols[i]))
#     ax2.loglog(frequencies,gwb_amps,alpha=0.25,color=cmap(cols[i]))

# pz_amp = test_pz(p_arr)
# ax1.loglog(p_arr,pz_amp,color='k',lw=1.5)

# ax2.loglog(frequencies,Omegas,color='k',lw=1.5,label='Truth')

# ax2.legend()
# ax1.set_ylabel(r'$P_{\zeta}(k)$')
# ax1.set_xlabel(r'$k$')
# ax2.errorbar(frequencies, Omegas, yerr=np.sqrt(np.diag(cov)), fmt="", color='k', label='data',capsize=2,ecolor='k')

# ax2.set_ylabel(r'$\Omega_{\mathrm{GW}}(k)$')
# ax2.set_xlabel(r'$k$')
# fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),ax=[ax1,ax2],label='Logprob')
# plt.savefig(f"./plots/osc_pchord_sigwfast_{num_nodes}.pdf")