from jaxbo.bo import BOBE
import numpy as np
from jaxbo.gp import DSLP_GP, SAAS_GP
from jaxbo.loglike import external_likelihood
from jaxbo.bo_utils import input_standardize, input_unstandardize
from getdist import plots, MCSamples, loadMCSamples
import time
from sigwfast import sigwfast_mod as gw
import matplotlib
import matplotlib.pyplot as plt
import math
import sys

OMEGA_R = 4.2 * 10**(-5)
CG = 0.39
rd_norm = CG * OMEGA_R 
nd = 150
SQRTEPS = math.sqrt(float(np.finfo(np.float64).eps))

from libraries import sdintegral_numba as sd

from collections import OrderedDict

# Global cache for storing kernels keyed by rounded w
kernel_cache = OrderedDict()

cache_counter = 0
def get_kernels(w, d1array, s1array, d2array, s2array, tolerance=4):
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
    if len(kernel_cache) >= 500:
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
    # print(f"nodes shape: {nodes.shape}, vals shape: {vals.shape}, frequencies shape: {frequencies.shape}")
    Integral = gw.compute_w_k_array(nodes = nodes, vals = vals, nk = nk,komega = frequencies, 
                                            kernel1 = kernel1, kernel2 = kernel2, d1array=d1array,
                                            s1array=s1array, d2array=d2array, s2array=s2array,
                                            darray=darray, nd = nd, ns1 = ns1, ns2 = ns2)
    f_rh = 10**log10_f_rh
    two_b = 2*b
    norm = rd_norm * (frequencies)**(-2*b) *  (f_rh/fref)**(two_b)   
    OmegaGW = norm * Integral
    return OmegaGW

def prior_transform(cube, w_min, w_max,free_nodes, left_node,right_node, y_min, y_max):
    """Transform the parameters from the unit cube to the original domain."""
    params = cube.copy()
    w = params[0]
    w = w * (w_max - w_min) + w_min
    xs = params[1:free_nodes+1]
    N = len(xs)
    t = np.zeros(N)
    t[N-1] = xs[N-1]**(1./N)
    for n in range(N-2, -1, -1):
        t[n] = xs[n]**(1./(n+1)) * t[n+1]
    xs = t*(right_node - left_node) + left_node
    ys = params[free_nodes+1:]
    ys = ys * (y_max - y_min) + y_min
    return np.concatenate([[w],xs, ys])

def inverse_prior_transform(params, w_min, w_max,free_nodes, left_node,right_node, y_min, y_max):
    """Inverse of the prior transform function. Transforms the parameters from the original domain to the unit cube."""
    params = params.copy()
    w = params[0]
    w = (w - w_min) / (w_max - w_min)
    xs = params[1:free_nodes+1]
    t = (xs-left_node) / (right_node - left_node)
    N = len(t)
    xs = np.zeros(N)
    xs[N-1] = t[N-1]**N
    for n in range(N-2,-1,-1):
        xs[n] = (t[n]/t[n+1])**(n+1)
    ys = params[free_nodes+1:]
    ys = (ys - y_min) / (y_max - y_min)
    return np.concatenate([[w],xs, ys])


def likelihood(params, log10_f_rh,free_nodes, left_node,right_node, frequencies, Omegas, cov):
    w = params[0]
    # log10_f_rh = params[1]
    nodes = params[1:free_nodes+1]
    nodes = np.pad(nodes, (1,1), 'constant', constant_values=(left_node, right_node))
    vals = params[free_nodes+1:]    
    omegagw = compute_w(w, log10_f_rh, nodes, vals, frequencies, use_mp=False, nd=nd)
    diff = omegagw - Omegas
    return -0.5 * np.dot(diff, np.linalg.solve(cov, diff))

class custom_likelihood(external_likelihood):

    def __init__(self, free_nodes, left_node, right_node, y_min, y_max, w_min, w_max,
                 loglikelihood, ndim, param_list = None, param_labels = None, param_bounds = None, noise_std = 0, name = None, vectorized = False, minus_inf = -100000):
        super().__init__(loglikelihood, ndim, param_list, param_labels, param_bounds, noise_std, name, vectorized, minus_inf)
        self.free_nodes = free_nodes
        self.left_node = left_node
        self.right_node = right_node
        self.y_min = y_min
        self.y_max = y_max
        self.w_min = w_min
        self.w_max = w_max

    def true_pz(self,p, pstar=5e-4, n1=2, n2=-1, sigma=2):
        nir = n1
        pl1 = (p / pstar) ** nir
        nuv = (n2 - n1) / sigma
        pl2 = (1 + (p / pstar) ** sigma) ** nuv
        return 1e-2 * pl1 * pl2
    
    def get_initial_points(self, n_init_sobol=8, n_cobaya_init=0):
        bf_nodes = np.linspace(left_node,right_node,self.free_nodes+2)
        bf_vals = np.log10(self.true_pz(p = 10**bf_nodes))
        bf_nodes = bf_nodes[1:self.free_nodes+1]
        bf_w = 0.5
        # now convert to [0,1]
        bf_point = np.concatenate([[bf_w], bf_nodes, bf_vals])
        bf_point_u = inverse_prior_transform(bf_point,free_nodes=self.free_nodes,w_min=self.w_min,w_max=self.w_max,
                                           left_node=self.left_node,right_node=self.right_node,
                                           y_min=self.y_min,y_max=self.y_max)
        test_bf = prior_transform(bf_point_u,free_nodes=self.free_nodes,w_min=self.w_min,w_max=self.w_max,
                                           left_node=self.left_node,right_node=self.right_node,
                                           y_min=self.y_min,y_max=self.y_max)
        print(np.allclose(bf_point,test_bf))
        print(f"shapes {np.shape(bf_point)}, {np.shape(bf_point_u)}")
        bf_ll = self.__call__(bf_point_u)
        bf_point_u = np.reshape(bf_point_u,(1,len(bf_point)))
        bf_ll = bf_ll.reshape(-1,1)
        points, ll = super().get_initial_points(n_init_sobol, n_cobaya_init)
        points = np.concatenate([points,bf_point_u])
        ll = np.concatenate([ll,bf_ll])
        return points, ll 

def main():
    global left_node, right_node, y_max, y_max, w_min, w_max
    data = np.load('./spectra_0p5_interp.npz')
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
    left_node = np.log10(pk_min)
    right_node = np.log10(pk_max)
    y_max =-2.
    y_min = -6.

    w_min = 0.25
    w_max = 0.75
    log10_f_rh = -5.

    ndim = 1 + free_nodes + num_nodes

    param_list = ['w']
    param_labels = ['w']
    for i in range(free_nodes):
        param_list.append(f'node_{i}')
        param_labels.append(f'x_{i}')
    for i in range(num_nodes):
        param_list.append(f'val_{i}')
        param_labels.append(f'y_{i}')
    param_bounds = np.array([[0.,1.]]*ndim).T

    def wrapped_likelihood(params):
        if params.shape[0] != ndim:
            params = params.reshape((ndim,))
        params = prior_transform(params,
                                 w_min=w_min, w_max=w_max,
                                 free_nodes=free_nodes, 
                                 left_node=left_node, right_node=right_node,
                                 y_min=y_min, y_max=y_max)
        val =  likelihood(params, 
                          log10_f_rh=log10_f_rh, 
                          free_nodes = free_nodes, 
                          left_node = left_node, right_node = right_node, 
                          frequencies = frequencies, Omegas = Omegas, cov = cov)
        point = {name: f"{float(val):.4f}" for name, val in zip(param_list, params)}
        print(f"Phyical point: {point} with val = {val:.4f}")
        return val


    minus_inf = -1e5

    sigw_likelihood = custom_likelihood(free_nodes=free_nodes,left_node=left_node,right_node=right_node,
                                        y_max=y_max,y_min=y_min,w_min=w_min,w_max=w_max,
                loglikelihood = wrapped_likelihood
                 ,ndim = ndim
                 ,param_list = param_list
                 ,param_labels = param_labels
                 ,param_bounds = param_bounds
                 ,name = f'{gwb_model}_{num_nodes}_bo_fixed_w',
                 minus_inf = minus_inf)

    start = time.time()
    sampler = BOBE(n_sobol_init = 2, 
        miniters=200, maxiters=1200,max_gp_size=1000,
        loglikelihood=sigw_likelihood,noise = 1e-12,
        fit_step = 20, update_mc_step = 5, ns_step = 50,
        num_hmc_warmup = 1024,num_hmc_samples = 1024, mc_points_size = 100,
        logz_threshold=5,mc_points_method='NUTS',
        lengthscale_priors='DSLP', use_svm=False,minus_inf=minus_inf,svm_threshold=150,svm_gp_threshold=5000,)
    # gp, ns_samples, logz_dict = sampler.run()

    # w = 0.5
    # names = ['w']
    # labels = ['w']
    # bounds = [[w_min,w_max]]
    # ranges = dict(zip(names,bounds))
    # print(ranges)
    # ws = ns_samples['x'][:,0]
    # ws = ws * (w_max - w_min) + w_min # convert from cube to w
    # weights = ns_samples['weights']
    # # gd_samples = MCSamples(samples=samples[:,0], names=names, labels=labels,ranges=ranges,weights=normalized_weights,loglikes=logl)
    # gd_samples = MCSamples(samples=ws, names=names, labels=labels,ranges=ranges,weights=weights)
    # g = plots.get_subplot_plotter(subplot_size=3.5)
    # blue = '#006FED'
    # g.settings.title_limit_fontsize = 14
    # g.settings.axes_fontsize=16
    # g.settings.axes_labelsize=18
    # g.plot_1d(gd_samples, 'w', marker=w, marker_color=blue, colors=[blue],title_limit=2)
    # g.export(f'{gwb_model}_bo_0p5_{num_nodes}_1D_w.pdf')
    # print(gd_samples.getMargeStats())

if __name__ == "__main__":
    main()