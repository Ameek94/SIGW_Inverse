# this module should decide the best fit spline model in terms of number (and position if sampled) of nodes and their amplitudes

from functools import partial
from jax import vmap, pmap, jit, random, value_and_grad,devices, device_count
from jax.lax import cond, map, scan
import jax.numpy as jnp
from typing import Callable, List
from jax import config
import numpy as np
config.update("jax_enable_x64", True)
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS
from numpyro.util import enable_x64
enable_x64()
num_devices = device_count()
import optax
from interpax import CubicSpline
from interpolation.omega_gw_grid import OmegaGWGrid
from interpolation.omega_gw_jax import OmegaGWjax
from scipy.optimize import basinhopping
import math
from interpolation.model import Interpolation_Model, Fixed_Nodes_Model, Variable_Nodes_Model, optim_scipy_bh, optim_optax


class sampler:

    def __init__(self,                  
                pz_kmin: float,
                pz_kmax: float,
                omgw_karr: jnp.ndarray,
                omgw_means: jnp.ndarray,
                omgw_cov: jnp.ndarray, 
                omgw_method: str = 'jax',
                omgw_method_kwargs: dict = {'s': jnp.linspace(0, 1, 15),   't': jnp.logspace(-4, 4, 200) },
                y_low: float = -5,
                y_high: float = 2,
                interpolation_method: str = 'CubicSpline',
                interpolation_model: str = 'FixedNodes',
                min_nodes: int = 2,
                max_nodes: int = 12,):
        
        self.omgw_karr = omgw_karr
        self.omgw_means = omgw_means
        self.omgw_cov = omgw_cov
        self.omgw_invcov = jnp.linalg.inv(omgw_cov)
        self.pz_kmin, self.pz_kmax = pz_kmin, pz_kmax
        self.lnk_min, self.lnk_max = jnp.log(pz_kmin), jnp.log(pz_kmax)
        self.bounds = jnp.array([pz_kmin,pz_kmax])
        self.y_low = y_low
        self.y_high = y_high
        self.omgw_method_kwargs = omgw_method_kwargs
        if omgw_method not in ['grid','jax']:
            raise ValueError(f"{omgw_method} is not a valid method for calculating OmegaGW. Choose from ['grid','jax'].")
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.interpolation_method = interpolation_method
        self.interpolation_model = interpolation_model
        self.run_info = {"interpolation_method": self.interpolation_method, "interpolation_model": self.interpolation_model}
        self.results_dict = {}


    def run(self):
        for n in range(self.min_nodes,self.max_nodes+1):
            self.optimise_model(n)        
        # for n in [min_nodes,max_nodes]
        #   optimize interpolation_model(n)
        #   results_dict_n = {"best_params": best_params, "chi2": chi2, "aic": aic}
        #   self.results_dict[str(n)] = results_dict_n
        return self.results_dict

    def optimise_model(self,n):
        nbins = n
        pz_model = Fixed_Nodes_Model(nbins=nbins,pz_kmin=self.pz_kmin,pz_kmax=self.pz_kmax,
                            omgw_karr=self.omgw_karr,omgw_means=self.omgw_means,
                            omgw_cov=self.omgw_cov,omgw_method='jax',omgw_method_kwargs=self.omgw_method_kwargs
                            ,y_low=self.y_low,y_high=self.y_high)
        samples, extras = pz_model.run_hmc_inference(num_warmup=512,num_samples=512)
        best_idx = jnp.argmin(extras['potential_energy'])
        x0 = samples['y'][best_idx]  #
        best_params, chi2 = optim_scipy_bh(x0 = x0,loss = pz_model.loss,bounds=(pz_model.logy_low,pz_model.logy_high)
                                           ,stepsize=0.2,niter=6*nbins) 
        aic = 2*nbins + chi2
        print(f"Number of nodes: {n}, chi2: {chi2:.4f}, aic: {aic:.4f}, 'best_params': {best_params}")
        results_dict_n = {'best_params': best_params, 'node_locations': pz_model.log_k_nodes, 'chi2': chi2, 'aic': aic}
        # print(results_dict_n)
        self.results_dict[str(n)] = results_dict_n
        del pz_model, x0, best_params, chi2, aic, results_dict_n

    # def plot_results(self,best_n=3):
    #     fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,4))

    #     ax1.loglog(p_arr,pz_bf(p_arr),color='r')
    #     ax1.loglog(p_arr,pz_amp,color='k',lw=1.5)
    #     ax2.plot(k_arr,omgw_amp,color='k',lw=1.5,label='Truth')
    #     ax2.loglog(k_arr,omgw_bf,color='r',label='reconstructed')
    #     ax2.fill_between(k_arr,omgw_amp+1.96*omks_sigma,omgw_amp-1.96*omks_sigma,alpha=0.2,color='C0')
    #     ax2.set(yscale='log',xscale='log')
    #     ax1.set_ylabel(r'$P_{\zeta}(k)$')
    #     ax1.set_xlabel(r'$k$')
    #     ax2.set_ylim(1e-4,1.)

    #     ax2.set_ylabel(r'$\Omega_{\mathrm{GW}}(k)$')
    #     ax2.set_xlabel(r'$k$')
    #     ax2.legend()
    #     for val in nodes:
    #         ax1.axvline(jnp.exp(val),color='k',ls='-.',alpha=0.5)
    #     ax1.scatter(jnp.exp(nodes),jnp.exp(best_params),color='r')
    #     fig.tight_layout()
    

        


