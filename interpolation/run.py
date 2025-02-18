# this module should decide the best fit spline model in terms of number (and position if sampled) of nodes and their amplitudes

from jax.lax import cond, map, scan
import jax.numpy as jnp
from typing import Callable, List
from jax import config
config.update("jax_enable_x64", True)
from numpyro.util import enable_x64
enable_x64()
from interpolation.model import Interpolation_Model, Fixed_Nodes_Model, Variable_Nodes_Model, optim_scipy_bh, optim_optax


class sampler:
    """
    Sampler class to run the inference using the interpolation model with different number of nodes.

    Arguments
    ---------
    pz_kmin: float,
        the minimum value of k for the interpolation. P_k = 0 for k < pz_kmin
    pz_kmax: float,
        the maximum value of k for the interpolation. P_k = 0 for k > pz_kmax
    gwb_karr: jnp.ndarray,
        the array of k values for the GWB, assumed to be the k-values over which GWB is inferred with its mean and covariance
    gwb_means: jnp.ndarray,
        the mean of the GWB at the k-values in gwb_karr
    gwb_cov: jnp.ndarray,
        the covariance of the GWB at the k-values in gwb_karr
    gwb_method: str,
        the method to use for the GWB interpolation. Either 'grid' or 'jax'
    gwb_method_kwargs: dict,
        the arguments for the GWB method
    y_low: float,
        the minimum possible value of the log10 P_zeta at the nodes
    y_high: float,
        the maximum possible value of the log10 P_zeta at the nodes
    interpolation_method: str,
        the method to use for the interpolation. Either 'CubicSpline' or 'LinearSpline' (not implemented yet)
    interpolation_model: str,
        the model to use for the interpolation. Either 'FixedNodes' or 'VariableNodes' (not implemented here)
    min_nodes: int,
        the minimum number of nodes to use for the interpolation
    max_nodes: int,
        the maximum number of nodes to use for the interpolation
    mc_sample: bool,
        whether to use MCMC sampling first and then use best sample as initial point for the optimizer
    """

    def __init__(self,                  
                pz_kmin: float,
                pz_kmax: float,
                gwb_karr: jnp.ndarray,
                gwb_means: jnp.ndarray,
                gwb_cov: jnp.ndarray, 
                gwb_method: str = 'jax',
                gwb_method_kwargs: dict = {'s': jnp.linspace(0, 1, 15),   't': jnp.logspace(-4, 4, 200) },
                y_low: float = -5,
                y_high: float = 2,
                interpolation_method: str = 'CubicSpline',
                interpolation_model: str = 'FixedNodes',
                min_nodes: int = 2,
                max_nodes: int = 12,
                mc_sample: bool = True,):
        
        self.gwb_karr = gwb_karr
        self.gwb_means = gwb_means
        self.gwb_cov = gwb_cov
        self.gwb_invcov = jnp.linalg.inv(gwb_cov)
        self.pz_kmin, self.pz_kmax = pz_kmin, pz_kmax
        self.lnk_min, self.lnk_max = jnp.log(pz_kmin), jnp.log(pz_kmax)
        self.bounds = jnp.array([pz_kmin,pz_kmax])
        self.y_low = y_low
        self.y_high = y_high
        self.gwb_method_kwargs = gwb_method_kwargs
        if gwb_method not in ['grid','jax']:
            raise ValueError(f"{gwb_method} is not a valid method for calculating OmegaGW. Choose from ['grid','jax'].")
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.interpolation_method = interpolation_method
        self.interpolation_model = interpolation_model
        self.mc_sample = mc_sample
        self.run_info = {"interpolation_method": self.interpolation_method, "interpolation_model": self.interpolation_model}
        self.results_dict = {}


    def run(self):
        for n in range(self.min_nodes,self.max_nodes+1):
            self.optimise_model(n)        
        return self.results_dict

    def optimise_model(self,n_nodes):
        pz_model = Fixed_Nodes_Model(n_nodes=n_nodes,pz_kmin=self.pz_kmin,pz_kmax=self.pz_kmax,
                            omgw_karr=self.omgw_karr,omgw_means=self.omgw_means,
                            omgw_cov=self.omgw_cov,omgw_method='jax',omgw_method_kwargs=self.omgw_method_kwargs
                            ,y_low=self.y_low,y_high=self.y_high)
        if self.mc_sample:
            samples, extras = pz_model.run_hmc_inference(num_warmup=512,num_samples=512)
            best_idx = jnp.argmin(extras['potential_energy'])
            x0 = samples['y'][best_idx]  #
        else:
            x0 = jnp.random.uniform(self.y_low,self.y_high,n_nodes)
        best_params, minus_loglike = optim_scipy_bh(x0 = x0,loss = pz_model.loss,bounds=(pz_model.logy_low,pz_model.logy_high)
                                           ,stepsize=0.2,niter=6*n_nodes) 
        chi2 = 2*minus_loglike
        aic = 2*n_nodes + chi2
        print(f"Number of nodes: {n_nodes}, chi2: {chi2:.4f}, aic: {aic:.4f}, 'best_params': {best_params}")
        results_dict_n = {'best_params': best_params, 'node_locations': pz_model.log_k_nodes, 'chi2': chi2, 'aic': aic}
        self.results_dict[str(n_nodes)] = results_dict_n
        del pz_model, x0, best_params, chi2, aic, results_dict_n

    # def plot_results(self,best_n=3):
    #     fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,4))

    #     ax1.loglog(p_arr,pz_bf(p_arr),color='r')
    #     ax1.loglog(p_arr,pz_amp,color='k',lw=1.5)
    #     ax2.plot(k_arr,gwb_amp,color='k',lw=1.5,label='Truth')
    #     ax2.loglog(k_arr,gwb_bf,color='r',label='reconstructed')
    #     ax2.fill_between(k_arr,gwb_amp+1.96*omks_sigma,gwb_amp-1.96*omks_sigma,alpha=0.2,color='C0')
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
    

        


