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

from interpolation.model import interpolation_model, fixed_nodes_model, variable_nodes_model, optim_scipy_bh, optim_optax


class sampler:


    def __init__(self,                  
                pz_kmin: float,
                pz_kmax: float,
                omgw_karr: jnp.ndarray,
                omgw_means: jnp.ndarray,
                omgw_cov: jnp.ndarray, 
                omgw_method: str = 'grid',
                omgw_method_kwargs: dict = {'s': jnp.linspace(0, 1, 15),   't': jnp.logspace(-4, 4, 200) },
                interpolation_method: str = 'CubicSpline',
                interpolation_model: str = 'FixedNodes',
                min_nodes: int = 3,
                max_nodes: int = 12,):
        
        self.omgw_karr = omgw_karr
        self.omgw_means = omgw_means
        self.omgw_cov = omgw_cov
        self.omgw_invcov = jnp.linalg.inv(omgw_cov)
        self.lnk_min, self.lnk_max = jnp.log(pz_kmin), jnp.log(pz_kmax)
        self.bounds = jnp.array([pz_kmin,pz_kmax])
        if omgw_method=='grid':
            self.omgw_func = OmegaGWGrid(omgw_karr=self.omgw_karr,)
        elif omgw_method=='jax':
            self.omgw_func = OmegaGWjax(f=self.omgw_karr,**omgw_method_kwargs)
        else:
            return ValueError("Not a valid method")
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.interpolation_method = interpolation_method
        self.interpolation_model = interpolation_model
        self.run_info = {"interpolation_method": self.interpolation_method, "interpolation_model": self.interpolation_model, ""}
        self.results_dict = {}


    def run(self):
        # for n in [min_nodes,max_nodes]
        #   optimize interpolation_model(n)
        #   results_dict_n = {"best_params": best_params, "chi2": chi2, "aic": aic}
        #   self.results_dict[str(n)] = results_dict_n
        return self.results_dict

    def optimise_model(self,n):
        results_dict_n = {}
        return results_dict_n

        


