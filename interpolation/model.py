from jax import pmap, jit, random, value_and_grad, devices, device_count
from jax.lax import map, scan
from jax.scipy.linalg import cholesky, cho_solve
import jax.numpy as jnp
from typing import Callable, List, Tuple
from jax import config
import numpy as np
config.update("jax_enable_x64", True)
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.util import enable_x64
enable_x64()
num_devices = device_count()
import optax
from interpax import CubicSpline
from interpolation.omega_gw_grid import OmegaGWGrid
from interpolation.omega_gw_jax import OmegaGWjax
from scipy.optimize import basinhopping
from math import pi

def pow10(x):
    return jnp.power(10,x)

def log10(x):
    return jnp.log10(x)

@jit
def unit_transform(x,bounds):
    """
    Transform array from original domain given by bounds to [0,1]. 
    """
    ux = (x - bounds[0])/(bounds[1]-bounds[0])
    return ux
@jit
def unit_untransform(x,bounds):
    """
    Transform array to original domain given by bounds from [0,1]
    """
    xu = x*(bounds[1]-bounds[0]) + bounds[0]
    return xu

@jit
def spline_predict_log(x_train,y_train,x_pred):
    """
    Cubic spline to interpolate log10 P_zeta as a function of log10_k
    """
    spl = CubicSpline(x_train,y_train,check=False)
    y_pred = spl(x_pred)
    return y_pred

@jit
def spline_predict(x_train,y_train,x_pred):
    """
    Obtain spline prediction after exponentiating log10 P_zeta spline. 
    """
    x_pred = log10(x_pred)
    return pow10(spline_predict_log(x_train,y_train,x_pred))

def optim_scipy_bh(loss,x0,bounds=None,minimizer_kwargs={'method': 'L-BFGS-B'  },stepsize=1/4,niter=15):
    """

    Minimise the loss function using the basinhopping algorithm from scipy. 
    This is a combination of a local optimiser (L-BFGS-B by default) which can use the gradient and a global search algorithm.

    Arguments
    ----------
    loss: callable,
        the loss function to be minimised
    x0: array,
        the initial guess for the parameters
    bounds: list of tuples,
        the bounds for the parameters
    minimizer_kwargs: dict,
        the arguments for the local minimiser
    stepsize: float,
        the stepsize for the global search algorithm
    niter: int,
        the number of iterations for the global search algorithm

    Returns
    ----------       
    results.x: array,
        the best fit parameters
    results.fun: float,
        the value of the loss function at the best fit parameters
    """

    @jit
    def func_scipy(x):
        val, grad = value_and_grad(loss)(x)
        return val, grad

    ndim = len(x0)    
    minimizer_kwargs['jac'] = True 
    bounds = ndim*[(0,1)] if bounds is None else bounds
    if len(bounds)==1:
        bounds = ndim*bounds
    minimizer_kwargs['bounds'] = bounds
    results = basinhopping(func_scipy,
                                        x0=x0,
                                        stepsize=stepsize,
                                        niter=niter,
                                        minimizer_kwargs=minimizer_kwargs) 
    return results.x, results.fun

def optim_optax(loss,x0,y_low,y_high,steps=100,start_learning_rate=1e-1, n_restarts = 4,jump_sdev = 0.1):
    """
    Minimise the loss function using the Adam optimiser from optax. Slow but can be parallelised.
    """

    optimizer = optax.adam(start_learning_rate)
    params = jnp.array(x0)
    ndim = len(x0)
    opt_state = optimizer.init(params)
    steps = ndim * steps
    # start = time.time()
    
        # model_info = numpyro.infer.util.initialize_model(
        #     rng_key,
        #     self.model,
        #     dynamic_args=True,)
            # model_args=[years],
            # model_kwargs={"tavg": era5_arr, "climsims": cmip6_arr}

    @jit
    def step(carry,xs):
        params, opt_state = carry
        loss_val, gradval = value_and_grad(loss)(params)
        updates, opt_state = optimizer.update(gradval, opt_state)
        params = optax.apply_updates(params, updates)
        params = jnp.clip(params,y_low,y_high) #optax.projections.projection_hypercube(params) # replace with jnp.clip or apply unit transform to params
        carry = params, opt_state
        return carry, loss_val
    
    def findoptim(x0):
        params = x0
        opt_state = optimizer.init(params)
        (params, _ ), loss_vals = scan(step,(params,opt_state),length=steps)
        return (params,loss_vals[-1])
    
    if device_count()>1:
        xi = x0 + jump_sdev*np.random.randn(device_count(),ndim)
        res = pmap(findoptim,devices=devices())(xi)
    else:
        xi = x0 + jump_sdev*np.random.randn(n_restarts,ndim)
        res =  map(findoptim,xi) # vmap?

    best_val, idx = np.min(res[1]), np.argmin(res[1])
    best_params = res[0][idx]

    return best_params, best_val

class Interpolation_Model:
    """
    Base class for interpolation, do not use directly.
    """

    def __init__(self,
                 n_nodes:int,
                 pz_kmin: float,
                 pz_kmax: float,
                 gwb_karr: jnp.ndarray,
                 gwb_means: jnp.ndarray,
                 gwb_cov: jnp.ndarray, 
                 gwb_method: str = 'grid',
                 gwb_method_kwargs: dict = {'s': jnp.linspace(0, 1, 15),   't': jnp.logspace(-4, 4, 200), 
                                            "kernel": "RD", "norm": "RD"},
                 gwb_args_bounds: List[Tuple] = None, 
                 ):
        
        self.n_nodes = n_nodes

        self.gwb_karr = gwb_karr
        self.gwb_means = gwb_means
        self.gwb_cov = gwb_cov
        self.gwb_sigma = jnp.sqrt(jnp.diag(gwb_cov))
        self.gwb_cho = cholesky(gwb_cov)
        self.gwb_logdet = 2*jnp.sum(jnp.log(jnp.diag(self.gwb_cho)))
        self.gwb_len = len(gwb_means)
        self.gwb_logfac = self.gwb_len*jnp.log(2*pi)

        self.logk_min, self.logk_max = log10(pz_kmin), log10(pz_kmax)
        self.k_bounds = jnp.array([pz_kmin,pz_kmax])
        if gwb_method=='grid':
            self.gwb_func = OmegaGWGrid(gwb_karr=self.gwb_karr,)
        elif gwb_method=='jax':
            self.gwb_func = OmegaGWjax(f=self.gwb_karr,**gwb_method_kwargs)
            if gwb_args_bounds is not None:
                self.gwb_args_lows = gwb_args_bounds[0]
                self.gwb_args_highs = gwb_args_bounds[1]
                print(self.gwb_args_lows,self.gwb_args_highs)
                self.extra_args = True
            else:
                self.extra_args = False
        else:
            return ValueError("Not a valid method")

    def model(self):
        raise NotImplementedError("Base interpolation class should not be used directly.")
          
    def run_hmc_inference(self,
                          num_warmup=256,
                          num_samples=256,
                          num_chains=1,
                          progress_bar=True,
                          thinning=1,
                          jit_model_args=False,
                          dense_mass=True,
                          max_tree_depth=6,
                          seed=42):
        """
        Run the HMC inference to sample the posterior distribution of the interpolation model parameters.

        Arguments
        ----------
        num_warmup: int,
            the number of warmup samples
        num_samples: int,
            the number of samples to take after warmup
        num_chains: int,
            the number of chains to run in parallel
        progress_bar: bool,
            whether to display a progress bar
        thinning: int,
            the thinning factor for the samples
        jit_model_args: bool,
            whether to jit the model arguments
        seed: int,
            the seed for the random number generator
        """

        kernel = NUTS(self.model,dense_mass=dense_mass,
                max_tree_depth=max_tree_depth)
        
        mcmc = MCMC(kernel,num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains,
                progress_bar=progress_bar,
                thinning=thinning,
                jit_model_args=jit_model_args)
        
        rng_key, _ = random.split(random.PRNGKey(seed), 2)
        mcmc.run(rng_key,self.extra_args,extra_fields=("potential_energy",)) # this can be made faster - parallel
        samples = mcmc.get_samples()
        extras = mcmc.get_extra_fields()
        mcmc.print_summary(exclude_deterministic=False)
        return samples, extras
    
class Fixed_Nodes_Model(Interpolation_Model):
    """
    Class for interpolation with fixed node positions, uniformly distributed in log10_k space between pz_kmin and pz_kmax.

    Arguments
    ----------
    n_nodes: int,
        the number of nodes
    pz_kmin: float,
        the minimum value of k for the interpolation. P_k = 0 for k < pz_kmin
    pz_kmax: float,
        the maximum value of k for the interpolation. P_k = 0 for k > pz_kmax
    gwb_karr: jnp.ndarray,
        the array of k values for the GWB, assumed to be the k-values over which the GWB has been previously inferred with its mean and covariance
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
    """
        
    def __init__(self,
                 n_nodes:int,
                 pz_kmin: float,
                 pz_kmax: float,
                 gwb_karr: jnp.ndarray,
                 gwb_means: jnp.ndarray,
                 gwb_cov: jnp.ndarray, 
                 gwb_method: str,
                 gwb_method_kwargs: dict = {'s': jnp.linspace(0, 1, 15),   't': jnp.logspace(-4, 4, 200), 
                                            "kernel": "RD", "norm": "RD"},
                 gwb_args_bounds: List[Tuple] = None, 
                 y_low: float = -5., 
                 y_high: float = 1.,
                 ):            
        super().__init__(n_nodes,pz_kmin,pz_kmax,gwb_karr,gwb_means,gwb_cov,gwb_method,
                         gwb_method_kwargs=gwb_method_kwargs,gwb_args_bounds=gwb_args_bounds)
        self.log_k_nodes = jnp.linspace(self.logk_min,self.logk_max,self.n_nodes) 
        self.k_nodes = pow10(self.log_k_nodes)
        self.logy_low = y_low
        self.logy_high = y_high

    def model(self,extra_args=False):
        """
        The numpyro model for the interpolation to be used in the HMC inference. 
        Currently assumes a uniform prior on the log10 P_zeta values at the nodes and a Gaussian distribution of the GWB.
        """
        train_y = self.sample_y()
        if extra_args:
            logk_max = numpyro.sample("logk_max",dist.Uniform(low=self.gwb_args_lows,high=self.gwb_args_highs))
            kmax = numpyro.deterministic("kmax",10**logk_max)
            etaR = numpyro.deterministic("etaR",1./kmax)
            extra_args = jnp.array([kmax,etaR])
            gwb = self.get_gwb_from_xy(self.log_k_nodes,train_y, extra_args)
        else:
            gwb = self.get_gwb_from_xy(x=self.log_k_nodes,y = train_y)
        numpyro.sample('omk',dist.Normal(loc=self.gwb_means,scale = self.gwb_sigma),obs=gwb) # this assumes omks are independent, if off-diagonal cov use below
        # numpyro.sample('omk',dist.MultivariateNormal(loc=self.gwb_means,covariance_matrix=self.gwb_cov),obs=gwb) # type: ignore

    def sample_y(self):
        train_y = numpyro.sample('y',dist.Uniform
                                 (low=self.logy_low*jnp.ones(self.n_nodes),
                                  high=self.logy_high*jnp.ones(self.n_nodes)))  # type: ignore
        return train_y
    
    def spline(self,x, y, k):
        pz = spline_predict(x_train=x,y_train=y,x_pred=k) #
        pz = jnp.where(log10(k)<self.logk_min,0.,pz)
        pz = jnp.where(log10(k)>self.logk_max,0.,pz)
        return pz

    def get_gwb_from_xy(self,x,y,extra_args=None):
        if self.extra_args:
            pz_interp = lambda k, arg1, arg2: self.spline(k=k,x=x,y=y) 
            gwb = self.gwb_func(pz_interp,self.gwb_karr,*extra_args)
        else:
            pz_interp = lambda k: self.spline(k=k,x=x,y=y) 
            gwb = self.gwb_func(pz_interp,self.gwb_karr)
        return gwb

    def loss(self,y):
        """
        The loss function to be minimised in the optimisation process. 
        This is the taken to be the negative log likelihood of the GWB data given the model.
        """
        gwb = self.get_gwb_from_xy(x=self.log_k_nodes,y=y)
        return self._loss(gwb)

    def _loss(self,gwb):
        dgwb = gwb - self.gwb_means
        alpha = cho_solve((self.gwb_cho,False),dgwb)
        chi2 = jnp.dot(dgwb,alpha)
        return 0.5*(chi2 + self.gwb_logdet + self.gwb_logfac) 


class Moving_Nodes_Model(Fixed_Nodes_Model):
    """
    Class for interpolation with variable node positions. 
    In the current implementation the node positions are allowed to vary in bins of size (-log10_kmin + log10_kmax)/(n_nodes-1).
    Note thatn an additional 2 nodes are fixed at pz_kmin and pz_kmax. 

    Arguments
    ----------
    n_nodes: int,
        the number of nodes
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
    """

    def __init__(self,
                 n_nodes:int,
                 pz_kmin: float,
                 pz_kmax: float,
                 gwb_karr: jnp.ndarray,
                 gwb_means: jnp.ndarray,
                 gwb_cov: jnp.ndarray, 
                 gwb_method: str,
                 gwb_method_kwargs: dict = {'s': jnp.linspace(0, 1, 15),   't': jnp.logspace(-4, 4, 200), 
                                            "kernel": "RD", "norm": "RD"},
                 y_low: float = -5.,
                 y_high: float = 1.,
                 ):
        super().__init__(n_nodes,pz_kmin,pz_kmax,gwb_karr,gwb_means,gwb_cov,gwb_method,gwb_method_kwargs,y_low,y_high)

        self.bin_edges = jnp.linspace(0.01,0.99,n_nodes-1) 
        self.bin_edges = unit_untransform(self.bin_edges,bounds=[self.logk_min,self.logk_max])
        self.lows = self.bin_edges[:-1]
        self.highs = self.bin_edges[1:]
        print(self.lows,self.highs)

    def model(self):
        x_bins = numpyro.sample("x_bins",dist.Uniform(low=self.lows,high=self.highs))
        x = numpyro.deterministic("x",jnp.concatenate([jnp.array([self.logk_min]),jnp.array(x_bins),jnp.array([self.logk_max])]) )
        train_y = self.sample_y()
        gwb = super().get_gwb_from_xy(x,train_y)
        # numpyro.sample('omk',dist.Normal(omks_mean,omks_sigma),obs=omks_gp) # this assumes omks are independent, if off-diagonal cov use below
        numpyro.sample('omk',dist.MultivariateNormal(loc=self.gwb_means,covariance_matrix=self.gwb_cov),obs=gwb)

    def loss(self,params):
        x_bins, y = params[:self.n_nodes-2], params[self.n_nodes-2:]
        x = jnp.concatenate([jnp.array([self.logk_min]),jnp.array(x_bins),jnp.array([self.logk_max])])
        gwb = self.get_gwb_from_xy(x=x,y=y)
        return super()._loss(gwb) 
    


    # can cache gwb results to avoid recomputing, may be overkill
    #     def spline(self, x, y, k):
    #     def compute_spline(x, y, k):
    #         pz = spline_predict(x_train=x, y_train=y, x_pred=k)
    #         pz = jnp.where(log10(k) < self.logk_min, 0., pz)
    #         pz = jnp.where(log10(k) > self.logk_max, 0., pz)
    #         return {'x': x, 'y': y, 'pz': pz}

    #     self._cached_spline = cond(
    #         self._cached_spline is None or not (jnp.array_equal(self._cached_spline['x'], x) and jnp.array_equal(self._cached_spline['y'], y)),
    #         lambda _: compute_spline(x, y, k),
    #         lambda _: self._cached_spline,
    #         operand=None
    #     )
    #     return self._cached_spline['pz']

    # def get_gwb_from_xy(self, x, y):
    #     def compute_gwb(x, y):
    #         pz_interp = lambda k: self.spline(x=x, y=y, k=k)
    #         gwb = self.gwb_func(pz_interp, self.gwb_karr)
    #         return {'x': x, 'y': y, 'gwb': gwb}

    #     self._cached_gwb = cond(
    #         self._cached_gwb is None or not (jnp.array_equal(self._cached_gwb['x'], x) and jnp.array_equal(self._cached_gwb['y'], y)),
    #         lambda _: compute_gwb(x, y),
    #         lambda _: self._cached_gwb,
    #         operand=None
    #     )
    #     return self._cached_gwb['gwb']