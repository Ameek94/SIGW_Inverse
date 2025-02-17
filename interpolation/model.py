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

# log10 = math.log(10.)

### start by defining some useful functions

def pow10(x):
    return jnp.power(10,x)

def log10(x):
    return jnp.log10(x)

@jit
def unit_transform(x,bounds):
    """
    Transform array from original domain given by bounds to [0,1]. 
    Used for k-values over which pzeta is reconstructed, 0 represents kmin and 1 represents kmax.
    """
    ux = (x - bounds[0])/(bounds[1]-bounds[0])
    return ux
@jit
def unit_untransform(x,bounds):
    """
    Transform array to original domain from [0,1]
    """
    xu = x*(bounds[1]-bounds[0]) + bounds[0]
    return xu

@jit
def spline_predict_log(x_train,y_train,x_pred):
    """
    Cubic spline to interpolate log10 P_zeta as a function of log_k
    """
    spl = CubicSpline(x_train,y_train,check=False)
    y_pred = spl(x_pred)
    return y_pred

@jit
def spline_predict(x_train,y_train,x_pred):
    """
    Obtain spline prediction after exponentiating log P spline. 
    """
    x_pred = log10(x_pred)
    return pow10(spline_predict_log(x_train,y_train,x_pred))

def optim_scipy_bh(loss,x0,minimizer_kwargs={'method': 'L-BFGS-B'  },stepsize=1/4,niter=15,bounds=None):

    @jit
    def func_scipy(x):
        val, grad = value_and_grad(loss)(x)
        return val, grad

    ndim = len(x0)    
    minimizer_kwargs['jac'] = True 
    minimizer_kwargs['bounds'] = ndim*[(0,1)] if bounds is None else ndim*[bounds]
    results = basinhopping(func_scipy,
                                        x0=x0,
                                        stepsize=stepsize,
                                        niter=niter,
                                        minimizer_kwargs=minimizer_kwargs) 
    # minimizer_kwargs is for the choice of the local optimizer, bounds and to provide gradient if necessary
    return results.x, results.fun

def optim_optax(loss,x0,y_low,y_high,steps=100,start_learning_rate=1e-1, n_restarts = 4,jump_sdev = 0.1):

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
                 nbins:int,
                 pz_kmin: float,
                 pz_kmax: float,
                 omgw_karr: jnp.ndarray,
                 omgw_means: jnp.ndarray,
                 omgw_cov: jnp.ndarray, 
                 omgw_method: str = 'grid',
                 omgw_method_kwargs: dict = {'s': jnp.linspace(0, 1, 15),   't': jnp.logspace(-4, 4, 200) },
                 ):
        
        self.nbins = nbins
        self.omgw_karr = omgw_karr
        self.omgw_means = omgw_means
        self.omgw_cov = omgw_cov
        self.omgw_sigma = jnp.sqrt(jnp.diag(omgw_cov))
        self.omgw_invcov = jnp.linalg.inv(omgw_cov)
        self.omgw_len = len(omgw_means)
        self.logfac = 0.5*self.omgw_len*jnp.log(2*math.pi) + 0.5*jnp.log(jnp.linalg.det(self.omgw_cov))
        self.logk_min, self.logk_max = log10(pz_kmin), log10(pz_kmax)
        self.bounds = jnp.array([pz_kmin,pz_kmax])
        if omgw_method=='grid':
            self.omgw_func = OmegaGWGrid(omgw_karr=self.omgw_karr,)
        elif omgw_method=='jax':
            self.omgw_func = OmegaGWjax(f=self.omgw_karr,**omgw_method_kwargs)
        else:
            return ValueError("Not a valid method")

    def model(self):
        raise NotImplementedError("Base interpolation class should not be used directly")
          
    def run_hmc_inference(self,
                          num_warmup=256,
                          num_samples=256,
                          num_chains=1,
                          progress_bar=True,
                          thinning=1,
                          jit_model_args=False,
                          seed=42):
        kernel = NUTS(self.model,dense_mass=True,
                max_tree_depth=6)
        mcmc = MCMC(kernel,num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains,
                progress_bar=progress_bar,
                thinning=thinning,jit_model_args=jit_model_args)
        seed=seed
        rng_key, _ = random.split(random.PRNGKey(seed), 2)
        mcmc.run(rng_key,extra_fields=("potential_energy",)) # this can be made faster - parallel
        samples = mcmc.get_samples()
        extras = mcmc.get_extra_fields()
        mcmc.print_summary(exclude_deterministic=False)
        print("First few samples:", {k: v[:5] for k, v in samples.items()})
        return samples, extras
    
class Fixed_Nodes_Model(Interpolation_Model):
        
    def __init__(self,
                 nbins:int,
                 pz_kmin: float,
                 pz_kmax: float,
                 omgw_karr: jnp.ndarray,
                 omgw_means: jnp.ndarray,
                 omgw_cov: jnp.ndarray, 
                 omgw_method: str,
                 omgw_method_kwargs: dict = {'s': jnp.linspace(0, 1, 15),   't': jnp.logspace(-4, 4, 200) },
                 y_low: float = - 5., 
                 y_high: float = 1.,
                 ):            
        super().__init__(nbins,pz_kmin,pz_kmax,omgw_karr,omgw_means,omgw_cov,omgw_method,omgw_method_kwargs=omgw_method_kwargs)
        self.log_k_nodes = jnp.linspace(self.logk_min,self.logk_max,self.nbins) # fixed nodes of the interpolation
        self.k_nodes = pow10(self.log_k_nodes)
        self.logy_low = y_low
        self.logy_high = y_high
        # print(self.omgw_karr.shape)

    def model(self,omgw_model_args=None,omgw_model_kwargs=None):
        # train_y = numpyro.sample('y',dist.Normal(loc=-6*jnp.ones(self.nbins),scale=jnp.ones(self.nbins))) 
        # pz_interp = lambda k: spline_predict(x_train=self.log_k_nodes,y_train=train_y,x_pred=k) #
        # omgw = self.omgw_func(pz_func = pz_interp) #,self.bounds,self.omgw_karr,*omgw_model_args,**omgw_model_kwargs)             numpyro.sample('omk',dist.MultivariateNormal(loc=self.omgw_means,covariance_matrix=self.omgw_cov),obs=omgw)
        train_y = self.sample_y()
        omgw = self.get_omgw_from_y(train_y)
        numpyro.sample('omk',dist.Normal(loc=self.omgw_means,scale = self.omgw_sigma),obs=omgw) # this assumes omks are independent, if off-diagonal cov use below
        # numpyro.sample('omk',dist.MultivariateNormal(loc=self.omgw_means,covariance_matrix=self.omgw_cov),obs=omgw) # type: ignore

    def sample_y(self):
        train_y = numpyro.sample('y',dist.Uniform(low=self.logy_low*jnp.ones(self.nbins),high=self.logy_high*jnp.ones(self.nbins)))  # type: ignore
        return train_y
    
    def spline(self,y, x):
        pz = spline_predict(x_train=self.log_k_nodes,y_train=y,x_pred=x) #
        pz = jnp.where(log10(x)<self.logk_min,0.,pz)
        pz = jnp.where(log10(x)>self.logk_max,0.,pz)
        return pz

    def get_omgw_from_y(self,y):
        pz_interp = lambda x: self.spline(y=y,x=x) #  partial(self.spline,y=y) #
        omgw = self.omgw_func(pz_interp,self.omgw_karr)
        return omgw

    def loss(self,y):
        omgw = self.get_omgw_from_y(y)
        domgw = omgw - self.omgw_means
        return 0.5*jnp.einsum("i,ij,j",domgw,self.omgw_invcov,domgw) + self.logfac
    
    # def run_optimiser(self, x0, steps=100, start_learning_rate=1e-1, n_restarts=4, jump_sdev=0.1):
        # return super().run_optimiser(self.loss, x0, self.y_low, self.y_high, steps, start_learning_rate, n_restarts, jump_sdev)

class Variable_Nodes_Model(Fixed_Nodes_Model):

    def __init__(self,
                 nbins:int,
                 pz_kmin: float,
                 pz_kmax: float,
                 omgw_karr: jnp.ndarray,
                 omgw_means: jnp.ndarray,
                 omgw_cov: jnp.ndarray, 
                 omgw_method: str,
                 omgw_method_kwargs: dict = {'s': jnp.linspace(0, 1, 15),   't': jnp.logspace(-4, 4, 200) },
                 y_low: float = - 8.,
                 y_high: float = 1.,
                 ):
        super().__init__(nbins,pz_kmin,pz_kmax,omgw_karr,omgw_means,omgw_cov,omgw_method,omgw_method_kwargs,y_low,y_high)

        self.bin_edges = jnp.linspace(0.01,0.99,nbins-1) 
        self.bin_edges = unit_untransform(self.bin_edges,bounds=[self.logk_min,self.logk_max])
        self.lows = self.bin_edges[:-1]
        self.highs = self.bin_edges[1:]
        print(self.lows,self.highs)

    def model(self):
        x_bins = numpyro.sample("x_bins",dist.Uniform(low=self.lows,high=self.highs))
        x = numpyro.deterministic("x",jnp.concatenate([jnp.array([self.logk_min]),jnp.array(x_bins),jnp.array([self.logk_max])]) )
        train_y = self.sample_y()
        omgw = self.get_omgw_from_xy(x,train_y)
        # numpyro.sample('omk',dist.Normal(omks_mean,omks_sigma),obs=omks_gp) # this assumes omks are independent, if off-diagonal cov use below
        numpyro.sample('omk',dist.MultivariateNormal(loc=self.omgw_means,covariance_matrix=self.omgw_cov),obs=omgw)

    def spline(self,x, y, k):
        pz = spline_predict(x_train=x,y_train=y,x_pred=k) #
        pz = jnp.where(log10(k)<self.logk_min,0.,pz)
        pz = jnp.where(log10(k)>self.logk_max,0.,pz)
        return pz

    def get_omgw_from_xy(self,x,y):
        pz_interp = lambda k: self.spline(x=x,y=y,k=k) #  partial(self.spline,y=y) #
        omgw = self.omgw_func(pz_interp,self.omgw_karr)
        return omgw

    def loss(self,params):
        x_bins, y = params[:self.nbins-2], params[self.nbins-2:]
        x = jnp.concatenate([jnp.array([self.logk_min]),jnp.array(x_bins),jnp.array([self.logk_max])])
        omgw = self.get_omgw_from_xy(x,y)
        domgw = omgw - self.omgw_means
        return 0.5*jnp.einsum("i,ij,j",domgw,self.omgw_invcov,domgw)
    





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

    # def get_omgw_from_xy(self, x, y):
    #     def compute_omgw(x, y):
    #         pz_interp = lambda k: self.spline(x=x, y=y, k=k)
    #         omgw = self.omgw_func(pz_interp, self.omgw_karr)
    #         return {'x': x, 'y': y, 'omgw': omgw}

    #     self._cached_omgw = cond(
    #         self._cached_omgw is None or not (jnp.array_equal(self._cached_omgw['x'], x) and jnp.array_equal(self._cached_omgw['y'], y)),
    #         lambda _: compute_omgw(x, y),
    #         lambda _: self._cached_omgw,
    #         operand=None
    #     )
    #     return self._cached_omgw['omgw']