import time
from jax import config, jit
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from sigw_utils import split_vmap
import numpy as np
try:
    from nautilus import Sampler
except ModuleNotFoundError:
    print("Please install the Nautilus package to run the nested sampler.")
    raise ModuleNotFoundError


def Gaussian_Likelihood(x, means, cov):
    """
    Compute the Log-likelihood for a Gaussian given the means and covariance matrix.
    """
    return multivariate_normal.logpdf(x, means, cov)

def LogNormal_Likelihood(x, means, sigmas):
    """
    Compute the Log-likelihood for a Log-normal given the means and covariance matrix.
    """
    cov = jnp.diag(sigmas**2)
    return multivariate_normal.logpdf(jnp.log(x), means, cov)

def prior_transform_1D(cube,n_free_nodes,lower_bounds,upper_bounds):
    """
    Transform the unit hypercube to the prior space. Params is assumed to be 1D array of size Ndim. Lower and upper bounds are assumed to be of Ndim shape.
    """
    # Order and transform nodes to be in the correct range
    params = cube.copy()
    x = params[:n_free_nodes]
        # x = params[:,:free_nodes]
        # Npoints = cube.shape[0]
    N = n_free_nodes
    t = np.zeros(N)
    t[N-1] = x[N-1]**(1./N)
    for n in range(N-2, -1, -1):
        t[n] = x[n]**(1./(n+1)) * t[n+1]
    params = np.concatenate((t, params[n_free_nodes:]))
    # now transform the params to the correct range
    params = params*(upper_bounds - lower_bounds) + lower_bounds
    return params

def prior_transform_vectorized(cube,n_free_nodes,lower_bounds,upper_bounds):
    """
    Transform the unit hypercube to the prior space. Params is assumed to be of Npoints x Ndim shape. Lower and upper bounds are assumed to be of Ndim shape.
    """
    params = cube.copy()
    # for the free nodes apply the ordering transform, this keeps the parameters in [0,1]
    x = params[:, :n_free_nodes]
    exponents = 1.0 / jnp.arange(1, n_free_nodes + 1)
    x_vals = x ** exponents  # shape (Npoints, free_nodes)
    t_arr = jnp.cumprod(x_vals[:, ::-1], axis=1)[:, ::-1]
    params = jnp.concatenate((t_arr, params[:, n_free_nodes:]), axis=1)
    # now transform the params to the correct range
    params = params*(upper_bounds[:,None] - lower_bounds[:,None]) + lower_bounds[:,None]
    return params

def interpolate(nodes, vals, x, left_node, right_node):
    # Create a cubic spline interpolation of log10(Pζ) and then convert back to linear scale.
    # spl = CubicSpline(nodes, vals, check=False)
    # Testing linear interpolation
    # spl = lambda x: 
    res = jnp.power(10, jnp.interp(x, nodes, vals))
    res = jnp.where(x < left_node, 0, res)
    res = jnp.where(x > right_node, 0, res)
    return res

def get_gwb(nodes, vals, gwb_calculator, frequencies, gwb_calculator_kwargs={}):
    # Given nodes and values, create a function for Pζ and compute Ω_GW. Returns a tuple so that it can be used with split vmap.
    pf = lambda k: interpolate(nodes, vals, jnp.log10(k))
    omegagw = gwb_calculator(pf, frequencies, **gwb_calculator_kwargs)
    return (omegagw,)


def run_nested(num_nodes, left_node, right_node, y_min, y_max,
               gwb_calculator,frequencies,gwb_calculator_kwargs={},likelihood_args=[], 
               sampler_kwargs = {'pool': None, 'filepath': 'model'}, 
               ns_run_kwargs = {'verbose':True, 'f_live':0.005, 'n_like_max': int(5e6)},
               vectorized=True,vmap_batch_size=100):
    """
    Run the nested sampler.
    """
    n_free_nodes = num_nodes - 2
    ndim = n_free_nodes + num_nodes
    lower_bounds = jnp.array([left_node]*n_free_nodes + [y_min]*num_nodes)
    upper_bounds = jnp.array([right_node]*n_free_nodes + [y_max]*num_nodes)

    


    # First prepare the prior transform and likelihood for the nested sampler.
    if vectorized:
        prior_transform = jit(lambda cube: prior_transform_vectorized(jnp.array(cube), num_nodes, lower_bounds, upper_bounds))
    else:
        prior_transform = lambda cube: prior_transform_1D(cube, num_nodes, lower_bounds, upper_bounds)
    
    # if likelihood args is empty, generate dummy means and cov based on frequencies
    if len(likelihood_args) == 0:
        likelihood_args = [jnp.ones_like(frequencies), jnp.eye(len(frequencies))]
    gwb = jit(lambda nodes,vals: get_gwb(nodes, vals, gwb_calculator, frequencies, gwb_calculator_kwargs))
    log_likelihood = jit(lambda params: Gaussian_Likelihood(params, *likelihood_args))

    def loglikelihood(params):
        params = jnp.atleast_2d(params)
        nodes = params[:, :n_free_nodes]
        # Pad nodes with fixed endpoints
        nodes = jnp.pad(nodes, ((0, 0), (1, 1)), 'constant',
                      constant_values=((0, 0), (left_node, right_node)))
        vals = params[:, n_free_nodes:]
        omegagw = split_vmap(gwb, (nodes, vals), batch_size=vmap_batch_size)[0]
        return log_likelihood(omegagw, *likelihood_args)

    start = time.time()
    outfile = sampler_kwargs.pop('filepath', 'model')
    sampler_kwargs['filepath'] = outfile+'.h5'
    sampler = Sampler(prior_transform,loglikelihood, ndim, pass_dict=False, vectorized=True, **sampler_kwargs)
    sampler.run(**ns_run_kwargs)
    end = time.time()
    print('Time taken: {:.4f} s'.format(end - start))
    print('log Z: {:.4f}'.format(sampler.log_z))

    # Retrieve posterior samples and save
    samples, logl, logwt = sampler.posterior()
    np.savez(outfile+f'_{num_nodes}.npz', samples=samples, logl=logl, logwt=logwt, logz=sampler.log_z)