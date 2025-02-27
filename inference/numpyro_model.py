# Numpyro model 

import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
from utils import unit_transform, unit_untransform
from interpolation.spline import spline_predict

def model(frequencies, data, cov, interpolator, gwb_calculator, 
          use_extra_params=False,
          sample_nodes=False, 
          fixed_nodes=None,
          pk_bounds = jnp.array([-8,1]), 
          num_nodes=5,
          amp_bounds = jnp.array([-8,1]),):
    """
    A NumPyro model that combines the P_zeta interpolation
    with the OmegaGW calculation.

    Arguments
        data: The observed data.
        cov: The covariance matrix for the likelihood.
        interpolator: An interpolation object instance
        gwb_calculator: A gravitational wave spectrum calculator instance.
        num_nodes: Number of spline nodes for the power spectrum.
        use_extra_params: Flag to indicate whether to sample extra parameters.
    """
    # Sample power spectrum parameters.
    if sample_nodes:
        unodes, nodes = interpolator.get_nodes()
    else:
        nodes = interpolator.get_nodes()
        # if fixed_nodes is None:
        #     raise ValueError("Fixed nodes must be provided if sample_nodes is False.")
        # else:
        #     nodes = fixed_nodes
    values = numpyro.sample("values", dist.Uniform(low=amp_bounds[0],high=amp_bounds[1]).expand([num_nodes]))
    
    # nodes = numpyro.deterministic("nodes",unit_untransform(nodes,pk_bounds))
    # Interpolate to form the power spectrum.
    # power_spectrum = lambda k: spline_predict(nodes,values,k) 
    power_spectrum = interpolator.interpolate(nodes, values)
    
    # Conditionally sample extra parameters and calculate the GW spectrum.
    extra_params = {}
    if use_extra_params:
        for param_name, param_dist in gwb_calculator.get_extra_param_specs().items():
            extra_params[param_name] = numpyro.sample(param_name, param_dist)
        gw_spectrum = gwb_calculator(power_spectrum, frequencies, **extra_params)

    else:
        # # Use default values for extra parameters.
        # for param_name in gwb_calculator.get_extra_param_specs().keys():
        #     extra_params[param_name] = 0.0
        gw_spectrum = gwb_calculator(power_spectrum, frequencies)

    # Define the likelihood.
    numpyro.sample("obs", dist.MultivariateNormal(gw_spectrum, cov), obs=data)

def ordering_transform(x):
    """
    A bijective transform to order spline nodes. Takes input in [0,1] and returns ordered output in [0,1].
    """
    len = x.shape[0]
    i = jnp.arange(len)
    inner = jnp.power(1-x,1/(len-i))
    return 1 - jnp.cumprod(inner)
