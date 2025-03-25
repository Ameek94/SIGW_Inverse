# base class for interpolation
import jax.numpy as jnp
# from utils import unit_untransform
import numpyro
from numpyro import distributions as dist
from jax import jit

@jit
def unit_untransform(x,bounds):
    """
    Transform array to original domain given by bounds from [0,1]
    """
    xu = x*(bounds[1]-bounds[0]) + bounds[0]
    return xu

class Interpolator:

    def __init__(self, k_min, k_max, sample_nodes=False, fixed_nodes = None, num_nodes=5):
        """
        Base class for interpolation methods.
        """
        self.k_min = jnp.log10(k_min)
        self.k_max = jnp.log10(k_max)
        self.bounds = jnp.array([self.k_min, self.k_max])
        self.sample_nodes = sample_nodes
        self.num_nodes = num_nodes
        self.fixed_nodes = fixed_nodes if fixed_nodes is not None else jnp.linspace(self.k_min, self.k_max, num_nodes)
        #add binned nodes

    def get_nodes(self):
        if self.sample_nodes:
            unodes = numpyro.sample("y", dist.Dirichlet(jnp.ones(self.num_nodes - 1)))
            # Compute the cumulative sum to obtain an ordered vector in (0,1)
            cumulative = jnp.cumsum(unodes)
            x_inner = cumulative[:-1]
            # Now, attach the fixed endpoints 0 and 1.
            x = jnp.concatenate([jnp.array([0.]), x_inner, jnp.array([1.])])
            nodes = numpyro.deterministic("nodes",unit_untransform(x, self.bounds))
            return unodes, nodes
        else:
            return self.fixed_nodes


    def interpolate(self, nodes, values):
        raise NotImplementedError("Interpolation method must be implemented.")
    
    def __call__(self, nodes, values, x):
        raise NotImplementedError("Interpolation method must be implemented.")
