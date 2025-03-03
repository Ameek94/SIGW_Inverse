# base class for interpolation
import numpy as np
from utils import unit_transform, unit_untransform
import numpyro
from numpyro import distributions as dist


class Interpolator:

    def __init__(self, k_min, k_max, sample_nodes=False, fixed_nodes = None, num_nodes=5):
        """
        Base class for interpolation methods.
        """
        self.k_min = np.log10(k_min)
        self.k_max = np.log10(k_max)
        self.bounds = np.array([self.k_min, self.k_max])
        self.sample_nodes = sample_nodes
        self.num_nodes = num_nodes
        self.fixed_nodes = fixed_nodes if fixed_nodes is not None else np.linspace(self.k_min, self.k_max, num_nodes)
        #add binned nodes

    def get_nodes(self):
        if self.sample_nodes:
            unodes = numpyro.sample("y", dist.Dirichlet(np.ones(self.num_nodes - 1)))
            # Compute the cumulative sum to obtain an ordered vector in (0,1)
            cumulative = np.cumsum(unodes)
            x_inner = cumulative[:-1]
            # Now, attach the fixed endpoints 0 and 1.
            x = np.concatenate([np.array([0.]), x_inner, np.array([1.])])
            nodes = numpyro.deterministic("nodes",unit_untransform(x, self.bounds))
            return unodes, nodes
        else:
            return self.fixed_nodes


    def interpolate(self, nodes, values):
        raise NotImplementedError("Interpolation method must be implemented.")
    
    def __call__(self, nodes, values, x):
        raise NotImplementedError("Interpolation method must be implemented.")
