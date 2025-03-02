from inference.base_inference import BaseInference
import tensorflow_probability.substrates.jax as tfp
from jaxns import NestedSampler, TerminationCondition
from jaxns import Model, Prior
from jaxns.framework.special_priors import ForcedIdentifiability
tfpd = tfp.distributions

class NestedSampler(BaseInference):
    """
    Nested sampling inference class using the JAX NestedSampler.
    """

    def __init__(self, model, model_args=None,
                  sampler_kwarsgs=None, termination_kwargs = None,verbose=True):
        """
        Initialize the NestedSampler inference object.
        
        Arguments
        ----------
        model (callable):
            The model function.
        model_args (dict, optional):
            Additional arguments for the model.
        num_live_points (int, optional): 
            Number of live points.
        termination_frac (float, optional): 
            Fraction of remaining evidence to terminate at.
        sampler_name (str, optional): 
            Name of the sampler to use.
        sampler_kwargs (dict, optional): 
            Additional keyword arguments for the sampler.
        verbose (bool, optional): 
            Whether to print progress.
        """
        super().__init__(model, model_args)

    def run_inference(self, data, **kwargs):
        """
        Run the NestedSampler inference process on the provided data.
        
        Arguments
        ----------
        data: Observed data to condition on.
        **kwargs: Additional keyword arguments for the inference routine.
        
        Returns
        -------
        Inference result (e.g., a NestedSampler object containing posterior samples).
        """

        # prepare prior and loglikelihood depending on whether to sample nodes,

        def prior():


        def loglikeihood(nodes, values)

        model = Model(self.model, data, **self.model_args)
        prior = Prior(self.model, **self.model_args)
        termination_conditions = [TerminationCondition(self.termination_frac)]
        sampler = NestedSampler(model, prior, termination_conditions, sampler_name=self.sampler_name, num_live_points=self.num_live_points, **self.sampler_kwargs)
        result = sampler.run()
        return result