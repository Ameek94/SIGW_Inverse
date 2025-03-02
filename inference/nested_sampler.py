from inference.base_inference import BaseInference
import tensorflow_probability.substrates.jax as tfp
from jaxns import NestedSampler, TerminationCondition
from jaxns import Model, Prior
from jaxns.framework.special_priors import ForcedIdentifiability
tfpd = tfp.distributions

class NS(BaseInference):
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

        # prepare prior and loglikelihood depending on whether to sample node locations
        # also handle sampling of additional variables required by the gwb model
        model_args = self.model_args.copy()

        def prior():


        def loglikeihood(nodes, values, extra_params):
            

        exact_ns = NestedSampler(model=model, max_samples=1e4,parameter_estimation=True,verbose=False,difficult_model=False)

        termination_reason, state = exact_ns(random.PRNGKey(42),term_cond=TerminationCondition(dlogZ=0.2))
        results = exact_ns.to_results(termination_reason=termination_reason, state=state)
        exact_ns.summary(results)
        return result