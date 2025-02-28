from abc import ABC, abstractmethod
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

class BaseInference(ABC):
    def __init__(self, model, model_args=None):
        """
        Base class for inference.

        Args:
            model (callable): A probabilistic model function (e.g., a numpyro model).
            model_args (dict, optional): Additional keyword arguments to be passed to the model.
        """
        self.model = model
        self.model_args = model_args if model_args is not None else {}

    @abstractmethod
    def run_inference(self, data, **kwargs):
        """
        Run the inference process on the provided data.

        Args:
            data: Observed data to condition on.
            **kwargs: Additional keyword arguments for the inference routine.

        Returns:
            Inference result (e.g., an MCMC object containing posterior samples).
        """
        pass


class NumpyroInference(BaseInference):
    """
    Numpyro inference class using the HMC/NUTS sampler.
    """

    def __init__(self, model, model_args=None, num_warmup=512, num_samples=512, num_chains=1, rng_key=None, verbose=True):
        """
        Initialize the Numpyro inference object.
        
        Arguments
        ----------
        model (callable):
            The numpyro-compatible probabilistic model.
        model_args (dict, optional):
            Additional arguments for the model.
        num_warmup (int, optional): 
            Number of warmup (burn-in) iterations.
        num_samples (int, optional): 
            Number of samples to draw after warmup.
        num_chains (int, optional): 
            Number of parallel chains.
        rng_key: JAX random key 
            (if None, a default key is created).
        """
        super().__init__(model, model_args)
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.verbose = verbose  
        if rng_key is None:
            import jax
            self.rng_key = jax.random.PRNGKey(0)
        else:
            self.rng_key = rng_key

    def run_inference(self, frequencies, data, cov, **kwargs):
        return self.run_hmc_inference(frequencies, data, cov, **kwargs)

    def run_hmc_inference(self, frequencies, data, cov, **kwargs):
        """
        Run inference using Numpyro's NUTS sampler.

        Args:
            data: The observed data.
            **kwargs: Additional keyword arguments passed to MCMC (e.g., progress_bar settings).

        Returns:
            mcmc: A MCMC object containing the posterior samples and diagnostics.
        """
        # Merge model_args with the observed data.
        model_args = self.model_args.copy()
        model_args['data'] = data
        model_args['cov'] = cov
        model_args['frequencies'] = frequencies

        # Set up the NUTS kernel with the provided model.
        kernel = NUTS(self.model,dense_mass=True,max_tree_depth=6)
        mcmc = MCMC(
            kernel,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            num_chains=self.num_chains,
            jit_model_args=False,
            **kwargs
        )
        mcmc.run(self.rng_key, **model_args,extra_fields=('potential_energy',))
        samples = mcmc.get_samples()
        extras = mcmc.get_extra_fields()
        if self.verbose:
            mcmc.print_summary(exclude_deterministic=False)
        return samples, extras