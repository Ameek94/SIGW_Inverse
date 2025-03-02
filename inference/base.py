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
