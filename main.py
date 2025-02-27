# main.py
from jax import config
config.update("jax_enable_x64", True)
import numpyro
numpyro.enable_x64()
from inference.numpyro_model import model
from inference.numpyro_infer import NumpyroInference
from interpolation.spline import Spline
from gwb.omega_gw_jax import OmegaGWjax
from sigw_config import config as sigw_config # Assume this holds your configuration settings

def main():
    # Instantiate your components.
    interpolator = Spline(**config['interpolator'])
    spectrum_calculator = OmegaGWjax(**config['spectrum'])
    
    # Set flags and fixed node values according to user input.
    sample_nodes = sigw_config.get('sample_nodes', True)
    fixed_nodes = sigw_config.get('fixed_nodes', None)  # Should be provided if sample_nodes is False
    
    # Create the inference engine with the unified model.
    inference_engine = NumpyroInference(
        model=model,
        model_args={
            'interpolator': interpolator,
            'spectrum_calculator': spectrum_calculator,
            'num_nodes': sigw_config.get('num_nodes', 5),
            'noise_std': sigw_config.get('noise_std', 1.0),
            'sample_nodes': sample_nodes,
            'fixed_nodes': fixed_nodes,
            'use_extra_params': sigw_config.get('use_extra_params', True)
        },
        num_warmup=1000,
        num_samples=1000,
        num_chains=1
    )
    
    # Load or generate your observed data.
    data = ...  # Your observation data here.
    
    # Run the inference.
    mcmc = inference_engine.run_inference(data)
    print(mcmc.get_samples())

if __name__ == '__main__':
    main()
