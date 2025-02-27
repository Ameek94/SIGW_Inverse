# config.py

config = {
    # Interpolation settings
    "interpolator": {
        "method": "cubic_spline",  # Could be "cubic_spline", "linear", etc.
    },

    # Scalar induced gravitational wave spectrum settings
    "gwb": {
        "method": "jax",  # Could be "jax", "grid", etc.
    },

    # k-range over which to interpolate the power spectrum, if k outside range pz=0
    "pz_k_range": "auto",

    # Number of nodes in the power spectrum interpolation
    "num_nodes": 5,

    # Whether to sample the node locations or use fixed values
    # If node locations are sampled, the total number of nodes will be num_nodes+2 with 2 fixed nodes at the boundaries
    # Else, fixed_nodes will be set automatically as jnp.geomspace(kmin, kmax, num_nodes)
    "sample_nodes": True,

    # Whether to include additional parameters in the inference
    "use_extra_params": True,

    # Path to gwb data (k, OmegaGW, cov) for likelihood calculation
    "data_file": "",

    # NumPyro inference settings
    "num_warmup": 512,  # Number of warm-up (burn-in) iterations
    "num_samples": 512,  # Number of posterior samples
    "num_chains": 1,  # Number of MCMC chains
    "thin": 1,  # Thinning factor for the MCMC samples
}
