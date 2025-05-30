# SIGW_Inverse
 
Code to reconstruct the primordial curvature power spectrum from the scalar induced GW spectrum based on the method outlined in [arxiv:2505.22534](https://arxiv.org/abs/2505.22534). See the jax and sigwfast folders for examples on how to use the code. 

## Requirements

To use this code, you will need the following Python packages (code is developed and tested with python>=3.12):

- numpy
- scipy
- matplotlib
- jax
- jaxlib
- getdist
- nautilus-sampler
- h5py
- tqdm (optional, can be commented out)
- mpi4py (for MPI parallelism, optional)


If you use this code in your works please cite our paper
```
@article{Ghaleb:2025xqn,
    author = "Ghaleb, Aya and Malhotra, Ameek and Tasinato, Gianmassimo and Zavala, Ivonne",
    title = "{Bayesian reconstruction of primordial perturbations from induced gravitational waves}",
    eprint = "2505.22534",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    month = "5",
    year = "2025"
}
```


Please also cite [SIGWAY](https://github.com/jonaselgammal/SIGWAY) and [SIGWFast](https://github.com/Lukas-T-W/SIGWfast) depending on which method you use to calculate the scalar induced gravitational wave background. If you use any of the nested sampling routines, consider also citing them.

If you would like to use SIGWFast for reconstruction in the general equation of state case, please go through the instructions in the sigwfast folder.
