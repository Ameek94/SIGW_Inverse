# SIGW_Inverse
 
Code to reconstruct the primordial curvature power spectrum from the scalar induced GW spectrum. See the jax and sigwfast folders for examples on how to use the code. 

## Requirements

To use this code, you will need the following Python packages:

- numpy
- scipy
- matplotlib
- jax
- jaxlib
- getdist
- nautilus-sampler
- h5py
- tqdm
- mpi4py (for MPI parallelism, optional)


If you use this code in your works please cite our paper
```
@article{,
}
```


Please also cite [SIGWAY](https://github.com/jonaselgammal/SIGWAY) and [SIGWFast](https://github.com/Lukas-T-W/SIGWfast) depending on which method you use to calculate the scalar induced gravitational wave background. If you use any of the nested sampling routines, consider also citing them.

We have slightly modified the SIGWFast code
