import jax.numpy as jnp
from jax import vmap, jit
import matplotlib.pyplot as plt

# minor modifications of https://github.com/martinjankowiak/saasbo/blob/main/util.py
def split_vmap(func,input_arrays,batch_size=8):
    """
    Utility to split vmap over a function taking multiple arrays as input into multiple chunks, useful for reducing memory usage.
    """
    num_inputs = input_arrays[0].shape[0]
    num_batches = (num_inputs + batch_size - 1 ) // batch_size
    batch_idxs = [jnp.arange( i*batch_size, min( (i+1)*batch_size,num_inputs  )) for i in range(num_batches)]
    res = [vmap(func)(*tuple([arr[idx] for arr in input_arrays])) for idx in batch_idxs]
    nres = len(res[0])
    # now combine results across batches and function outputs to return a tuple (num_outputs, num_inputs, ...)
    results = tuple( jnp.concatenate([x[i] for x in res]) for i in range(nres))
    return results

def plot_mcmc_spectra(samples,):
    pass
@jit
def unit_transform(x,bounds):
    """
    Transform array from original domain given by bounds to [0,1]. 
    """
    ux = (x - bounds[0])/(bounds[1]-bounds[0])
    return ux
@jit
def unit_untransform(x,bounds):
    """
    Transform array to original domain given by bounds from [0,1]
    """
    xu = x*(bounds[1]-bounds[0]) + bounds[0]
    return xu


def plot_functional_posterior(funcs,samples,k_arr = [], intervals = [95,68],ylabels=[r'$P_{\zeta}$',r'$\Omega_{\rm GW}$'],aspect_ratio = (6,4), interval_cols = [('r',0.2),('b',0.5)]):
    # given a function y = f(k|x) with x~Posterior samples, plot the posterior of y at k_arr, with symmetric credible intervals

    nfuncs = len(funcs)

    fig, ax = plt.subplots(1,nfuncs,figsize=(aspect_ratio[0]*nfuncs,aspect_ratio[1]),constrained_layout=True)
    if nfuncs == 1:
        ax = [ax]
    for i,func in enumerate(funcs):
        y = func(k_arr[i],samples) # so y should have shape (nsamples, nk)
        for j,interval in enumerate(intervals):
            y_low, y_high = np.percentile(y,[50-interval/2,50+interval/2],axis=0)
            ax[i].fill_between(k_arr[i],y_low,y_high,color=interval_cols[j])
        ax[i].plot(k_arr[i],np.median(y,axis=0),color='b',lw=2)
        ax[i].set_ylabel(ylabels[i])
    for x in ax:
        x.set(xscale='log',yscale='log',xlabel=r'$k$')
    return fig, ax 