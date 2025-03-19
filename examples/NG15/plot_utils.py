import jax.numpy as jnp
from jax import vmap, jit
import matplotlib.pyplot as plt
from jaxns.internals.random import resample_indicies
import numpy as np
from scipy.special import logsumexp

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

import math
SQRTEPS = math.sqrt(float(np.finfo(np.float64).eps))
def resample_equal(samples,logl, logwt, rstate):
    # Extract the weights and compute the cumulative sum.
    # wt = np.exp(logwt)
    # weights = wt / wt.sum()
    log_total = logsumexp(logwt)
    weights = np.exp(logwt - log_total) 
    cumulative_sum = np.cumsum(weights)

    # if abs(cumulative_sum[-1] - 1.) > SQRTEPS:
    #     # same tol as in numpy's random.choice.
    #     # Guarantee that the weights will sum to 1.
    #     warnings.warn("Weights do not sum to 1 and have been renormalized.")
    cumulative_sum /= cumulative_sum[-1]
    # this ensures that the last element is strictly == 1

    # Make N subdivisions and choose positions with a consistent random offset.
    nsamples = len(weights)
    print(nsamples)

    positions = (rstate.random() + np.arange(nsamples)) / nsamples

    # Resample the data.
    idx = np.zeros(nsamples, dtype=int)
    i, j = 0, 0
    while i < nsamples:
        if positions[i] < cumulative_sum[j]:
            idx[i] = j
            i += 1
        else:
            j += 1
    
    perm = rstate.permutation(nsamples)
    resampled_samples = samples[idx][perm]
    resampled_logl = logl[idx][perm]
    return resampled_samples, resampled_logl

def plot_functional_posterior(vals = [],k_arr = [], intervals = [99.7,95.,68.],
                              ylabels=[r'$P_{\zeta}$',r'$\log_{10} \Omega_{\rm GW}$'],aspect_ratio = (6,5)
                              , interval_cols = [('#006FED',0.2),('#006FED',0.4),('#006FED',0.6)]):
    # given a function y = f(k|x) with x~Posterior samples, plot the posterior of y at k_arr, with symmetric credible intervals

    nfuncs = len(vals)

    fig, ax = plt.subplots(1,nfuncs,figsize=(aspect_ratio[0]*nfuncs,aspect_ratio[1]),constrained_layout=True)
    if nfuncs == 1:
        ax = [ax]
    for i,val in enumerate(vals):
        for j,interval in enumerate(intervals):
            y_low, y_high = np.percentile(val,[50-interval/2,50+interval/2],axis=0)
            ax[i].fill_between(k_arr[i],y_low,y_high,color=interval_cols[j])
        ax[i].plot(k_arr[i],np.median(val,axis=0),color='#006FED',lw=2.5)
        ax[i].set_ylabel(ylabels[i])
    # for x in ax:
    #     x.set(xscale='log',yscale='linear',xlabel=r'$f\,{\rm [Hz]}$')
    return fig, ax 

