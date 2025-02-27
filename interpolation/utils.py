import jax.numpy as jnp
from jax import vmap, tree
import matplotlib.pyplot as plt
from jaxns.internals.random import resample_indicies

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


def plot_best_spectra():
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,4))

    ax1.loglog(p_arr,pz_bf(p_arr),color='r')
    ax1.loglog(p_arr,pz_amp,color='k',lw=1.5)
    ax2.plot(k_arr,omgw_amp,color='k',lw=1.5,label='Truth')
    ax2.loglog(k_arr,omgw_bf,color='r',label='reconstructed')
    ax2.fill_between(k_arr,omgw_amp+1.96*omks_sigma,omgw_amp-1.96*omks_sigma,alpha=0.2,color='C0')
    ax2.set(yscale='log',xscale='log')
    ax1.set_ylabel(r'$P_{\zeta}(k)$')
    ax1.set_xlabel(r'$k$')
    ax2.set_ylim(1e-4,1.)

    ax2.set_ylabel(r'$\Omega_{\mathrm{GW}}(k)$')
    ax2.set_xlabel(r'$k$')
    ax2.legend()
    for val in nodes:
        ax1.axvline(jnp.exp(val),color='k',ls='-.',alpha=0.5)
    ax1.scatter(jnp.exp(nodes),jnp.exp(best_params),color='r')
    fig.tight_layout()


# minor modifications of jax internal utils to get loglike at samples
def resample(key, samples, log_weights, S: int = None,
             replace: bool = True):
    """
    Resample the weighted samples into uniformly weighted samples.

    Args:
        key: PRNGKey
        samples: samples from nested sampled results
        log_weights: log-posterior weight
        S: number of samples to generate. Will use Kish's estimate of ESS if None.
        replace: whether to sample with replacement

    Returns:
        equally weighted samples
    """
    idx = resample_indicies(key, log_weights, S=S, replace=replace)
    return tree.map(lambda s: s[idx, ...], samples), idx
