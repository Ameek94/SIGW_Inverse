import numpy as np
import math
SQRTEPS = math.sqrt(float(np.finfo(np.float64).eps))
import matplotlib.pyplot as plt

# dynesty routine to get equally weighted samples, slightly modified
def resample_equal(results, rstate):
    # Extract the weights and compute the cumulative sum.
    logwt = results['logwt'] - results['logz'][-1]
    wt = np.exp(logwt)
    weights = wt / wt.sum()
    cumulative_sum = np.cumsum(weights)

    # if abs(cumulative_sum[-1] - 1.) > SQRTEPS:
    #     # same tol as in numpy's random.choice.
    #     # Guarantee that the weights will sum to 1.
    #     warnings.warn("Weights do not sum to 1 and have been renormalized.")
    cumulative_sum /= cumulative_sum[-1]
    # this ensures that the last element is strictly == 1

    # Make N subdivisions and choose positions with a consistent random offset.
    nsamples = len(weights)
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

    samples = results['samples']
    logl = results['logl']
    
    perm = rstate.permutation(nsamples)
    resampled_samples = samples[idx][perm]
    resampled_logl = logl[idx][perm]
    return resampled_samples, resampled_logl

#------------------------------------------------------------Plotting------------------------------------------------------------#

def plot_functional_posterior(funcs,samples,k_arr,intervals = [95,68],ylabel='y',aspect_ratio = (6,4), interval_cols = [('r',0.75),('r',0.5)]):
    # given a function y = f(k|x) with x~Posterior samples, plot the posterior of y at k_arr, with symmetric credible intervals

    nfuncs = len(funcs)

    fig, ax = plt.subplots(1,nfuncs,figsize=(aspect_ratio[0]*nfuncs,aspect_ratio[1]),constrained_layout=True)
    if nfuncs == 1:
        ax = [ax]
    for i,func in enumerate(funcs):
        y = [func(k,samples) for k in k_arr] 
        for j,interval in enumerate(intervals):
            y_low, y_high = np.percentile(y,[50-interval/2,50+interval/2],axis=0)
            ax[i].fill_between(k_arr,y_low,y_high,color=interval_cols[j])
        ax[i].plot(k_arr,np.median(y,axis=0),color='k',lw=2)

    return fig, ax 

    