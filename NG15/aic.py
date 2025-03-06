import numpy as np
from GWB_Jax import OmegaGWjax
from interpax import CubicSpline
import matplotlib
from matplotlib import cm, colors
import natpy as nat
import warnings
from getdist import plots, MCSamples, loadMCSamples
import sys
font = {'size'   : 24, 'family':'serif'}
axislabelfontsize='large'
matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=True) # if using latex in plots
matplotlib.rc('legend', fontsize=24)

ll = []
aic = []

nodes = [2,3,4,5]

for n in nodes:
    best_n = -1e5
    for i in range(4):
        chain_data = np.loadtxt(f"./chains/sigw_hd_{n}_nodes/chain_{i}/chain_1.txt")
        best = np.max(chain_data[:, n+1])
        if best_n < best:
            best_n = best
    ll.append(best_n)
    aic_n = 2*n - 2*best_n
    aic.append(aic_n)

results = dict(zip(nodes, zip(ll,aic)))
print(results)