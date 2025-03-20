import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from omega_gw_jax import OmegaGWjax
from getdist import plots, MCSamples, loadMCSamples
from interpax import CubicSpline
from jax import config, vmap
config.update("jax_enable_x64", True)


# Set matplotlib parameters
font = {'size': 16, 'family': 'serif'}
axislabelfontsize = 'large'
matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=True)
matplotlib.rc('legend', fontsize=16)

# input model and files

model = str(sys.argv[1])


import re
# ...existing code...

# Check current working directory for files matching the pattern
pattern = re.compile(rf'nautilus_{model}_(\d+)_nodes\.npz')

# List all files in the current working directory
files_in_dir = os.listdir(os.getcwd())

# Filter files matching the pattern where n > 2
matching_files = [f for f in files_in_dir if pattern.match(f) and int(pattern.match(f).group(1)) > 2]



plt.figure(figsize=(6,4))
plt.plot()
plt.xlabel(r'Number of nodes')
plt.ylabel(r'$\log Z$')
# Get data from matching files
Num_nodes = []
logZ = []
if matching_files:
    print("Matching files found:")
    for file in matching_files:
        n = int(pattern.match(file).group(1))
        Num_nodes.append(n)
        data = np.load(file)
        logZ.append(data['logz'].item())
        # print(file)
else:
    print("No matching files found.")

Num_nodes, logZ = zip(*sorted(zip(Num_nodes, logZ)))
# Convert back to lists
Num_nodes = list(Num_nodes)
logZ = list(logZ)

print(logZ, Num_nodes)
plt.plot(Num_nodes, logZ, '-.',color='k',alpha=0.9)
plt.scatter(Num_nodes, logZ, color='k',marker='x',s=20)
# Annotate each point with its logZ value
ax = plt.gca()
y_min = min(logZ) - 25
y_max = max(logZ) + 25
ax.set_ylim(y_min, y_max)
ax.set_xlim(min(Num_nodes) - 0.5, max(Num_nodes) + 0.5)
y_mid = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2
for x, y in zip(Num_nodes, logZ):
    plt.text(x+0.05, y-20, f'({y:.2f})', fontsize=10, ha='center', va='bottom')
plt.savefig(f'./{model}_logz.pdf',bbox_inches='tight')