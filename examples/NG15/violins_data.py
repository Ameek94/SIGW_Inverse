import numpy as np
from jax import config
config.update("jax_enable_x64", True)
from omega_gw_jax import OmegaGWjax
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
import matplotlib.pyplot as plt
import jax.numpy as jnp
from plot_utils import split_vmap, plot_functional_posterior, resample_equal

# # Command line argument for number of nodes
# if len(sys.argv) != 2:
#     print("Usage: python script.py <num_nodes>")
#     sys.exit(1)

# num_nodes = int(sys.argv[1])
# free_nodes = num_nodes - 2
# Load data files
datadir = './NG15_Ceffyl/30f_fs{hd}_ceffyl/'
density = np.load(datadir + "density.npy").squeeze(axis=0)  # shape: (n_frequencies, n_grid_points)
log10rhogrid = np.load(datadir + "log10rhogrid.npy")  # grid for log10rho values
freqs = np.load(datadir + "freqs.npy")  # GW frequencies
# Number of samples to draw from each KDE distribution
n_samples = 100000

# Prepare a list to store sampled data for each frequency
data_list = []

np.random.seed(1000)

# Assuming density has shape (n_frequencies, n_grid_points)
for i in range(density.shape[0]):
    log_pdf = density[i]
    # Exponentiate the log PDF (subtract max for numerical stability)
    pdf = np.exp(log_pdf - np.max(log_pdf))
    # Normalize the PDF so that its sum equals 1
    pdf /= np.sum(pdf)

    # Draw samples from the log10rho grid weighted by the PDF
    samples = np.random.choice(log10rhogrid, size=n_samples, p=pdf)
    data_list.append(samples)

data_list = np.array(data_list)

# convert to OmegaGW
Tspan = 505861299.1401643 #15 * 365 * 24 * 3600
h = 0.672
H_0 = h * 100 * nat.convert(nat.km * nat.s**-1 * nat.Mpc**-1, nat.GeV)  # Hubble constant (GeV)
H_0_Hz = H_0 * nat.convert(nat.GeV, nat.Hz)
OmegaGW_data = (8 * np.pi**4 * freqs[:, None]**5 * h**2 * 10**(2 * data_list + np.log10(Tspan))) / (H_0_Hz**2)
OmegaGW_data = np.log10(OmegaGW_data)

np.savez('./violins_data.npz', OmegaGW_data=OmegaGW_data, freqs=freqs)