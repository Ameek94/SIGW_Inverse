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

# Command line argument for number of nodes
if len(sys.argv) != 2:
    print("Usage: python script.py <num_nodes>")
    sys.exit(1)

num_nodes = int(sys.argv[1])
free_nodes = num_nodes - 2
# Load data files
datadir = './NG15_Ceffyl/30f_fs{hd}_ceffyl/'
# density = np.load(datadir + "density.npy").squeeze(axis=0)  # shape: (n_frequencies, n_grid_points)
# log10rhogrid = np.load(datadir + "log10rhogrid.npy")  # grid for log10rho values
# freqs = np.load(datadir + "freqs.npy")  # GW frequencies
# # Number of samples to draw from each KDE distribution
# n_samples = 50000

# # Prepare a list to store sampled data for each frequency
# data_list = []

# # Assuming density has shape (n_frequencies, n_grid_points)
# for i in range(density.shape[0]):
#     log_pdf = density[i]
#     # Exponentiate the log PDF (subtract max for numerical stability)
#     pdf = np.exp(log_pdf - np.max(log_pdf))
#     # Normalize the PDF so that its sum equals 1
#     pdf /= np.sum(pdf)

#     # Draw samples from the log10rho grid weighted by the PDF
#     samples = np.random.choice(log10rhogrid, size=n_samples, p=pdf)
#     data_list.append(samples)

# data_list = np.array(data_list)

# # convert to OmegaGW
# Tspan = 505861299.1401643 #15 * 365 * 24 * 3600
# h = 0.672
# H_0 = h * 100 * nat.convert(nat.km * nat.s**-1 * nat.Mpc**-1, nat.GeV)  # Hubble constant (GeV)
# H_0_Hz = H_0 * nat.convert(nat.GeV, nat.Hz)
# OmegaGW_data = (8 * np.pi**4 * freqs[:, None]**5 * h**2 * 10**(2 * data_list + np.log10(Tspan))) / (H_0_Hz**2)
# OmegaGW_data = np.log10(OmegaGW_data)

violin_data = np.load('./violins_data.npz')
OmegaGW_data = violin_data['OmegaGW_data']
freqs = violin_data['freqs']

# v1 = ax.violinplot(list(OmegaGW_data), np.log10(freqs), widths=0.05)

frequencies = np.load(f'{datadir}/freqs.npy')
left_node = -9.5
right_node = -7
y_max = -1.
y_min = -4.
s = jnp.linspace(0, 1, 15)  # First rescaled internal momentum
t = jnp.logspace(-5,5, 200)  # Second rescaled internal momentum
t_expanded = jnp.expand_dims(t, axis=-1)
## Repeat t along the new axis to match the shape (100, 1000)
t = jnp.repeat(t_expanded, len(frequencies), axis=-1)
gwb_calculator = OmegaGWjax(s=s,t=t,f=frequencies,norm="RD",jit=True,to_numpy=False)
p_arr = jnp.logspace(left_node+0.001, right_node-0.001, 50)

def interpolate(nodes,vals,x):
    spl = CubicSpline(nodes,vals,check=False)
    res = jnp.power(10,spl(x))
    res = jnp.where(x<left_node, 0., res)
    res = jnp.where(x>right_node, 0., res)
    return res

def get_pz_gwb(nodes,vals):
    pf = lambda k: interpolate(nodes=nodes,vals=vals,x=jnp.log10(k))
    pz = pf(p_arr)
    omegagw = gwb_calculator(pf,frequencies)
    return (pz, omegagw)

# run data
run_data = np.load(f'samples_{num_nodes}.npz')
samples = run_data['samples']
logl = run_data['logl']
logwt = run_data['logwt']

rstate = np.random.RandomState(42)
resampled_samples, resampled_logl = resample_equal(samples, logl, logwt, rstate)
thinning = len(resampled_samples) // 256
ys = resampled_samples[:,free_nodes:][::thinning]
ys = resampled_samples[:,free_nodes:][::thinning]
ys = jnp.array(ys)
# if free_nodes>1:
xs = resampled_samples[:,:free_nodes][::thinning]
xs = jnp.pad(xs, ((0,0),(1,1)), 'constant', constant_values=((0,0),(left_node, right_node)))
# else:
#     xs = jnp.array([[left_node,right_node] for _ in range(len(ys))])

pz_amps,gwb_amps = split_vmap(get_pz_gwb,(xs,ys),batch_size=32)

# print(pz_amps[0])
# print(gwb_amps[0])

log_pz_amps = np.log10(pz_amps)
log_gwb_amps = np.log10(gwb_amps)

fig, (ax1,ax2) = plot_functional_posterior(vals=[pz_amps,log_gwb_amps],
                                    k_arr=[p_arr,np.log10(frequencies)],intervals=[95.,68.],
                                    aspect_ratio=(8,6))
ax2.set(xscale='linear',yscale='linear',xlabel=r'$\log_{10} f\,{\rm [Hz]}$')
ax1.set(xscale='log',yscale='log',xlabel=r'$f\,{\rm [Hz]}$')
k_mpc_f_hz = 2*np.pi * 1.03 * 10**14
for x in [ax1]:#,ax2]:
    secax = x.secondary_xaxis('top', functions=(lambda x: x * k_mpc_f_hz, lambda x: x / k_mpc_f_hz))
    secax.set_xlabel(r"$k\,{\rm [Mpc^{-1}]}$",labelpad=10) 
# ax[0].set_xlim(min(p_arr),max(p_arr))
from matplotlib import colors
v1 = ax2.violinplot(list(OmegaGW_data), positions=np.log10(freqs), widths=0.05)
for pc in v1['bodies']:
    pc.set_facecolor(('#E03424', 0.25))
    # pc.set_facecolor(('blue',0.25))
    pc.set_edgecolor(('#E03424',0.75))
    pc.set_linestyle('solid')
    # pc.set_alpha(0.8)
    pc.set_linewidth(1.5)
v1['cmins'].set_color(('#E03424',0.5))
v1['cmaxes'].set_color(('#E03424',0.5))
v1['cbars'].set_color(('#E03424',0.5))
ax2.set_ylim(-12,-4)
ax2.set_xlim(-8.8, -7.68)# plt.show()
    # for x in ax:
    #     x.set(xscale='log',yscale='linear',xlabel=r'$f\,{\rm [Hz]}$')
plt.savefig(f'NG15_recon_{num_nodes}_nodes.pdf', bbox_inches='tight')


# Results:

# N = 2
# log Z: -53.6508

# N = 3 
# log Z: -54.5899

# N = 4
# log Z: -51.0853

# N = 5
# log Z: -51.3488
