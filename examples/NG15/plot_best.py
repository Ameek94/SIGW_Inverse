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
from matplotlib import colors

# # Command line argument for number of nodes
# if len(sys.argv) != 2:
#     print("Usage: python script.py <num_nodes>")
#     sys.exit(1)

# Load data files
datadir = './NG15_Ceffyl/30f_fs{hd}_ceffyl/'

violin_data = np.load('./violins_data.npz')
OmegaGW_data = violin_data['OmegaGW_data']
freqs = violin_data['freqs']

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

# best fit

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(15,6),constrained_layout=True)
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

# run data
n_nodes = [3,4,5]

for i,n in enumerate(n_nodes):
    num_nodes = n
    free_nodes = num_nodes - 2
    print(f"Best fit for {n} nodes")
    data = np.load(f'best_fit_{n}.npz')
    params = data['best_params']
    best_fit = data['best_fit']
    print(f"Best fit: {best_fit}, Best params: {params}")
    xs = params[:free_nodes]
    ys = params[free_nodes:]
    xs = jnp.pad(xs, (1,1), 'constant', constant_values=(left_node,right_node))
    ys = jnp.array(ys)
    pz_amps,gwb_amps = get_pz_gwb(xs,ys)
    ax1.loglog(p_arr, pz_amps, label=rf'$N={n}$')
    ax1.scatter(10**xs, 10**ys)
    ax2.plot(np.log10(frequencies), np.log10(gwb_amps), label=rf'$N={n}$',color='C%i'%i)
ax1.legend()
ax1.set_xlabel(r'$f\,{\rm [Hz]}$')
ax1.set_ylabel(r'$P_{\zeta}$')
ax2.set_xlabel(r'$\log_{10} f\,{\rm [Hz]}$')
ax2.set_ylabel(r'$\log_{10} \Omega_{\rm GW}$')

k_mpc_f_hz = 2*np.pi * 1.03 * 10**14
for x in [ax1]:#,ax2]:
    secax = x.secondary_xaxis('top', functions=(lambda x: x * k_mpc_f_hz, lambda x: x / k_mpc_f_hz))
    secax.set_xlabel(r"$k\,{\rm [Mpc^{-1}]}$",labelpad=10) 
plt.savefig('best_fit.pdf',bbox_inches='tight')
plt.show()