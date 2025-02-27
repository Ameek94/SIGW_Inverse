import os
import sys
os.environ['XLA_FLAGS'] = f"--xla_force_host_platform_device_count={int(sys.argv[1])}"
sys.path.append('../')
from jax import vmap, jit, grad, random, jacfwd
import jax.numpy as jnp
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
import warnings
from jax import config
config.update("jax_enable_x64", True)
import numpyro
from interpolation.omega_gw_grid import OmegaGWGrid
from interpolation.omega_gw_jax import OmegaGWjax
from interpolation.model import Fixed_Nodes_Model
from interpolation.run import sampler
font = {'size'   : 16, 'family':'serif'}
axislabelfontsize='large'
matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=True) # if using latex in plots
matplotlib.rc('legend', fontsize=16)

# set up the power spectrum and omgw
psize = 50
ksize = 50
p_arr = jnp.geomspace(2e-5,2.5e-2,psize)
k_arr = jnp.geomspace(5e-5,1e-2,ksize)

f = k_arr #jnp.geomspace(5e-5, 1e-2, ksize)  # The frequencies to calculate Omega_GW
s = jnp.linspace(0, 1, 10)  # First rescaled internal momentum
t = jnp.logspace(-4,4, 100)  # Second rescaled internal momentum

## Expand t to add a new axis
t_expanded = jnp.expand_dims(t, axis=-1)
## Repeat t along the new axis to match the shape (100, 1000)
t = jnp.repeat(t_expanded, len(f), axis=-1)

@jit
def pz(p,pstar=5e-4,n1=3,n2=-2,sigma=2):
    nir = n1
    pl1 = (p/pstar)**nir
    nuv = (n2 - n1)/sigma
    pl2 = (1+(p/pstar)**sigma)**nuv
    osc = (1 + 16.4*jnp.cos(1.4*jnp.log(p/1.))**2)
    return 1e-2*pl1 * pl2 *osc

gwb_calculator =  OmegaGWjax(s, t, f=f, kernel="RD", upsample=False, norm="RD")


pz_amp = pz(p_arr)
gwb_amp = gwb_calculator(pz,f)

print(gwb_amp.shape)
pz_amp = pz(p_arr)
pstar=1e-3
sigma=0.1
amp=1e-6
floor=1e-3
gwb_amp = amp *(floor+ jnp.exp(-0.5*((jnp.log(f/pstar)/sigma)**2)))
 #gwb_calculator(pz,f)

print(gwb_amp.shape)
kstar = 1e-3
omks_sigma = gwb_amp*( 0.08*(np.log(k_arr/kstar))**2 + 0.05) # 2% error at kstar + more towards edges
gwb_cov = jnp.diag(omks_sigma**2)

#run AIC and plotting
kmin, kmax = min(p_arr), max(p_arr)
gwb_method_kwargs = {'s': s, 't': t, 'norm': 'RD', 'kernel': 'RD', 'upsample': False}
mc_sample_kwargs = { 'num_warmup': 1024,'num_samples': 2048,'verbose': True }
osc = sampler(pz_kmin=kmin,pz_kmax=kmax,gwb_karr=k_arr,
                 gwb_means=gwb_amp,gwb_cov=gwb_cov,
                 gwb_method='jax',gwb_method_kwargs=gwb_method_kwargs,
                y_low=-8,y_high=0.,interpolation_method='CubicSpline',min_nodes=7,max_nodes=14
                ,aic=True,mc_sample=True,mc_sample_kwargs=mc_sample_kwargs,
                bayes_factor=False)
results = osc.run()

from interpolation.model import spline_predict

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15,4))
# ax1.loglog(p_arr,pz_amp,color='k',lw=1.5)
ax2.loglog(f,gwb_amp,label=r'Truth',color='k',lw=1.5)
n_nodes = np.arange(7,13)
for n in n_nodes:
    res = results[str(n)]
    nodes = res['node_locations']
    best_params = res['best_params']
    def pz_bf(x):
        val = spline_predict(x_train=nodes,y_train=best_params,x_pred=x) #
        val = jnp.where(jnp.log(x)<jnp.log(kmin),0.,val)
        val = jnp.where(jnp.log(x)>jnp.log(kmax),0.,val)
        return val
    pz = pz_bf(p_arr)
    gwb_bf = gwb_calculator(pz_bf,k_arr) #jnp.einsum("i,j,kij->k",pz_bf,pz_bf,gwb_calculator.omkij)
    ax1.loglog(p_arr,pz,label=f'$N = {n}$',alpha=0.75)
    ax1.scatter(10**(nodes),10**(best_params),s=10)
    ax2.loglog(k_arr,gwb_bf,label=f'$N = {n}$',alpha=0.5)
# for node in nodes:
#     ax1.axvline(10**(node),color='gray',ls='--',lw=0.5)
for x in [ax1,ax2]:
    x.set(xlabel=r'$k$',yscale='log',xscale='log')
ax1.set_ylabel(r'$P_{\zeta}(k)$')
ax2.fill_between(k_arr,gwb_amp+1.96*omks_sigma,gwb_amp-1.96*omks_sigma,alpha=0.2,color='k')
ax2.set_ylabel(r'$\Omega_{\mathrm{GW}}(k)$')
# ax1.legend(ncol=2)
ax2.legend(ncol=3)
ax1.axvspan(min(k_arr),max(k_arr),color='gray',alpha=0.2)
# ax2.set_ylim(1e-5,1.)
plt.savefig('peaked_gwb_aic.pdf',bbox_inches='tight')

#run Bayes factor and plotting
kmin, kmax = min(p_arr), max(p_arr)
gwb_method_kwargs = {'s': s, 't': t, 'norm': 'RD', 'kernel': 'RD', 'upsample': False}
nested_sampler_kwargs = {'max_samples': 1e6,'parameter_estimation': True, 'difficult_model': False, 'verbose': False}
peaked = sampler(pz_kmin=kmin,pz_kmax=kmax,gwb_karr=k_arr,
                 gwb_means=gwb_amp,gwb_cov=gwb_cov,
                 gwb_method='jax',gwb_method_kwargs=gwb_method_kwargs,
                y_low=-8.,y_high=0.,interpolation_method='CubicSpline',min_nodes=7,max_nodes=12,aic=False,
                bayes_factor=True, nested_sampler_kwargs=nested_sampler_kwargs)
results = peaked.run()