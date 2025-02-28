from GWB_Jax import OmegaGWjax
import jax.numpy as jnp
from jax import jit
import matplotlib.pyplot as plt



# set up the power spectrum and omgw
psize = 50
ksize = 60
p_arr = jnp.geomspace(2.5e-5,2e-2,psize)
k_arr = jnp.geomspace(5e-5,1e-2,ksize)

# f = k_arr #jnp.geomspace(5e-5, 1e-2, ksize)  # The frequencies to calculate Omega_GW
s = jnp.linspace(0, 1, 10)  # First rescaled internal momentum
t = jnp.logspace(-3,3, 100)  # Second rescaled internal momentum

# ## Expand t to add a new axis
t_expanded = jnp.expand_dims(t, axis=-1)
# print(_)
# ## Repeat t along the new axis to match the shape (100, 1000)
# t = jnp.repeat(t_expanded, len(f), axis=-1)

@jit
def pz(p,pstar=5e-4,n1=3,n2=-2,sigma=2):
    nir = n1
    pl1 = (p/pstar)**nir
    nuv = (n2 - n1)/sigma
    pl2 = (1+(p/pstar)**sigma)**nuv
    # osc = (1 + 16.4*jnp.cos(1.4*jnp.log(p/1.))**2)
    return pl1 * pl2 #*osc

t = t_expanded
print(t.shape)
gwb_calculator =  OmegaGWjax(s, t, f=None, kernel="RD", upsample=False,to_numpy=True)


pz_amp = pz(p_arr)

f = jnp.array([k_arr[5]])
print(f.shape)

gwb_amp = gwb_calculator(pz,f)

print(gwb_amp)

# print(gwb_amp)

# fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,3.5))
# ax1.scatter(p_arr,pz_amp,s=2)
# ax1.set_ylabel(r'$P_{\zeta}(k)$')
# ax2.set_ylabel(r'$\Omega_{\mathrm{GW}}(k)$')
# ax2.scatter(f,gwb_amp,s=2)
# for ax in [ax1,ax2]:
#     ax.set(yscale='log',xscale='log',xlabel=r'$k$')
# fig.tight_layout();

