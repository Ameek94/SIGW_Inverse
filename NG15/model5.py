import sys
sys.path.append('/scratch/s.ameek.malhotra/SIGW_Inverse/NG15')
from ptarcade.models_utils import prior
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from GWB_Jax import OmegaGWjax
from interpax import CubicSpline
from jax import jit
import numpy as np

num_nodes = 5
h = 0.67
smbhb = False
name = 'sigw_'+str(num_nodes)+'_nodes'

parameters = {'y%i' % i: prior("Uniform", -10, -2) for i in range(num_nodes)} # 

# print(parameters.keys())

nodes = jnp.linspace(-9,-7,num_nodes)

# @jit
def spline(nodes, values, x):
    """
    Spline for frequency spectrum of Pz
    """
    spl = CubicSpline(nodes, values, check=False)
    res = 10**(spl(jnp.log10(x)))
    res = jnp.where(x < 10**nodes[0], 0.0, res)
    res = jnp.where(x > 10**nodes[-1], 0.0, res)
    return res

gw_calc = OmegaGWjax(s=nodes, t=nodes, f=None, kernel="RD", upsample=False,to_numpy=True,jit=False)


def spectrum(f, y0,y1,y2,y3,y4):
    """
    Spectrum for the frequency
    """
    # print(f.shape)
    # print(f.ndim)
    shape = f.shape
    if f.ndim == 2:
        f = f.flatten()  
    # print(f.shape)
    # print(t.shape)
    values = jnp.asarray([y0,y1,y2,y3,y4]).flatten()
    # print(nodes.shape)
    # print(values.shape)
    pz_spline = lambda x: spline(nodes, values, x)
    f = jnp.asarray(f)
    # print(f.shape)
    gw = gw_calc(pz_spline, f)
    # print(gw.shape)
    gw = gw.reshape(shape)
    return h**2 * gw