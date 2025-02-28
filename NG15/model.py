from ptarcade.models_utils import prior
import os
import numpy as np
import ptarcade.models_utils as aux
import jax.numpy as jnp
from GWB_Jax import OmegaGWjax
from interpax import CubicSpline
from jax import jit

num_nodes = 3
h = 0.67
smbhb = False
name = 'sigw'

parameters = {'y%i' % i: prior("Uniform", -10, -2) for i in range(num_nodes)} # 

# print(parameters.keys())


nodes = jnp.linspace(-9,-7,num_nodes)

@jit
def spline(nodes, values, x):
    """
    Spline for frequency spectrum of Pz
    """
    spl = CubicSpline(nodes, values, check=False)
    res = 10**(spl(jnp.log10(x)))
    res = jnp.where(x < 10**nodes[0], 0.0, res)
    res = jnp.where(x > 10**nodes[-1], 0.0, res)
    return res

s = jnp.linspace(0, 1, 10)  # First rescaled internal momentum
t = jnp.logspace(-3,3, 200)  # Second rescaled internal momentum
# ## Expand t to add a new axis
t = jnp.expand_dims(t, axis=-1)
# print(t.shape)
# print(s.shape)

gw_calc = OmegaGWjax(s, t, f=None, kernel="RD", upsample=False,to_numpy=True)

def spectrum(f, y0,y1,y2):
    """
    Spectrum for the frequency
    """
    values = jnp.array([y0,y1,y2])
    pz_spline = lambda x: spline(nodes, values, x)
    f = jnp.array([f])
    gw = gw_calc(pz_spline, f)
    return h**2 * gw

# f = jnp.geomspace(1e-9, 1e-7, 10)

# for x in f:
#     res = spectrum(x, 1, 1, 1)
#     print(res.dtype)
#     print(res.shape)
#     print(res)