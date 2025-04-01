import sys
sys.path.append('/scratch/s.ameek.malhotra/SIGW_Inverse/NG15')
from ptarcade.models_utils import prior
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from omega_gw_jax import OmegaGWjax
from jax import jit
import numpy as np
from HD_config import gwb_components

num_nodes = 2
h = 0.67
smbhb = False
name = 'sigw_hd_'+str(num_nodes)+'_nodes'

# print(parameters.keys())

left_node = -9.
right_node = -7.
nodes = jnp.linspace(left_node,right_node,num_nodes)
y_min = -8.
y_max = 0.
parameters = {'x%i' % i: prior("Uniform", left_node, right_node) for i in range(num_nodes-2)}
parameters.update({'y%i' % i: prior("Uniform", y_min, y_max) for i in range(num_nodes)})

# @jit
def interpolate(nodes, vals, x):
    # Create a cubic spline interpolation of log10(Pζ) and then convert back to linear scale.
    # spl = CubicSpline(nodes, vals, check=False)
    # Testing linear interpolation
    # spl = lambda x: 
    res = jnp.power(10, jnp.interp(jnp.log10(x), nodes, vals))
    res = jnp.where(x < -9., 0, res)
    res = jnp.where(x > -7., 0, res)
    return res

data_dir = './NG15_Ceffyl/30f_fs{hd}_ceffyl/'
freqs = np.load(f'{data_dir}/freqs.npy')
frequencies = freqs[:gwb_components]
s = jnp.linspace(0, 1, 15)  # First rescaled internal momentum
t = jnp.logspace(-5,5, 200)  # Second rescaled internal momentum
t_expanded = jnp.expand_dims(t, axis=-1)
## Repeat t along the new axis to match the shape (100, 1000)
t = jnp.repeat(t_expanded, len(frequencies), axis=-1)

gwb_calculator = OmegaGWjax(s=s, t=t, f=frequencies, kernel="RD", upsample=False,to_numpy=True,jit=False)

def interpolate(nodes, vals, x):
    # Create a cubic spline interpolation of log10(Pζ) and then convert back to linear scale.
    # spl = CubicSpline(nodes, vals, check=False)
    # Testing linear interpolation
    res = jnp.power(10, jnp.interp(x, nodes, vals))
    res = jnp.where(x < left_node, 0, res)
    res = jnp.where(x > right_node, 0, res)
    return res

def get_gwb(frequencies,nodes,vals):
    pf = lambda k: interpolate(nodes=nodes,vals=vals,x=jnp.log10(k)
                                )
    omegagw = gwb_calculator(pf,frequencies)
    return omegagw

gwb_func = jax.jit(get_gwb)

def spectrum(f,y0,y1): # x0,
    """
    Spectrum for the frequency
    """
    # if isinstance(x0, np.ndarray):
    #     # print(f"len x0 = {len(x0)}")
    #     # print(f"shape x0 = {x0.shape}")
    #     # print(f'x0 = {x0}')
    #     xs = jnp.array([left_node,x0.item(),right_node])
    # else:
    #     xs = jnp.array([left_node,x0,right_node])
    # print(xs)
    xs = nodes
    ys = jnp.array([y0,y1])
    # print(xs)
    xs = jnp.array(xs).flatten()
    ys = jnp.array(ys).flatten()
    # print(f'xs ys shapes {xs.shape},{ys.shape}')
    # print(f'xs = {xs}, ys = {ys}')
    shape = f.shape
    if f.ndim == 2:
        f = f.flatten()  
    # print(f.shape)
    # nodes = p
    # values = jnp.asarray([y0,y1]).flatten()
    # print(nodes.shape)
    # print(values.shape)
    # pz_spline = lambda x: interpolate(nodes=xs, vals=ys, x=x)
    f = jnp.asarray(f)
    # print(f.shape)
    gw = get_gwb(frequencies=f,nodes=xs,vals=ys) #gw_calc(pz_spline, f)
    # print(gw.shape)
    gw = gw.reshape(shape)
    return h**2 * gw