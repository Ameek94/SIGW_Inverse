import sys,os
cpus_per_task= int(sys.argv[2])
import numpy as np
import matplotlib.pyplot as plt
from ceffyl import Ceffyl
import natpy as nat
from jax import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
from omega_gw_jax import OmegaGWjax
from enterprise.signals import parameter
import time
from mpi4py.futures import MPIPoolExecutor

def prior1D(cube,free_nodes,left_node,right_node,y_min,y_max):
    # Order and transform nodes to be in the correct range, from Polychord SortedUniformPrior
    # params = np.atleast_2d(cube.copy())
    params = cube.copy()
    ys = params[free_nodes:]
    ys = ys*(y_max - y_min) + y_min
    # x = params[:free_nodes]
    # N = free_nodes
    # t = np.zeros(N)
    # t[N-1] = x[N-1]**(1./N)
    # for n in range(N-2, -1, -1):
    #     t[n] = x[n]**(1./(n+1)) * t[n+1]
    # xs = t*(right_node - left_node) + left_node
    # return np.concatenate([xs,ys]) # array
    # else:
    return ys

from interpax import CubicSpline
from functools import partial

def interpolate(nodes, vals, x,left_node,right_node):
    # Create a cubic spline interpolation of log10(Pζ) and then convert back to linear scale.
    # spl = CubicSpline(nodes, vals, check=False)
    # Testing linear interpolation
    res = jnp.power(10, jnp.interp(x, nodes, vals))
    res = jnp.where(x < left_node, 0, res)
    res = jnp.where(x > right_node, 0, res)
    return res


def get_gwb(gwb_calculator,frequencies,nodes,vals,left_node,right_node):
    pf = lambda k: interpolate(nodes=nodes,vals=vals,x=jnp.log10(k)
                               ,left_node=left_node,right_node=right_node)
    omegagw = gwb_calculator(pf,frequencies)
    return omegagw

h = 0.672
H_0  = h * 100 * nat.convert(nat.km * nat.s**-1 * nat.Mpc**-1, nat.GeV) # Hubble constant (GeV)
H_0_Hz  = H_0 * nat.convert(nat.GeV, nat.Hz)
# convert OmegaGW to rho
def psd(f,Tspan,get_gwb_func,free_nodes,left_node,right_node,**kwargs):
    # xs = jnp.array([kwargs['x%i' % i].item() for i in range(free_nodes)])
    # xs = jnp.pad(xs, (1,1), 'constant', constant_values=(left_node,right_node))
    # xs = jnp.array(xs)
    xs = jnp.array([left_node,right_node])
    ys = jnp.array([kwargs['y%i' % i].item() for i in range(free_nodes+2)])
    # print(f'xs ys shapes {xs.shape},{ys.shape}')
    # print(f'xs = {xs}, ys = {ys}')
    gwb = get_gwb_func(nodes=xs,vals=ys)
    gwb = jnp.reshape(gwb, f.shape)
    # print(f'gwb_shape = {gwb.shape}')
    # print(f'frequncies shape = {f.shape}')
    psd = gwb*H_0_Hz**2 / (8 * np.pi**4 * f**5)
    # print(f'psd shape = {psd.shape}')
    return psd / Tspan

def log_likelihood(params,pta=None):
    # print(params)
    # print(params.shape)
    res = pta.ln_likelihood(params)
    # print(res.shape)
    # print(res)
    res = np.where(np.isnan(res),-1e4, res)
    res = np.where(res<-1e4,-1e4, res)
    return res

def main():
    # the data file
    data_dir = './NG15_Ceffyl/30f_fs{hd}_ceffyl/'
    freqs = np.load(f'{data_dir}/freqs.npy')
    frequencies = freqs[:14]
    # set up the power spectrum and GWB model
    num_nodes = int(sys.argv[1])
    print(f"Running inference with number of nodes: {num_nodes}, free nodes: {num_nodes-2}")
    free_nodes = num_nodes - 2
    left_node = np.log10(min(frequencies)/5) #-9.
    right_node = np.log10(max(frequencies)*5) #-7.5
    print(f"left_node = {left_node}, right_node = {right_node}")
    y_max = 0.
    y_min = -8
    s = jnp.linspace(0, 1, 15)  # First rescaled internal momentum
    t = jnp.logspace(-5,5, 500)  # Second rescaled internal momentum
    t_expanded = jnp.expand_dims(t, axis=-1)
    ## Repeat t along the new axis to match the shape (100, 1000)
    t = jnp.repeat(t_expanded, len(frequencies), axis=-1)
    gwb_calculator = OmegaGWjax(s=s,t=t,f=frequencies,norm="RD",jit=True,to_numpy=True)
    get_gwb_func = jax.jit(partial(get_gwb,
                                   gwb_calculator=gwb_calculator,
                                   frequencies=frequencies,
                                   left_node=left_node,
                                   right_node=right_node))
    # set up the PTA
    pta = Ceffyl.ceffyl(datadir=data_dir)
    print("Initialized PTA")
    print("Tspan(s) = ", pta.Tspan)
    # set up params for pta
    params = []
    # add free nodes to params
    for i in range(free_nodes):
        params.append(parameter.Uniform(left_node, right_node)(f'x{i}'))
    # add y parameters to params
    for i in range(num_nodes):
        params.append(parameter.Uniform(y_min, y_max)(f'y{i}'))
    print("Initialized params",params)
    # set up psd
    psd_use = partial(psd
                      ,get_gwb_func=get_gwb_func
                      ,free_nodes=free_nodes
                      ,left_node=left_node
                      ,right_node=right_node)
    # add signals to pta
    gw = Ceffyl.signal(psd=psd_use,N_freqs=len(frequencies),name='gw',params=params)
    pta.add_signals([gw])
    # set up the loglike and prior for sampler
    loglike = partial(log_likelihood,pta=pta)
    prior = partial(prior1D,
                    free_nodes=free_nodes,
                    left_node=left_node,
                    right_node=right_node,
                    y_min=y_min,
                    y_max=y_max)

    # set up the sampler
    from nautilus import Sampler
    ndim = free_nodes + num_nodes
    sampler = Sampler(prior, loglike, ndim, pass_dict=False,vectorized=False
                      ,filepath=f'Ceffyl_samples_{num_nodes}_linear.h5'
                      ,pool=None )
    # print("Starting Sampling")
    start = time.time()
    sampler.run(verbose=True,f_live=0.005,n_like_max=int(1e6))
    end = time.time()
    print('Sampling complete, time taken: {:.4f} s'.format(end-start))
    print('log Z: {:.4f}'.format(sampler.log_z))
    samples, logl, logwt = sampler.posterior()
    np.savez(f'samples_{num_nodes}_linear.npz',samples=samples,logl=logl,logwt=logwt,logz=sampler.log_z)

if __name__=='__main__':
    main()