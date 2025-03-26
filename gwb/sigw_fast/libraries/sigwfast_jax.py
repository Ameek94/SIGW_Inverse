import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import libraries.sdintegral_numba as sd
from interpax import CubicSpline
from collections import OrderedDict
import matplotlib.pyplot as plt

OMEGA_R = 4.2 * 10**(-5)
CG = 0.39
rd_norm = CG * OMEGA_R 

# Global cache for storing kernels keyed by rounded w
kernel_cache = OrderedDict()

cache_counter = 0

def get_kernels(w, d1array, s1array, d2array, s2array, tolerance=3):
    global cache_counter

    # Round w to the desired tolerance (number of decimals)
    key = round(w, tolerance)
    # If already cached, update the order and return
    if key in kernel_cache:
        cache_counter += 1
        kernel_cache.move_to_end(key)
        return kernel_cache[key]
    
    # Otherwise compute the kernels
    b = sd.beta(w)
    kernel1 = sd.kernel1_w(d1array, s1array, b)
    kernel2 = sd.kernel2_w(d2array, s2array, b)
    
    # If cache size is 4, remove the least recently used entry
    if len(kernel_cache) >= 100:
        kernel_cache.popitem(last=False)
    
    # Store and return the result
    kernel_cache[key] = (jnp.array(kernel1), jnp.array(kernel2))
    return kernel_cache[key]


def interpolate(nodes, vals, x, left_node,right_node):
    # Create a cubic spline interpolation of log10(Pζ) and then convert back to linear scale.
    spl = CubicSpline(nodes, vals, check=False)
    res = jnp.power(10, spl(x))
    res = jnp.where(x < left_node, 0, res)
    res = jnp.where(x > right_node, 0, res)
    return res

def compute_w(w,log10_f_rh,nodes,vals,frequencies,use_mp=False,nd=100,fref=1.):

    Pz = jax.jit(lambda f: interpolate(nodes=nodes,vals=vals,x=jnp.log10(f),left_node=nodes[0],right_node=nodes[-1]))

    nd,ns1,ns2, darray,d1array,d2array, s1array,s2array = sd.arrays_w(w,frequencies,nd=nd)
    b = sd.beta(w)
    kernel1, kernel2 = get_kernels(w, d1array, s1array, d2array, s2array)
    # kernel1 = sd.kernel1_w(d1array, s1array, b)
    # kernel2 = sd.kernel2_w(d2array, s2array, b)

    # print array shapes for debugging
    # print(f"nd = {nd}, ns = {ns1}, {ns2}, darray shape = {darray.shape}")
    # print(f"nd x ns = {nd*ns1}, {nd*ns2}")
    # print(f"d1array shape = {d1array.shape}, s1array shape = {s1array.shape}")
    # print(f"d2array shape = {d2array.shape}, s2array shape = {s2array.shape}")
    # print(f"kernel1 shape = {kernel1.shape}, kernel2 shape = {kernel2.shape}")

    # convert to jax arrays
    darray = jnp.array(darray)
    d1array = jnp.array(d1array)
    d2array = jnp.array(d2array)
    s1array = jnp.array(s1array)
    s2array = jnp.array(s2array)
    s1 = jnp.array(s1array.reshape((nd, ns1)))
    s2 = jnp.array(s2array.reshape((nd, ns2)))
    K1 = kernel1.reshape((nd, ns1))
    K2 = kernel2.reshape((nd, ns2))

    @jax.jit
    def compute_single_f(f):
        psq1 = Pz(f/2 * (s1array + d1array)) * Pz(f/2 * (s1array - d1array))  #psquared_jax(d1array, s1array, Pz, f).reshape((nd, ns1))
        psq1 = psq1.reshape((nd, ns1))
        psq2 = Pz(f/2 * (s2array + d2array)) * Pz(f/2 * (s2array - d2array))  #psquared_jax(d2array, s2array, Pz, f).reshape((nd, ns2))
        psq2 = psq2.reshape((nd, ns2))
        Int_ds1 = K1 * psq1
        Int_ds2 = K2 * psq2
        int_s1 = jnp.trapezoid(Int_ds1,x=s1,axis=1)
        int_s2 = jnp.trapezoid(Int_ds2,x=s2,axis=1)
        int_d = int_s1+int_s2
        res = jnp.trapezoid(int_d,x=darray)
        return res

    Integral = jax.vmap(compute_single_f)(frequencies)
    f_rh = 10**log10_f_rh
    two_b = 2*b
    norm = rd_norm * (frequencies)**(-2*b) *  (f_rh/fref)**(two_b)   
    OmegaGW = norm * Integral
    return OmegaGW


# The intarray1D function integrates using the trapezoidal rule.
@jax.jit
def intarray1D_jax(f, dx):
    # Compute the differences between adjacent elements in dx.
    dx_diff = dx[1:] - dx[:-1]
    # Use vectorized computation for the trapezoidal sum.
    return 0.5 * jnp.sum((f[:-1] + f[1:]) * dx_diff)

def split_vmap(func,input_arrays,batch_size=8):
    """
    Utility to split vmap over a function taking multiple arrays as input into multiple chunks, useful for reducing memory usage.
    """
    num_inputs = input_arrays[0].shape[0]
    num_batches = (num_inputs + batch_size - 1 ) // batch_size
    batch_idxs = [jnp.arange( i*batch_size, min( (i+1)*batch_size,num_inputs  )) for i in range(num_batches)]
    res = [jax.vmap(func)(*tuple([arr[idx] for arr in input_arrays])) for idx in batch_idxs]
    nres = len(res[0])
    # now combine results across batches and function outputs to return a tuple (num_outputs, num_inputs, ...)
    results = tuple( jnp.concatenate([x[i] for x in res]) for i in range(nres))
    return results

def bpl(p, pstar=5e-4, n1=2, n2=-1, sigma=2):
    nir = n1
    pl1 = (p / pstar) ** nir
    nuv = (n2 - n1) / sigma
    pl2 = (1 + (p / pstar) ** sigma) ** nuv
    return 1e-2 * pl1 * pl2

def main():
    w = 0.8
    log10_f_rh = -5.
    nodes = jnp.array([-5., -4.383505, -3.76701, -3.150515, -2.53402, -1.917525, -1.30103 ])
    vals = jnp.log10(bpl(10**nodes)) #jnp.array([-4.05813325,-3.90081585, -2.87794112, -2.36267636, -2.74488783, -3.45095726, -5.41698371])
    kmin, kmax = 5e-5, 1e-2
    frequencies = jnp.logspace(jnp.log10(kmin), jnp.log10(kmax), 50)
    omegagw = compute_w(w, log10_f_rh, nodes, vals, frequencies, use_mp=False, nd=150)
    data = np.load('../spectra_0p8_interp.npz')
    frequencies = jnp.array(data['frequencies'])
    gwb_model = 'bpl'
    Omegas = data[f'gw_{gwb_model}'] 
    plt.loglog(frequencies, Omegas, label='non jax')
    plt.loglog(frequencies, omegagw, label='jax')
    plt.legend()
    plt.xlabel('frequencies')
    plt.ylabel('OmegaGW')
    plt.show()

if __name__ == "__main__":
    main()