import numpy as np
import h5py
from libraries import sdintegral_numba as sd
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import h5py


data = np.load('./spectra_0p5_interp.npz')
frequencies = data['frequencies']

# 1. Specify the grid of w you want to cover
w_min = 0.0001
w_max = 0.9999
N_w = 1800
w_grid = np.linspace(w_min, w_max, N_w)  # e.g. np.linspace(–2, +2, 1001)

# 2. Open an HDF5 file to write
with h5py.File("precomputed_kernels.h5", "w") as f:
    # Create a dataset for the w values themselves
    f.create_dataset("w_vals", data=w_grid, dtype="f8")


    nd,ns1,ns2, darray,d1array,d2array, s1array,s2array = sd.arrays_w(0.5,frequencies,nd=150)


    # Create two 2D datasets: shape (N_w, N_pts)
    # where N_pts = len(d1array) = len(d2array) = …
    N_pts_1 = len(d1array)
    N_pts_2 = len(d2array)
    dset1 = f.create_dataset("kernel1", shape=(len(w_grid), N_pts_1), dtype="f8")
    dset2 = f.create_dataset("kernel2", shape=(len(w_grid), N_pts_2), dtype="f8")

    # Loop once and fill
    for idx, w in enumerate(tqdm(w_grid)):
        nd,ns1,ns2, darray,d1array,d2array, s1array,s2array = sd.arrays_w(w,frequencies,nd=150)
        b = sd.beta(w)
        k1 = sd.kernel1_w(d1array, s1array, b)
        k2 = sd.kernel2_w(d2array, s2array, b)
        dset1[idx, :] = k1
        dset2[idx, :] = k2



# tolerance in 

# def compute_w(w,log10_f_rh,nodes,vals,frequencies,use_mp=False,nd=150,fref=1.):
#     nd,ns1,ns2, darray,d1array,d2array, s1array,s2array = sd.arrays_w(w,frequencies,nd=nd)
#     b = sd.beta(w)
#     kernel1, kernel2 = get_kernels_from_file(w)
#     # kernel1 = sd.kernel1_w(d1array, s1array, b)
#     # kernel2 = sd.kernel2_w(d2array, s2array, b)
#     nk = len(frequencies)
#     Integral = np.empty_like(frequencies)
#     Integral = gw.compute_w_k_array(nodes = nodes, vals = vals, nk = nk,komega = frequencies, 
#                                             kernel1 = kernel1, kernel2 = kernel2, d1array=d1array,
#                                             s1array=s1array, d2array=d2array, s2array=s2array,
#                                             darray=darray, nd = nd, ns1 = ns1, ns2 = ns2)
#     f_rh = 10**log10_f_rh
#     two_b = 2*b
#     norm = rd_norm * (frequencies)**(-2*b) *  (f_rh/fref)**(two_b)   
#     OmegaGW = norm * Integral
#     return OmegaGW