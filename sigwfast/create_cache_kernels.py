import numpy as np
import h5py
try:
    from libraries import sdintegral_numba as sd
except:
    from libraries import sdintegral as sd
from tqdm import tqdm
import numpy as np


data = np.load('./spectra_0p5_interp.npz')
frequencies = data['frequencies']
# set your own frequencies if needed

# 1. Specify the grid of w you want to cover
w_min = 0.0001
w_max = 0.9999
N_w = 2000
w_grid = np.linspace(w_min, w_max, N_w)  # e.g. np.linspace(–2, +2, 1001)

# 2. Open an HDF5 file to write
with h5py.File("precomputed_kernels.h5", "w") as f:
    # Create a dataset for the w values themselves
    f.create_dataset("w_vals", data=w_grid, dtype="f8")


    nd,ns1,ns2, darray,d1array,d2array, s1array,s2array = sd.arrays_w(0.5,frequencies,nd=150) # just a dummy w


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
