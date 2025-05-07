from functools import partial
import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
import sys
sys.path.append('../')
import time
from sigw_fast.RD import compute
import numpy as np
import matplotlib.pyplot as plt
# try:
#     from sigw_fast.libraries import sdintegral_numba as sd
# except ImportError:
#     print("Numba not installed, using pure python version of sdintegral")
from sigw_fast.libraries import sdintegral as sd
from sigw_fast.sigwfast import sigwfast_mod as gw


def test_pz(p, pstar=5e-4, n1=2, n2=-1, sigma=2):
    nir = n1
    pl1 = (p / pstar) ** nir
    nuv = (n2 - n1) / sigma
    pl2 = (1 + (p / pstar) ** sigma) ** nuv
    return 1e-2 * pl1 * pl2


OMEGA_R = 4.2 * 10**(-5)
CG = 0.39
rd_norm = CG * OMEGA_R 
nd = 150

def compute_w_cs1(w,log10_f_rh,nodes,vals,frequencies,use_mp=False,nd=150,fref=1.):
    nd, ns, darray, ddarray, ssarray = sd.arrays_1(w,frequencies,nd=nd)
    b = sd.beta(w)
    kernel = sd.kernel_1(ddarray, ssarray, b)
    print(f"Nans in kernel: {np.isnan(kernel).any()}")
    print(f"Infs in kernel: {np.isinf(kernel).any()}")
    nk = len(frequencies)   
    Integral = np.empty_like(frequencies)
    Integral = gw.compute_1_k_array(nodes = nodes, vals = vals, nk = nk, komega = frequencies, 
                                            kernel = kernel, darray=darray,
                                            ssarray=ssarray, ddarray=ddarray, 
                                            nd = nd, ns=ns)
    print(f"Nans in Integral: {np.isnan(Integral).any()}")
    print(f"Infs in Integral: {np.isinf(Integral).any()}")
    print(f"Max Integral: {np.max(Integral)}")
    print(f"Min Integral: {np.min(Integral)}")
    f_rh = 10**log10_f_rh
    two_b = 2*b
    norm = rd_norm * (frequencies)**(-2*b) *  (f_rh/fref)**(two_b)   
    OmegaGW = norm * Integral
    return OmegaGW


def main():
    frequencies = np.logspace(-4, -2, 100)
    niter = int(sys.argv[1])

    w = 0.5

    print(f"Calculating for {len(frequencies)} frequencies and {niter} iterations per method")

    # Custom w fluid - original SIGWfast code vs our modified version
    from sigw_fast.EOS import compute
    print("\nTesting custom w_fluid\n")
    start = time.time()
    for i in range(niter):
        omega_gw1 = compute(test_pz, frequencies, Use_Cpp=False, w=w, cs_equal_one=True, f_rh=1e-5)
    end = time.time()
    print(f"Using full python code, {niter} iterations took {end-start:.4f} seconds, average {(end-start)/niter:.4f} seconds, {(end-start)/niter / len(frequencies):.4f} per frequency")
    print(f"Omega mean, std = {np.mean(omega_gw1):.4e}, {np.std(omega_gw1):.4e}")


    # start = time.time()
    # for i in range(niter):
    #     omega_gw2 = compute(test_pz, frequencies, Use_Cpp=True, w=2/3, cs_equal_one=True, f_rh=1e-5)
    # end = time.time()
    # print(f"Using python and CPP code, {niter} iterations took {end-start:.4f} seconds, average {(end-start)/niter:.4f} seconds, {(end-start)/niter / len(frequencies):.4f} per frequency")

    nodes = np.log10(np.geomspace(2e-5, 5e-2, 150))
    vals = np.log10(test_pz(10**nodes))
    start = time.time()
    for i in range(niter):
        omega_gw3 = compute_w_cs1(w = w, log10_f_rh=-5, 
                                  nodes=nodes, vals=vals, frequencies=frequencies, use_mp=False, nd=150)
    end = time.time()
    print(f"Using Fortran function with full Fortran integrator, {niter} iterations took {end-start:.4f} seconds, average {(end-start)/niter:.4f} seconds, {(end-start)/niter / len(frequencies):.4f} per frequency")
    print(f"Omega mean, std = {np.mean(omega_gw3):.4e}, {np.std(omega_gw3):.4e}")

    plt.loglog(frequencies, omega_gw1, label="Full python version", lw=2.5)
    # plt.loglog(frequencies, omega_gw2, label="Python and CPP version", linestyle="-.")
    plt.loglog(frequencies, omega_gw3, label="Fortran func with full Fortran integrator", linestyle="--")
    plt.legend()
    plt.suptitle("Custom w, cs=1")
    plt.show()


if __name__ == '__main__':
    main()