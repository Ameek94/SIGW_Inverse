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
from scipy.interpolate import interp1d
from sigw_fast.sigwfast import sigwfast_mod as gw


def test_pz(p, pstar=5e-4, n1=2, n2=-1, sigma=2):
    nir = n1
    pl1 = (p / pstar) ** nir
    nuv = (n2 - n1) / sigma
    pl2 = (1 + (p / pstar) ** sigma) ** nuv
    return 1e-2 * pl1 * pl2


def main():
    fmin = 1e-5
    fmax = 1e-1
    pz_freqs = np.geomspace(fmin,fmax,100)
    niter = int(sys.argv[1])
    num_nodes = int(sys.argv[2])
    nodes = np.log10(np.geomspace(fmin,fmax,num_nodes))
    vals = np.log10(test_pz(10**nodes))
    p_fmin = test_pz(fmin)
    p_fmax = test_pz(fmax)
    fill_value = (np.log10(p_fmin), np.log10(p_fmax))
    pz_interp = interp1d(nodes, vals,fill_value=fill_value,bounds_error=False)
    test_pz_interp = lambda x: 10**pz_interp(np.log10(x))
    pz_fortran = gw.power_spectrum_k_array(nodes, vals, pz_freqs)

    plt.scatter(10**nodes, test_pz(10**nodes), marker='x',color='k')
    plt.loglog(pz_freqs, 10**pz_interp(np.log10(pz_freqs)), label="Scipy linear interp",ls = '--')
    plt.loglog(pz_freqs, pz_fortran, label="Fortran linear interp",ls='-.')
    plt.loglog(pz_freqs, test_pz(pz_freqs), label="Original")
    plt.legend()
    plt.show()

    frequencies = np.logspace(-4, -2, 50)
    print(f"Calculating for {len(frequencies)} frequencies and {niter} iterations per method")

    # RD - original SIGWfast code vs our modified version
    omega_gw0 = compute(test_pz, frequencies, Use_Cpp=False)


    print("Testing RD\n")
    start = time.time()
    for i in range(niter):
        omega_gw1 = compute(test_pz_interp, frequencies, Use_Cpp=False)
    end = time.time()
    print(f"Using full python code, {niter} iterations took {end-start:.4f} seconds, average {(end-start)/niter:.4f} seconds, {(end-start)/niter / len(frequencies):.4f} per frequency")

    start = time.time()
    for i in range(niter):
        omega_gw2 = compute(test_pz_interp, frequencies, Use_Cpp=True)
    end = time.time()
    print(f"Using python and CPP code, {niter} iterations took {end-start:.4f} seconds, average {(end-start)/niter:.4f} seconds, {(end-start)/niter / len(frequencies):.4f} per frequency")

    from sigw_fast.sigwfast_fortran import compute_rd

    start = time.time()
    for i in range(niter):
        omega_gw3 = compute_rd(nodes, vals, frequencies,use_mp=False,nd=150)
    end = time.time()
    print(f"Using Fortran function with full Fortran integrator, {niter} iterations took {end-start:.4f} seconds, average {(end-start)/niter:.4f} seconds, {(end-start)/niter / len(frequencies):.4f} per frequency")


    # from sigw_fast.gwb_jax import OmegaGWjax
    # import jax.numpy as jnp
    # import jax
    # from jax import jit, block_until_ready
    # from jax.sharding import PartitionSpec as P, NamedSharding, Mesh
    # from jax.experimental import mesh_utils
    
    # # mesh2d = jax.make_mesh((8, 1), ('x', 'y'))
    # # devices = mesh_utils.create_device_mesh((jax.device_count(),))  # 1D mesh
    # # # Step 2: Create a mesh and named sharding strategy
    # # mesh = Mesh(devices, axis_names=('x',))  # Name the axis "x"
    # # sharding = NamedSharding(mesh, P('x'))  # Distribute along 'x'
    # # s = jnp.linspace(0, 1, 96)  # First rescaled internal momentum
    # # ss = jax.device_put(s, sharding) #NamedSharding(mesh1d, P('x')))
    # s = jnp.linspace(0, 1, 25)
    # t = jnp.logspace(-5,5, 1800)  # Second rescaled internal momentum
    # freqs = jnp.array(frequencies)
    # ## Expand t to add a new axis
    # t_expanded = jnp.expand_dims(t, axis=-1)
    # ## Repeat t along the new axis to match the shape (ns,nt)
    # t = jnp.repeat(t_expanded, len(freqs), axis=-1)
    # # tt = jax.device_put(t, NamedSharding(mesh2d, P('x', 'y')))
    # start = time.time()
    # # gwb_calculator =  OmegaGWjax(s=ss, t=tt, f=freqs, kernel="RD", upsample=False)
    # gwb_calculator =  OmegaGWjax(s=s, t=t, f=freqs, kernel="RD", upsample=False)
    # pz = jit(test_pz)
    # for i in range(niter):
    #     gwb_amp = gwb_calculator(pz,freqs)
    #     block_until_ready(gwb_amp)
    # end = time.time()
    # print(f"Using jax, {niter} iterations took {end-start:.4f} seconds, average {(end-start)/niter:.4f} seconds, {(end-start) /niter / len(frequencies):.4f}  per frequency")

    plt.loglog(frequencies, omega_gw0, label="Original", lw=3,alpha=0.5)
    plt.loglog(frequencies, omega_gw1, label="Python, interpolated", linestyle=":")
    plt.loglog(frequencies, omega_gw2, label="Python and CPP version, interpolated", linestyle="-.")
    plt.loglog(frequencies, omega_gw3, label="Fortran integrator, Fortran interpolation", linestyle="--")
    # plt.loglog(frequencies,gwb_amp,label="Jax",linestyle=':')
    plt.legend()
    plt.suptitle("RD")
    plt.show()

    # Custom w fluid - original SIGWfast code vs our modified version
    # from sigw_fast.EOS import compute_w
    # print("\nTesting custom w_fluid\n")
    # start = time.time()
    # for i in range(niter):
    #     omega_gw1 = compute_w(test_pz, frequencies, Use_Cpp=False, w=0.001)
    # end = time.time()
    # print(f"Using full python code, {niter} iterations took {end-start:.4f} seconds, average {(end-start)/niter:.4f} seconds, {(end-start)/niter / len(frequencies):.4f} per frequency")

    # start = time.time()
    # for i in range(niter):
    #     omega_gw2 = compute_w(test_pz, frequencies, Use_Cpp=True, w=0.001)
    # end = time.time()
    # print(f"Using python and CPP code, {niter} iterations took {end-start:.4f} seconds, average {(end-start)/niter:.4f} seconds, {(end-start)/niter / len(frequencies):.4f} per frequency")

    # from sigw_fast.sigwfast_fortran import compute_w_fld

    # start = time.time()
    # for i in range(niter):
    #     omega_gw3 = compute_w_fld(frequencies, w=0.001)
    # end = time.time()
    # print(f"Using Fortran function with full Fortran integrator, {niter} iterations took {end-start:.4f} seconds, average {(end-start)/niter:.4f} seconds, {(end-start)/niter / len(frequencies):.4f} per frequency")

    # plt.loglog(frequencies, omega_gw1, label="Full python version", lw=2.5)
    # plt.loglog(frequencies, omega_gw2, label="Python and CPP version", linestyle="-.")
    # plt.loglog(frequencies, omega_gw3, label="Fortran func with full Fortran integrator", linestyle="--")
    # plt.legend()
    # plt.suptitle("Custom w")
    # plt.show()


if __name__ == '__main__':
    main()