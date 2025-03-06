import os
os.environ["OMP_NUM_THREADS"] = "8"
import sys
sys.path.append('../../')
import time
from gwb.sigw_fast.RD import compute
import numpy as np
from jax import jit


@jit
def test_pz(p,pstar=5e-4,n1=2,n2=-1,sigma=2):
    nir = n1
    pl1 = (p/pstar)**nir
    nuv = (n2 - n1)/sigma
    pl2 = (1+(p/pstar)**sigma)**nuv
    return 1e-2 * pl1 * pl2

frequencies = np.logspace(-4,-2, 100)

niter = int(sys.argv[1])

start = time.time()
for i in range(niter):
    omega_gw = compute(test_pz, frequencies,Use_Cpp=True)
end = time.time()
print(f"with CPP, {niter} iterations took {end-start} seconds, average {(end-start)/niter} seconds, {(end-start)/niter / len(frequencies)} per frequency")


start = time.time()
for i in range(niter):
# niter = 1
    omega_gw = compute(test_pz, frequencies,Use_Cpp=False)
end = time.time()
print(f"without CPP, {niter} iterations took {end-start} seconds, average {(end-start)/niter} seconds, {(end-start)/niter / len(frequencies)} per frequency")
