# Test omega_gw_dblquad.py
# 2020-07-01

import numpy as np
import jax.numpy as jnp
from insert_name_here.omega_gw_dblquad import OmegaGWDblquad
from insert_name_here.omega_gw_jax import OmegaGWjax
from insert_name_here.omega_gw_numpy import OmegaGWnumpy
from jax import jit
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time

################################################################

A = 0.01
Delta = 0.5
ks = 0.001

kstart = 1e-5 
kend = 10**(-2.3) 
nk = 100    
ns = 10
nt = 1000

niterations = 100

################################################################

s = jnp.linspace(0, 1, ns)

def t(k, A, Delta, ks):
    nt = 10000
    lower_t = jnp.exp(-5 * Delta) / (k/ks)
    # set upper_t to 1 if any of the values is smaller than 1
    upper_t = jnp.exp(3 * Delta) / (k/ks)
    upper_t = jnp.where(upper_t < 100., 100., upper_t)
    return jnp.geomspace(lower_t, upper_t, nt)

@jit
def pzeta_ln(k, A, Delta, ks):
    term1 = A / ((2 * jnp.pi)**0.5) / Delta
    term2 = jnp.exp(-(1 / (2 * Delta**2)) * (jnp.log(k/ks))**2)
    return term1 * term2

kvals = jnp.geomspace(kstart, kend, nk)
fvec = kvals/2/jnp.pi

calculator_jax = OmegaGWjax(pzeta_ln, s, t, f=fvec)

# Initial evalutaion to compile the function (not included in timing)
omega_gw_jax = calculator_jax(fvec, A, Delta, ks)
start_time = time.time()
for _ in range(niterations):
    omega_gw_jax = calculator_jax(fvec, A, Delta, ks)
time_jax = time.time() - start_time
print(f"Jax ({niterations} iterations): {time_jax} seconds")

################################################################

s = np.linspace(0, 1, ns)

def t(k, A, Delta, ks):
    nt = 10000
    lower_t = np.exp(-5 * Delta) / (k/ks)
    upper_t = np.exp(3 * Delta) / (k/ks)
    upper_t = np.where(upper_t < 100., 100., upper_t)
    return np.geomspace(lower_t, upper_t, nt)

class pzeta_ln:
    def __init__(self):
        self.nevals = 0
    def __call__(self, k, A, Delta, ks):
        term1 = A / ((2 * np.pi)**0.5) / Delta
        term2 = np.exp(-(1 / (2 * Delta**2)) * (np.log(k/ks))**2)
        if np.iterable(k):
            self.nevals += len(k)
        else:
            self.nevals += 1
        return term1 * term2
pzeta_ln = pzeta_ln()

kvals = np.geomspace(kstart, kend, nk)
fvec = kvals/2/np.pi

calculator_dblquad = OmegaGWDblquad(pzeta_ln, s, t, f=fvec)

start_time = time.time()
omega_gw_dblquad = calculator_dblquad(fvec, A, Delta, ks)
time_dblquad = time.time() - start_time
print(f"Dblquad (single iteration): {time_dblquad} seconds")

################################################################

s = np.linspace(0, 1, ns)

def t(k, A, Delta, ks):
    nt = 10000
    lower_t = np.exp(-3 * Delta) / (k/ks)
    upper_t = np.exp(3 * Delta) / (k/ks)
    return np.geomspace(lower_t, upper_t, nt)

def pzeta_ln(k, A, Delta, ks):
    term1 = A / ((2 * np.pi)**0.5) / Delta
    term2 = np.exp(-(1 / (2 * Delta**2)) * (np.log(k/ks))**2)
    return term1 * term2

kvals = np.geomspace(kstart, kend, nk)
fvec = kvals/2/np.pi

calculator_numpy = OmegaGWnumpy(pzeta_ln, s, t, f=fvec, upsample=False)

start_time = time.time()
for _ in range(niterations):
    omega_gw_numpy = calculator_numpy(fvec, A, Delta, ks)
time_numpy = time.time() - start_time
print(f"Numpy ({niterations} iterations): {time_numpy} seconds")

################################################################

fig = plt.figure(figsize=(8, 8))

# Create a GridSpec with 2 rows and 1 column with the height ratio 2:1
gs = GridSpec(2, 1, height_ratios=[2, 1])

# Create the subplots
ax1 = fig.add_subplot(gs[0])    
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax1.loglog(kvals, omega_gw_jax, alpha=0.5, label="jax")
ax1.loglog(kvals, omega_gw_numpy, alpha=0.5, label="numpy")
ax1.loglog(kvals, omega_gw_dblquad, alpha=0.5, label="dblquad")
ax1.set_ylabel('$\Omega_{GW}$')
ax1.set_ylim(1e-15, omega_gw_dblquad.max()*1.1)
ax1.legend()
ax1.grid()

ax2.loglog(kvals, np.abs(omega_gw_dblquad - omega_gw_jax)/omega_gw_dblquad, label='jax')
ax2.loglog(kvals, np.abs(omega_gw_dblquad - omega_gw_numpy)/omega_gw_dblquad, label='numpy simps')
ax2.set_xlabel('k')
ax2.set_ylabel('Relative difference in $\Omega_{GW}$')
ax2.grid()

plt.tight_layout()
# plt.savefig('omega_gw_ln_manual_combined.png')
plt.show()