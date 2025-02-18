import numpy as np
import jax.numpy as jnp
from insert_name_here.omega_gw_dblquad import OmegaGWDblquad
from interpolation.omega_gw_jax import OmegaGWjax, I_sq_IRD_LV, I_sq_IRD_res
from insert_name_here.omega_gw_numpy import OmegaGWnumpy
from jax import jit
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
from scipy.special import sici

### Plotting
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 14
# plt.rcParams['text.color'] = 'black'

################################################################
# Analytical results for a scale invariant P_zeta from https://arxiv.org/abs/1904.12879

def omega_test_LV(k, kmax, etaR, As):
    xmaxR = etaR * kmax
    k = jnp.where(k <= 2 * 1.2e-3, k, 0)
    xR = k * etaR
    k_tilde = xR / xmaxR  # equivalent to k/kmax

    si, ci = sici(xR / 2)
    num = (4 * ci**2 + (jnp.pi - 2 * si)**2) / 86016000000.
    prefactor = num * As**2 * xR**3 * xmaxR**5

    term1 = jnp.heaviside(xmaxR - xR, 0.) * (
        5376. - 17640. * k_tilde + 23760. * k_tilde**2
        - 16425. * k_tilde**3 + 5825. * k_tilde**4 - 847. * k_tilde**5
    )

    term2 = jnp.heaviside(xR - xmaxR, 0.) * (
        k_tilde**-5. * (2. - k_tilde)**6
        * (4. - 8. * k_tilde - 9. * k_tilde**2
           + 13. * k_tilde**3 + 49. * k_tilde**4)
    )

    result = prefactor * (term1 + term2)
    return 4 * result

def s0(xR, xmaxR):
    threshold1 = 2 * xmaxR / (1 + jnp.sqrt(3))
    threshold2 = 2 * xmaxR / jnp.sqrt(3)
    condition1 = xR <= threshold1
    condition2 = (xR >= threshold1) & (xR <= threshold2)
    result = jnp.where(condition1, 1.0,
                       jnp.where(condition2, 2 * xmaxR / xR - jnp.sqrt(3), 0.0))
    return result

def omega_test_res(k, kmax, etaR, As, fudge=2.3):
    xmaxR = etaR * kmax
    xR = k * etaR
    k_tilde = xR / xmaxR  # so k/kmax

    s_0 = s0(xR, xmaxR)
    coefficient = 2.30285/102400000. * jnp.sqrt(3.)
    polynomial = 15. - 10. * s_0**2. + 3. * s_0**4.

    result = fudge * coefficient * As**2. * xR**7 * s_0 * polynomial

    return 4 * result


def omega_test_tot(k, kmax, etaR, As=1):
    return omega_test_LV(k, kmax, etaR, As) + omega_test_res(k, kmax, etaR, As)

################################################################
# Test parameters for the numerical calculation
As = 1.
kmax = 1e-3
etaR = 1./kmax
################################################################

kstart = 10**(-5) # Minimum k value to compute Omega_GW
kend = 10**(-2.3) # Maximum k value to compute Omega_GW
nk = 300 # Number of k values to compute Omega_GW (logarithmically spaced)
ns = 50 # Number of s values to compute Omega_GW (linearly spaced)
nt = 500 # Number of t values to compute Omega_GW (logarithmically spaced)

niterations = 10 # Number of iterations to time the calculation

################################################################

kvals = jnp.geomspace(kstart, kend, nk)
fvec = kvals/2/jnp.pi # Calculator lives in f-space

s = jnp.linspace(0, 1, ns)
t = jnp.geomspace(1e-3, 1e3, nt)
t = jnp.stack([t for _ in range(len(kvals))], axis=-1)

# We use a heaviside function as P_zeta as we have analyical results 
# for the LV and Resonant parts
@jit
def pzeta_heaviside(k, As, kmax, etaR):
    result = jnp.heaviside(kmax - k, 0)*As
    return result

# Mock kernel to turn off LV or res kernels (returns 0)
@jit
def zero_kernel(t, s, k, kmax, etaR):
    return 0.

# Define the Omega_GW calculator with the full kernel, the LV kernel and the Resonant kernel
# Currently they are only fully implemented in Jax.
calculator_full = OmegaGWjax(pzeta_heaviside, s, t, f=fvec, norm="CT", kernel="I_MD_to_RD")
calculator_LV = OmegaGWjax(pzeta_heaviside, s, t, f=fvec, norm="CT", kernel=[I_sq_IRD_LV, zero_kernel])
calculator_res = OmegaGWjax(pzeta_heaviside, s, t, f=fvec, norm="CT", kernel=[zero_kernel, I_sq_IRD_res])

# Initial evalutaion to compile the function (not included in timing)
omega_gw_full = calculator_full(fvec, As, kmax, etaR)
start_time = time.time()
# Time the calculation
# We are only interested in the time it takes to compute the full Omega_GW
for _ in range(niterations):
    omega_gw_full = calculator_full(fvec, As, kmax, etaR)
time_jax = time.time() - start_time
print(f"{niterations} iterations took: {time_jax} seconds")

# Compute Omega_GW for the LV and Resonant parts separately
omega_gw_LV = calculator_LV(fvec, As, kmax, etaR)
omega_gw_res = calculator_res(fvec, As, kmax, etaR)

################################################################
# Plot the results
fig = plt.figure(figsize=(8, 5))
ax1 = fig.add_subplot()

ax1.loglog(kvals/kmax, omega_gw_LV/As**2., alpha=0.5, label="Large V numerical")
ax1.loglog(kvals/kmax, omega_test_LV(kvals, kmax, etaR, As)/As**2., alpha=0.5,  
           linestyle=':', label="Large V analytical")

ax1.loglog(kvals/kmax, omega_gw_res/As**2., alpha=0.5, label="Resonant numerical")
ax1.loglog(kvals/kmax, omega_test_res(kvals, kmax, etaR, As)/As**2., alpha=0.5, linestyle=':', label="Resonant analytical")

ax1.loglog(kvals/kmax, omega_gw_full/As**2., alpha=0.7, label="Total numerical")
ax1.loglog(kvals/kmax, omega_test_tot(kvals, kmax, etaR, As)/As**2., alpha=0.7,  linestyle=':', label="Total analytical")

ax1.set_ylabel('$\Omega_{GW}(\eta_c,k)/A_s^2$')
ax1.set_xlabel('$k/k_{max}$')
ax1.legend()
plt.tight_layout()
plt.show()