# This is based on the quick integration method of GT and MP
# I have ignored some constant factors in the definitions

import math
import numpy as np
pi = math.pi

class OmegaGWGrid:

    def __init__(self, omgw_karr, pz_karr=None):

        self.omgw_karr = np.asarray(omgw_karr)
        # Use omgw_karr for pz_karr if not provided.
        self.pz_karr = np.asarray(pz_karr) if pz_karr is not None else self.omgw_karr
        
        # Compute the grid for each k value in a fully vectorized manner.
        # Let N be the number of k values and M the number of p values.
        # We want a result of shape (N, M, M) where for each k,
        # each (i,j) entry is given by omk_ij(k, pz[i], pz[j]).
        self.omkij = self._compute_omkij(self.omgw_karr, self.pz_karr)
        
    def _compute_omkij(self, omgw_karr, pz_karr):
        # Reshape arrays for broadcasting:
        # k: shape (N, 1, 1)
        # pi: shape (N, M, 1) 
        # pj: shape (N, 1, M)
        k = omgw_karr[:, None, None]
        pi = pz_karr[None, :, None] / k
        pj = pz_karr[None, None, :] / k
        
        # Compute the common terms
        pipj3 = pi**2 + pj**2 - 3
        den = 1024 * (pi**8) * (pj**8)
        
        # Compute term1 and apply condition cond1
        term1 = (pi * pj * (4 * pi**2 - (1 - pj**2 + pi**2)**2)**2 *
                 (3 * pipj3**2))
        cond1 = np.logical_and((pj - np.abs(1 - pi)) > 0, (1 + pi - pj) > 0)
        ht1 = np.where(cond1, 1.0, 0.0)
        term1 *= ht1
        
        # Compute term2 and apply condition cond2
        logterm = (3 - (pi + pj)**2) / (3 - (pi - pj)**2)
        term2 = (-4 * pi * pj + pipj3 * np.log(np.abs(logterm)))**2 + pi**2
        ht2 = np.where((pi + pj - np.sqrt(3)) > 0, 1.0, 0.0)
        term2 *= ht2
        
        # Combine terms to obtain the final result
        res = term1 * term2 / den
        return res

    def get_pk_coeffs(self, pz_func):
        # Apply the provided function to the pz_karr.
        return pz_func(self.pz_karr)

    def __call__(self, pz_func, k):
        # Get the coefficients from pz_func
        pz_amps = self.get_pk_coeffs(pz_func)
        # Perform the contraction:
        #   Sum over indices i and j: pz_amps[i] * pz_amps[j] * omkij[k,i,j]
        # This yields an array of length equal to the number of k values.
        omgw = np.einsum("i,j,kij->k", pz_amps, pz_amps, self.omkij)
        return omgw
