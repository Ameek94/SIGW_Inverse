# Interface to SIGWFAST for Omega_GW calculation

import time
from matplotlib import pyplot as plt
import numpy as np
from base import BaseSpectrumCalculator
from functools import partial

class SIGW_FAST(BaseSpectrumCalculator):
    """
    Omega_GW calculator using SIGWFAST
    """
    def __init__(self, frequencies, Use_Cpp=True, RD = True, w=1/3, 
                 cs_equal_one = False, sample_w = False, w_bounds = None, 
                 log10_f_rh =  0., sample_f_rh = False, f_rh_bounds = None):
        """
        """
        self.fref = 1e-3
        self.frequencies = frequencies/self.fref
        self.RD = RD
        self.Use_Cpp = Use_Cpp
        self.w_bounds = w_bounds if w_bounds is not None else [0.,1.]
        self.f_rh_bounds = f_rh_bounds if f_rh_bounds is not None else [-1,1]
        if RD:
            try:
                from sigw_fast.RD import compute
                self.gwb_calculator = lambda power_spectrum, frequencies: compute(power_spectrum, frequencies, Use_Cpp=self.Use_Cpp)
                self.w = 1/3
                self.sample_w = False
                self.cs_equal_one = False
            except ImportError:
                raise ImportError("SIGWfast is not available.")
        else:
            try:
                from gwb.sigw_fast.EOS import compute
                self.gwb_calculator = partial(compute, Use_Cpp = self.Use_Cpp, fref = self.fref)
                self.log10_f_rh = log10_f_rh
                self.w = w
                self.sample_w = sample_w
                self.cs_equal_one = cs_equal_one
            except ImportError:
                raise ImportError("SIGWfastEOS is not available.")

    def __call__(self, power_spectrum, *extra_params):
        return self.gwb_calculator(power_spectrum, self.frequencies, *extra_params)

    def get_extra_param_specs(self):
        specs = {}
        if self.RD:
            specs['w'] = {'value': self.w, 'sampled': False}
        else:
            specs['w'] = {'value': self.w, 'sampled': self.sample_w, 'bounds': self.w_bounds}
            specs['log10_f_rh'] = {'value': self.log10_f_rh, 'sampled': self.sample_f_rh, 'bounds': self.f_rh_bounds }
        return specs
    
def test_pz(p,pstar=5e-4,n1=2,n2=-1,sigma=2):
    nir = n1
    pl1 = (p/pstar)**nir
    nuv = (n2 - n1)/sigma
    pl2 = (1+(p/pstar)**sigma)**nuv
    return 1e-2 * pl1 * pl2

def main():
    frequencies = np.logspace(-5,1, 100)
    start = time.time()
    calculator = SIGW_FAST(frequencies, RD = True, Use_Cpp=True)
    results = calculator(test_pz)
    print("Time taken: ", time.time() - start)  
    plt.loglog(frequencies, results)
    plt.show()

    start = time.time()
    calculator = SIGW_FAST(frequencies, RD = True, Use_Cpp=False)
    results = calculator(test_pz)
    print("Time taken: ", time.time() - start)
    plt.loglog(frequencies, results)
    plt.show()

if __name__ == "__main__":
    main()