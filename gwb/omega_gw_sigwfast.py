# Interface to SIGWFAST for Omega_GW calculation

from gwb.base import BaseSpectrumCalculator

class SIGW_FAST(BaseSpectrumCalculator):
    """
    Omega_GW calculator using SIGWFAST
    """
    def __init__(self, frequencies, Use_Cpp=True, RD = True):
        self.frequencies = frequencies
        self.RD = RD
        self.Use_Cpp = Use_Cpp

    def __call__(self, power_spectrum, *extra_params):
        from sigwfast import OmegaGW
        return OmegaGW(power_spectrum, self.fparams, *extra_params, **self.kwargs)
# Compare this snippet from gwb/omega_gw_sigwfast.py:
#