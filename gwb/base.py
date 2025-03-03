# Abstract base class for GWB spectrum calculation from power spectrum

class BaseSpectrumCalculator:

    def get_extra_param_specs(self):
        """
        Returns a dictionary where keys are the names of extra parameters
        and values are the corresponding numpyro distribution objects.
        """
        return {}

    def __call__(self, power_spectrum, *extra_params):
        """
        Calculate the Omegagw spectrum given the primordial curvature power spectrum and extra parameters.
        Must be implemented by subclasses.

        Arguments
        ---------
        power_spectrum: callable,
            the function for the primordial curvature power spectrum P_zeta(k)
        extra_params: list|tuple,
            extra parameters required by Omegagw calculation method
        """
        raise NotImplementedError