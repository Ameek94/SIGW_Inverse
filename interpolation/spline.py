import numpy as np
from functools import partial
from scipy.interpolate import CubicSpline
from interpolation.base import Interpolator
from utils import unit_untransform

def pow10(x):
    return np.power(10,x)

def log10(x):
    return np.log10(x)

def unit_transform(x,bounds):
    """
    Transform array from original domain given by bounds to [0,1]. 
    """
    ux = (x - bounds[0])/(bounds[1]-bounds[0])
    return ux
def unit_untransform(x,bounds):
    """
    Transform array to original domain given by bounds from [0,1]
    """
    xu = x*(bounds[1]-bounds[0]) + bounds[0]
    return xu

# def spline_predict_log(x_train,y_train,x_pred):
#     """
#     Cubic spline to interpolate log10 P_zeta as a function of log10_k
#     """
#     spl = CubicSpline(x_train,y_train,check=False)
#     y_pred = spl(x_pred)
#     return y_pred

# def spline_predict(x_train,y_train,x_pred):
#     """
#     Obtain spline prediction after exponentiating log10 P_zeta spline. 

#     Arguments
#     ----------
#     x_train: np.ndarray,
#         the training x values
#     y_train: np.ndarray,
#         the training y values
#     x_pred: np.ndarray,
#         the x values to predict at
#     """
#     x_pred = log10(x_pred)
#     return pow10(spline_predict_log(x_train,y_train,x_pred))


class Spline(Interpolator):
    k_min: float
    k_max: float
    method: str
    bounds: np.array
    """
    Spline interpolation class. Wrapper around interpax.interp1d. Interpolation is done in log10-log10 space.
    Node locations and bounds are assumed to be in log10 space.
    """

    def __init__(self,
                 k_min,
                 k_max,
                 sample_nodes: bool = False,
                 fixed_nodes = None,               
                 num_nodes = 5,  
                 method: str = "cubic",):
        super().__init__(k_min, k_max,sample_nodes,fixed_nodes,num_nodes)
        self.method = method
        #add binned nodes


    def __call__(self, nodes, values, x):
        """
        Obtain the spline prediction at x given nodes and values.
        The nodes and values are assumed to be in log10 space while x is in physical space.

        Arguments
        ---------
        nodes: array-like,
            the node locations
        values: array-like,
            the values at the node locations
        x: array-like,
            the points at which to evaluate the spline
        """
        spline = self.interpolate(nodes, values)
        return spline(x)
        

    def interpolate(self, nodes, values):
        """
        Create a spline interpolation function given nodes and values. 
        
        Arguments
        ---------
        nodes: array-like,
            the node locations
        values: array-like,
            the values at the node locations
        """

        # nodes = unit_untransform(nodes, self.bounds)
        func = CubicSpline(x = nodes, y = values, extrapolate=True)

        # Interpolate in log10-log10 space
        def spline(x):
            logx = np.log10(x)
            res = np.power(10,func(logx) )
            res = np.where(logx<self.k_min, 0.0, res)
            res = np.where(logx>self.k_max, 0.0, res)
            return res

        return spline

    def derivative(self, nodes, values):
        """
        Obtain the derivative of the spline interpolation at x given nodes and values.
        The nodes and values are assumed to be in log10 space while x is in physical space.

        Arguments
        ---------
        nodes: array-like,
            the node locations
        values: array-like,
            the values at the node locations
        x: array-like,
            the points at which to evaluate the spline
        """
        func = CubicSpline(x = nodes, y = values, extrapolate=True)
        der = func.derivative()
        def spline_derivative(x):
            logx = np.log10(x)
            res = 10*func(logx)*der(logx)/x
            return res
        return spline_derivative