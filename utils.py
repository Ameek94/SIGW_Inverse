from jax import jit

@jit
def unit_transform(x,bounds):
    """
    Transform array from original domain given by bounds to [0,1]. 
    """
    ux = (x - bounds[0])/(bounds[1]-bounds[0])
    return ux
@jit
def unit_untransform(x,bounds):
    """
    Transform array to original domain given by bounds from [0,1]
    """
    xu = x*(bounds[1]-bounds[0]) + bounds[0]
    return xu
