import jax.numpy as jnp
from jax import vmap
from jax.lax import map
import math

pi = math.pi

# This is based on the quick integration method of GT and MP
# I have ignored some constant factors in the definitions


class OmegaGWGrid:

    def __init__(self,
                 omgw_karr: jnp.array,
                 pz_karr: jnp.array = None):
        
        self.omgw_karr = omgw_karr
        if pz_karr is None:
            self.pz_karr = omgw_karr
        else:
            self.pz_karr = pz_karr
        om_ij = lambda k: vmap(vmap(self.omk_ij,in_axes=(None,None,0),out_axes=(0))
                               ,in_axes=(None,0,None),out_axes=(0))(k,self.pz_karr,self.pz_karr)
        self.omkij = map(om_ij,self.omgw_karr,batch_size=25)

    def omk_ij(self,k,pi,pj): # using p_i = p_i / k
        pi = pi/k
        pj = pj/k
        pipj3 = pi**2 + pj**2 - 3
        den = 1024 * pi**8 * pj**8
        term1 = pi*pj * (4*pi**2 - (1 -pj**2 + pi**2)**2 ) **2 *  3*pipj3**2
        cond1 = jnp.logical_and((pj - abs(1-pi))>0,(1+pi-pj)>0)
        ht1 = jnp.where(cond1,1.,0.)
        term1 = term1 * ht1
        logterm = (3-(pi+pj)**2)/(3-(pi-pj)**2)
        term2 = (-4*pi*pj + pipj3*jnp.log(abs(logterm)))**2 + pi**2
        ht2 = jnp.where((pi+pj-jnp.sqrt(3))> 0,1.,0.)
        term2 = term2 * ht2
        res = term1 * term2 / den
        return res

    def get_pk_coeffs(self,pz_func):
        return pz_func(self.pz_karr)

    def __call__(self,pz_func):
        pz_amps = self.get_pk_coeffs(pz_func)
        omgw = jnp.einsum("i,j,kij->k",pz_amps,pz_amps,self.omkij)
        return omgw
