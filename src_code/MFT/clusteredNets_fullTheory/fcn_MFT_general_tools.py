
import numpy as np
from scipy import special


#%%

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# DEFINITION OF MAIN FUNCTION INTEGRAL (INCLUDES SQRT PI FACTOR)
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def fcn_TF_integrand(u):
    
    # smart way of setting up the integrand to avoid numerical issues!
    
    # use asymptotic expansion of erfc if u is very negative
    if u < -15:
        return (1-1/(2*u**2)+3/(4*u**4)-15/(8*u**6))*(-1/u)
    else:
        A = u*u
        B = special.erfc(-u)
        return np.sqrt(np.pi)*np.exp(A + np.log(B))
    
    # This way causes problems b/c of multiplication of large # x small #
    #return np.sqrt(np.pi)*np.exp(u**2)*(1+special.erf(u))


#%%

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Brunel-Sergi correction for transfer function integral
# Takes into account effects of synaptic time constants
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def fcn_BrunelSergi_correction(tau_m, tau_s):
    a = -special.zeta(1/2)/np.sqrt(2) 
    BS = a*np.sqrt(tau_s/tau_m)
    return BS
    


