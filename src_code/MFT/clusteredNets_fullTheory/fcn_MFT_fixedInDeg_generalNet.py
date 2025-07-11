import numpy as np
from scipy import integrate
from scipy import optimize
import matplotlib.pyplot as plt

import fcn_MFT_general_tools


'''
set of all functions needed for MFT calculation of LIF model
assumes m dynamical populations 
can work with exponential synapses
assumes all neurons in the same population receive the same number of inputs
no quenched heterogeneity

main inputs are:
    Vr, Vth:                reset and threshold voltages, vectors of length m
    taur, taum, taus:       refractory, membrane, synaptic time constants; vectors of length m
    nu_ext:                 external input rates; vector of length m
    ext_variance:           whether or not external inputs should be considered poisson or deterministic; vector of length m
    Jab, Jab_ext:           recurrent and external synaptic weight matrices; size mxm and mx1, respectively
    Cab, Cab_ext:           recurrent and external degree matrices; size mxm and mx1, respectively
    nu_vec:                 initial guess for the mft solution; vector of size m
'''


     
#%%
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# UPPER TRANSFER FUNCTION LIMIT
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def fcn_thresh_lim(Mu, sigma, Vth):
        
        thresh_lim = ((Vth - Mu) / sigma)
        return thresh_lim


#%%
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# LOWER TRANSFER FUNCTION LIMIT
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def fcn_reset_lim(Mu, sigma, Vr):
        
        reset_lim = ((Vr - Mu) / sigma)
        return reset_lim
    


#%%
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# COMPUTE FIRING RATE
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def fcn_compute_rate(Vr, Vth, Mu, sigma, tau_r, tau_m, tau_s):
    
        BS = fcn_MFT_general_tools.fcn_BrunelSergi_correction(tau_m, tau_s)
        lower_lim = fcn_reset_lim(Mu, sigma, Vr) + BS  
        upper_lim = fcn_thresh_lim(Mu, sigma, Vth) + BS
        
        integral, err = \
        integrate.quad(fcn_MFT_general_tools.fcn_TF_integrand, lower_lim, upper_lim, \
                       epsabs=1e-12, epsrel=1e-12)
            
        inv_rate = (tau_r + tau_m*integral)
        nu = 1/inv_rate 
        return nu
    
#%%
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# COMPUTE INVERSE OF FIRING RATE
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def fcn_compute_inv_rate(Vr, Vth, Mu, sigma, tau_r, tau_m, tau_s):
    
        BS = fcn_MFT_general_tools.fcn_BrunelSergi_correction(tau_m, tau_s)
        lower_lim = fcn_reset_lim(Mu, sigma, Vr) + BS  
        upper_lim = fcn_thresh_lim(Mu, sigma, Vth) + BS
        
        integral, err = \
        integrate.quad(fcn_MFT_general_tools.fcn_TF_integrand, lower_lim, upper_lim, \
                       epsabs=1e-12, epsrel=1e-12)
            
        inv_rate = (tau_r + tau_m*integral)
        return inv_rate
    

#%%           
#------------------------------------------------------------------------------  
#------------------------------------------------------------------------------        
# Compute the mean of the input current 

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------  
def fcn_compute_Mu(nu, nu_ext, Jab, Cab, Jab_ext, Cab_ext, tau_m):   
    
           
    # MEAN OF INPUT TO EACH POPULATION
    mu_recurrent_vec = np.matmul( (Cab*Jab) , nu) * tau_m
    mu_external_vec = Jab_ext*Cab_ext*tau_m*nu_ext
    mu_vec =  mu_recurrent_vec + mu_external_vec
                      
    return mu_vec           
    

#%%
#------------------------------------------------------------------------------  
#------------------------------------------------------------------------------        
# Compute the variance of the input current
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------  
def fcn_compute_Sigma2(nu, nu_ext, Jab, Cab, Jab_ext, Cab_ext, tau_m, ext_variance):   
    
    
    # VARIANCE OF INPUT TO EACH POPULATION
    sig2_recurrent_vec = np.matmul( (Cab*Jab*Jab) , nu) * tau_m
    sig2_external_vec = Jab_ext*Jab_ext*Cab_ext*tau_m*nu_ext*ext_variance
    sig2_vec =  sig2_recurrent_vec + sig2_external_vec
    
    return sig2_vec


#%%       
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# COMPUTE STATIONARY RATES BY SOLVING DYNAMICAL EQUATIONS
#
# all parameters (eg time constants, threshold voltages etc should be vectors of length = # dynamical populations) 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def fcn_compute_MFT_rates_DynEqs(nSteps, dt, T, stop_thresh, plot, \
                                 tau_r, tau_m, tau_s, Vr,  Vth,  \
                                 Jab, Cab, Jab_ext, Cab_ext, ext_variance, nu_ext, nu_vec_in):
    
    
#------------------------------------------------------------------------------
# SETUP
#------------------------------------------------------------------------------
    
    # initialize
    n_pops = np.size(nu_vec_in)
    
    nu = np.zeros((n_pops, nSteps+1))
    
    
#------------------------------------------------------------------------------
# MAIN LOOP
#------------------------------------------------------------------------------     

    # set initial conditions
    nu[:,0] = nu_vec_in.copy()
    
    # time loop
    for i in range(0,nSteps,1):
        
        # compute information for next time step
        
        # compute mean of inputs
        Mu = fcn_compute_Mu(nu[:,i], nu_ext, Jab, Cab, Jab_ext, Cab_ext, tau_m)
    
        # compute variance of inputs
        Sigma2 = fcn_compute_Sigma2(nu[:,i], nu_ext, Jab, Cab, Jab_ext, Cab_ext, tau_m, ext_variance)
            
        # compute standard deviations
        sigma = np.sqrt(Sigma2)
        
        # compute output rates
        phi = np.zeros(n_pops)
        
        for pop_ind in range(0,n_pops):
            
            phi[pop_ind] = fcn_compute_rate(Vr[pop_ind], Vth[pop_ind], \
                                            Mu[pop_ind], sigma[pop_ind], \
                                            tau_r[pop_ind], tau_m[pop_ind], tau_s[pop_ind])
            
                       
        # update rates
        nu[:,i+1] = nu[:,i] + (-nu[:,i]/T + phi/T)*dt
        
        # check tolerances
        nu_check = all(abs(nu[:,i+1]-nu[:,i]) < stop_thresh)
        
        if (nu_check == True):
            
            # delete remaining elements
            nu = np.delete(nu, np.arange(i+1,nSteps+1), 1)
            
            # return final estimates of the rates
            final_rate = nu[:,-1]

            # end loop
            break
        
    else:
        print('ERROR: solution did not converge!')  
        final_rate = np.nan*np.ones(n_pops)

    # plot to see convergence
    if plot == 1:
        plt.figure()
        for i in range(0,n_pops,1):
            plt.plot(nu[i],label=('pop %d' % i))

        plt.ylabel(r'$\nu^\mathrm{mft} \mathrm{\ [spks/sec]}$',fontsize=16)
        plt.xlabel(r'$ \mathrm{iteration \ step,} n}$',fontsize=16)
        plt.legend()
        
    
    return final_rate
 


#%%
#------------------------------------------------------------------------------  
#------------------------------------------------------------------------------  
# COMPUTE WITH ROOT FINDING (JACOBIAN NUMERICALLY ESTIMATED)
#------------------------------------------------------------------------------  
#------------------------------------------------------------------------------ 

# DEFINE ROOT EQUATION
def fcn_root_eqs(nu_vec, \
                 tau_r, tau_m, tau_s, \
                 Vr, Vth,  nu_ext, \
                 Jab, Cab, Jab_ext, Cab_ext, ext_variance):
      
        
    # total number of populations
    n_pops = np.size(nu_vec)
    
    # compute mean and sd               
    Mu = fcn_compute_Mu(nu_vec, nu_ext, \
                        Jab, Cab, Jab_ext, Cab_ext, tau_m)
        
    
    Sigma2 = fcn_compute_Sigma2(nu_vec, nu_ext, Jab, Cab, Jab_ext, Cab_ext, tau_m, ext_variance)

    sigma = np.sqrt(Sigma2)
      
    
    F = np.empty((n_pops))
    
    for i in range(0,n_pops):
        F[i] = nu_vec[i] - fcn_compute_rate(Vr[i], Vth[i], Mu[i], sigma[i], \
                                            tau_r[i], tau_m[i], tau_s[i])

    
    Fvec = np.ndarray.tolist(F)
        
    return Fvec



# SOLVE ROOT EQUATION    
def fcn_MFT_rate_roots(nu_vec_in, nu_ext, \
                       Cab, Jab, Jab_ext, Cab_ext, ext_variance, \
                       tau_r, tau_m, tau_s, Vr, Vth):
    
    # solve self-consistent equations
    sol = optimize.root(fcn_root_eqs, nu_vec_in, \
                        args=(tau_r, tau_m, tau_s, Vr, Vth, nu_ext, \
                              Jab, Cab, Jab_ext, Cab_ext, ext_variance),\
                        jac=False, method='hybr',
                        tol=1e-12,options={'xtol':1e-12})
        
        
    # return solution    
    return sol




#%% STABILITY CALCULATION
### FROM MASCARO & AMIT 1999


def fcn_stability_matrix(nu_fixed_point, \
                         tau_r, tau_m, tau_s, Vr, Vth, nu_ext, \
                         Jab, Cab, Jab_ext, Cab_ext, externalNoise):
    
        
    
    # total number of dynamical populations
    n_dynPops = np.size(nu_fixed_point)
   
    # mean input at fixed point
    Mu_vec = fcn_compute_Mu(nu_fixed_point, nu_ext, Jab, Cab, Jab_ext, Cab_ext, tau_m)
        
    # standard deviation at fixed point
    Sigma2 = fcn_compute_Sigma2(nu_fixed_point, nu_ext, Jab, Cab, Jab_ext, Cab_ext, tau_m, externalNoise)
    Sigma_vec = np.sqrt(Sigma2)
        

    
    # COMPUTE STABILITY MATRIX ELEMENTS
    
    # dphi_m/dnu_n
    dphi_m_dnu_n = np.zeros((n_dynPops, n_dynPops))
    dphi_m_dsig2_m = np.zeros((n_dynPops, n_dynPops))
    dsig2_m_dnu_n = np.zeros((n_dynPops, n_dynPops))
    delta_m_n = np.zeros((n_dynPops, n_dynPops))
    np.fill_diagonal(delta_m_n,1)
    
    S = np.zeros((n_dynPops, n_dynPops))
    
    
    # LOOP OVER ALL POPULATIONS
    for m in range(0, n_dynPops):

        
        phi = nu_fixed_point[m]
            
        BS = fcn_MFT_general_tools.fcn_BrunelSergi_correction(tau_m[m], tau_s[m])
                    
        bm = (Vth[m] - Mu_vec[m])/Sigma_vec[m] + BS
        am = (Vr[m] - Mu_vec[m])/Sigma_vec[m] + BS
                
        gm_b = fcn_MFT_general_tools.fcn_TF_integrand(bm)
        gm_a = fcn_MFT_general_tools.fcn_TF_integrand(am)
                
        dbm_dsigm = -(Vth[m] - Mu_vec[m])/(Sigma_vec[m]**2)
        dam_dsigm = -(Vr[m] - Mu_vec[m])/(Sigma_vec[m]**2)
        
        dphi_m_dsig_m = -(phi**2)*tau_m[m]*( gm_b*dbm_dsigm - gm_a*dam_dsigm )

        
        # take derivatives with respect to all others       
        for n in range(0, n_dynPops):
                        
                
            dbm_dnun = -tau_m[m]*( (Cab[m,n]*Jab[m,n])/Sigma_vec[m] + (Vth[m]-Mu_vec[m])*(Cab[m,n]*Jab[m,n]**2)/(2*Sigma_vec[m]**3) )
            
            dam_dnun = -tau_m[m]*( (Cab[m,n]*Jab[m,n])/Sigma_vec[m] + (Vr[m]-Mu_vec[m])*(Cab[m,n]*Jab[m,n]**2)/(2*Sigma_vec[m]**3) )      
                                               
            dphi_m_dsig2_m[m,n] = dphi_m_dsig_m/(2*Sigma_vec[m])
            dphi_m_dnu_n[m,n] = -(phi**2)*tau_m[m]*(gm_b*dbm_dnun - gm_a*dam_dnun)
            dsig2_m_dnu_n[m,n] = tau_m[m]*Jab[m,n]*Jab[m,n]*Cab[m,n] 

            S[m,n] = (1/tau_m[m])*( dphi_m_dnu_n[m,n] -  dphi_m_dsig2_m[m,n]*dsig2_m_dnu_n[m,n] - delta_m_n[m,n] ) 
                
            
    
    # compute eigenvalues
    eigenvals_S = np.linalg.eigvals(S)
    realPart_eigvals_S = np.real(eigenvals_S)
            
    return S, eigenvals_S, realPart_eigvals_S       


#%% ALTERNATE STABILITY CALCULATION
### from ostojic 2014, pitta and brunel, bachman and morrison

def fcn_stability_matrix_alternate(nu_fixed_point, \
                                         tau_r, tau_m, tau_s, Vr, Vth, nu_ext, \
                                             Jab, Cab, Jab_ext, Cab_ext, externalNoise, \
                                                 tau_dynamics):
    
        
    
    # total number of dynamical populations
    n_dynPops = np.size(nu_fixed_point)
   
    # mean input at fixed point
    Mu_vec = fcn_compute_Mu(nu_fixed_point, nu_ext, Jab, Cab, Jab_ext, Cab_ext, tau_m)
        
    # standard deviation at fixed point
    Sigma2 = fcn_compute_Sigma2(nu_fixed_point, nu_ext, Jab, Cab, Jab_ext, Cab_ext, tau_m, externalNoise)
    Sigma_vec = np.sqrt(Sigma2)
        

    
    # COMPUTE STABILITY MATRIX ELEMENTS
    
    dphi_m_dmu_m = np.zeros((n_dynPops))
    dphi_m_dsig_m = np.zeros((n_dynPops))
    
    dmu_m_dnu_n = np.zeros((n_dynPops, n_dynPops))    
    dsig_m_dnu_n = np.zeros((n_dynPops, n_dynPops))

    dphi_m_dnu_n = np.zeros((n_dynPops, n_dynPops))
    
    delta_m_n = np.zeros((n_dynPops, n_dynPops))
    np.fill_diagonal(delta_m_n,1)
    
    S = np.zeros((n_dynPops, n_dynPops))
    
    
    # LOOP OVER ALL POPULATIONS
    for m in range(0, n_dynPops):

        
        phi = nu_fixed_point[m]
            
        BS = fcn_MFT_general_tools.fcn_BrunelSergi_correction(tau_m[m], tau_s[m])
                    
        bm = (Vth[m] - Mu_vec[m])/Sigma_vec[m] + BS
        am = (Vr[m] - Mu_vec[m])/Sigma_vec[m] + BS
                
        gm_b = fcn_MFT_general_tools.fcn_TF_integrand(bm)
        gm_a = fcn_MFT_general_tools.fcn_TF_integrand(am)
                
        dbm_dsigm = -(Vth[m] - Mu_vec[m])/(Sigma_vec[m]**2)
        dam_dsigm = -(Vr[m] - Mu_vec[m])/(Sigma_vec[m]**2)
        
        dbm_dmum = -1/Sigma_vec[m]
        dam_dmum = -1/Sigma_vec[m]
        
        dphi_m_dmu_m[m] =  -(phi**2)*tau_m[m]*( gm_b*dbm_dmum - gm_a*dam_dmum )
        dphi_m_dsig_m[m] = -(phi**2)*tau_m[m]*( gm_b*dbm_dsigm - gm_a*dam_dsigm )

        
        # take derivatives with respect to all others       
        for n in range(0, n_dynPops):
                        
                
            dmu_m_dnu_n[m,n] = tau_m[m]*Jab[m,n]*Cab[m,n]             
            dsig_m_dnu_n[m,n] = (1 / (2*Sigma_vec[m])) * tau_m[m]*Jab[m,n]*Jab[m,n]*Cab[m,n] 
            
            dphi_m_dnu_n[m,n] = dphi_m_dmu_m[m]*dmu_m_dnu_n[m,n] + dphi_m_dsig_m[m]*dsig_m_dnu_n[m,n]
            

            S[m,n] = (1/tau_dynamics[m])*( dphi_m_dnu_n[m,n] - delta_m_n[m,n] ) 
                
            
    
    # compute eigenvalues
    eigenvals_S = np.linalg.eigvals(S)
    realPart_eigvals_S = np.real(eigenvals_S)
            
    return S, eigenvals_S, realPart_eigvals_S    

