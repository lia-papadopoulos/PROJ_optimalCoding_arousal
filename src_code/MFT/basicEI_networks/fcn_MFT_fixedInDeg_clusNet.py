import numpy as np
from scipy import special
from scipy import integrate
from scipy import optimize
import matplotlib.pyplot as plt
import sys



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# COMPUTE FIRING RATES OF E-I NETWORKS WITH MEAN FIELD THEORY
# 
# ASSUMPTIONS 
# ALL NEURONS RECEIVE SAME NUMBER OF INPUTS (HOMOGENEOUS CONNECTIVITY)
# ALL SYNAPTIC WEIGHTS OF TYPE J_ab ARE THE SAME
# 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#%% FUNCTION TO COMPUTE DEPRESSION FACTORS FOR CLSUTERSE
def fcn_compute_depressFactors(p, bgrE, bgrI, \
                               JplusEE, JplusII, JplusEI, JplusIE):
    
    
    # fraction of E/I neurons per clusters
    fE = (1-bgrE)/p  
    fI = (1-bgrI)/p       


    # potentiation and depression    
    if fI == 0:
        gII = 1
    else:
        gII = (fI + fI - p*fI*fI - fI*fI*JplusII)/(fI + fI - p*fI*fI - fI*fI)
               
    if fE == 0:
        gEE = 1
    else:
        gEE = (fE + fE - p*fE*fE - fE*fE*JplusEE)/(fE + fE - p*fE*fE - fE*fE)
            
    if (fI==0 and fE==0):
        gEI = 1
        gIE = 1
    else:
        gEI = (fI + fE - p*fI*fE - fI*fE*JplusEI)/(fI + fE - p*fI*fE - fI*fE)
        gIE = (fI + fE - p*fI*fE - fI*fE*JplusIE)/(fI + fE - p*fI*fE - fI*fE)   


    # comment if you want to use version where inter-clsuter weights are depressed
    gEE = 1.
    gEI = 1.
    gIE = 1.
    gII = 1.
    
    print('NOTE: ONLY W/IN CLUSTER POTENTIATION. TOTAL SYNAPTIC WEIGHT NOT PRESERVED.')         
        
        
    return gEE, gEI, gIE, gII

     
#%%
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# DEFINITION OF MAIN FUNCTION INTEGRAL (INCLUDES SQRT PI FACTOR)
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def fcn_TF_integrand(u):
    
    # smart way of setting up the integrand to avoid numerical difficulties!
    
    # use asymptotic expansion of erfc if u is very negative
    if u < -15:
        return (1-1/(2*u**2)+3/(4*u**4)-15/(8*u**6))*(-1/u)
    else:
        A = u*u
        B = special.erfc(-u)
        return np.sqrt(np.pi)*np.exp(A + np.log(B))
    
#%%
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# COMPUTE UPPER TRANSFER FUNCTION LIMIT
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def fcn_thresh_lim(Mu, sigma, Vth):
        
        thresh_lim = ((Vth - Mu) / sigma)
        return thresh_lim

#%%
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# COMPUTE LOWER TRANSFER FUNCTION LIMIT
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def fcn_reset_lim(Mu, sigma, Vr):
        
        reset_lim = ((Vr - Mu) / sigma)
        return reset_lim


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

#%%
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# COMPUTE FIRING RATE
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def fcn_compute_rate(Vr, Vth, Mu, sigma, tau_r, tau_m, tau_s):
    
        BS = fcn_BrunelSergi_correction(tau_m, tau_s)
        lower_lim = fcn_reset_lim(Mu, sigma, Vr) + BS  
        upper_lim = fcn_thresh_lim(Mu, sigma, Vth) + BS
        
        integral, err = \
        integrate.quad(fcn_TF_integrand, lower_lim, upper_lim, \
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
    
        BS = fcn_BrunelSergi_correction(tau_m, tau_s)
        lower_lim = fcn_reset_lim(Mu, sigma, Vr) + BS  
        upper_lim = fcn_thresh_lim(Mu, sigma, Vth) + BS
        
        integral, err = \
        integrate.quad(fcn_TF_integrand, lower_lim, upper_lim, \
                       epsabs=1e-12, epsrel=1e-12)
            
        inv_rate = (tau_r + tau_m*integral)
        return inv_rate
    
#%%
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# COMPUTE WEIGHT AND DEGREE MATRICES
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def fcn_compute_weight_degree_mat(
                   tau_m_e, tau_m_i, \
                   Cee, Cei, Cie, Cii, Cext, \
                   Jee, Jei, Jie, Jii, Jee_ext, Jie_ext, \
                   Jee_p, Jee_m, Jei_p, Jei_m, Jie_p, Jie_m, Jii_p, Jii_m, \
                   fE, fEb, fI, fIb, n_e_pops, n_i_pops):
    
    
    # NUMBER OF CLUSTERS/BG POPS OF EACH TYPE
    n_Eclu = n_e_pops - 1
    n_Iclu = n_i_pops - 1
    
    # COMPUTE WEIGHT AND DEGREE MATRICES
    Jab = np.zeros((n_e_pops+n_i_pops, n_e_pops+n_i_pops))
    Cab = np.zeros((n_e_pops+n_i_pops, n_e_pops+n_i_pops))
    Jab_ext = np.zeros(n_e_pops+n_i_pops)
    Cab_ext = np.zeros(n_e_pops+n_i_pops)   
    tau_m_mat = np.zeros((n_e_pops+n_i_pops,n_e_pops+n_i_pops))
    
    # inputs to E clusters ---------------------------------------------------
    for i in range(0, n_Eclu):

        # input from E clusters
        for j in range(0, n_Eclu):
            # input from same cluster
            if j == i:
                Jab[i,j] = Jee_p
                Cab[i,j] = fE*Cee
            # input from different cluster
            elif j!=i:
                Jab[i,j] = Jee_m
                Cab[i,j] = fE*Cee
                
        # input from E background
        j = n_e_pops - 1
        Jab[i,j] = Jee_m
        Cab[i,j] = fEb*Cee
        
        # input from I clusters
        for j in range(0, n_Iclu):
            # input from same cluster
            if j == i:
                Jab[i,j+n_e_pops] = -Jei_p
                Cab[i,j+n_e_pops] = fI*Cei            
            # input from different cluster
            elif j!=i:
                Jab[i,j+n_e_pops] = -Jei_m
                Cab[i,j+n_e_pops] = fI*Cei
                
        # input from I background
        j = n_i_pops - 1
        Jab[i,j+n_e_pops] = -Jei_m
        Cab[i,j+n_e_pops] = fIb*Cei 
                               
        # input from external sources
        Jab_ext[i] = Jee_ext
        Cab_ext[i] = Cext
        
        # tau
        tau_m_mat[i,i] = tau_m_e 
                
        
    # inputs to E background---------------------------------------------------
    i = n_e_pops-1
    
    # input from E clusters
    for j in range(0,n_Eclu):

        Jab[i,j] = Jee_m
        Cab[i,j] = fE*Cee
        
    # input from E background
    j = n_e_pops-1
    Jab[i,j] = Jee
    Cab[i,j] = fEb*Cee
    
    # input from I clusters
    for j in range(0, n_Iclu):

        Jab[i,j+n_e_pops] = -Jei_m
        Cab[i,j+n_e_pops] = fI*Cei        
            
    # input from I background
    j = n_i_pops - 1
    Jab[i,j+n_e_pops] = -Jei
    Cab[i,j+n_e_pops] = fIb*Cei      
     
    # input from external sources
    Jab_ext[i] = Jee_ext
    Cab_ext[i] = Cext
    
    # tau
    tau_m_mat[i,i] = tau_m_e 


    # inputs to I clusters---------------------------------------------------
    for i in range(0, n_Iclu):
        
        # input from E clusters
        for j in range(0, n_Eclu):           
            # input from same cluster
            if j == i:
                Jab[i+n_e_pops,j] = Jie_p
                Cab[i+n_e_pops,j] = fE*Cie               
            # input from different cluster
            elif j!=i:
                Jab[i+n_e_pops,j] = Jie_m
                Cab[i+n_e_pops,j] = fE*Cie
                                
        # input from E background
        j = n_e_pops - 1
        Jab[i+n_e_pops,j] = Jie_m
        Cab[i+n_e_pops,j] = fEb*Cie
                
                
        # input from I clusters
        for j in range(0, n_Iclu):
            # input from same cluster
            if j == i:
                Jab[i+n_e_pops,j+n_e_pops] = -Jii_p
                Cab[i+n_e_pops,j+n_e_pops] = fI*Cii            
            # input from different cluster
            elif j!=i:
                Jab[i+n_e_pops,j+n_e_pops] = -Jii_m
                Cab[i+n_e_pops,j+n_e_pops] = fI*Cii
                
        # input from I background
        j = n_i_pops - 1
        Jab[i+n_e_pops,j+n_e_pops] = -Jii_m
        Cab[i+n_e_pops,j+n_e_pops] = fIb*Cii 
        
        # input from external sources
        Jab_ext[i+n_e_pops] = Jie_ext
        Cab_ext[i+n_e_pops] = Cext
        
        # tau
        tau_m_mat[i+n_e_pops,i+n_e_pops] = tau_m_i  
        
        
    # inputs to I background---------------------------------------------------
    i = n_i_pops-1
    
    # input from E clusters
    for j in range(0,n_Eclu,1):

        Jab[i+n_e_pops,j] = Jie_m
        Cab[i+n_e_pops,j] = fE*Cie
                        
    # input from E background
    j = n_e_pops-1
    Jab[i+n_e_pops,j] = Jie
    Cab[i+n_e_pops,j] = fEb*Cie
    
    # input from I clusters
    for j in range(0, n_Iclu):
        Jab[i+n_e_pops,j+n_e_pops] = -Jii_m
        Cab[i+n_e_pops,j+n_e_pops] = fI*Cii            
        
    # input from I background
    j = n_i_pops - 1
    Jab[i+n_e_pops,j+n_e_pops] = -Jii
    Cab[i+n_e_pops,j+n_e_pops] = fIb*Cii     
      
    # input from external sources
    Jab_ext[i+n_e_pops] = Jie_ext
    Cab_ext[i+n_e_pops] = Cext
    
    # tau
    tau_m_mat[i+n_e_pops,i+n_e_pops] = tau_m_i 
    
    
    return Jab, Cab, Jab_ext, Cab_ext, tau_m_mat

        

#%%           
#------------------------------------------------------------------------------  
#------------------------------------------------------------------------------        
# Compute the mean of the input current 

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------  
def fcn_compute_Mu(nu_e, nu_i, nu_ext_e, nu_ext_i, \
                   tau_m_e, tau_m_i, \
                   Cee, Cei, Cie, Cii, Cext, \
                   Jee, Jei, Jie, Jii, Jee_ext, Jie_ext, \
                   Jee_p, Jee_m, Jei_p, Jei_m, Jie_p, Jie_m, Jii_p, Jii_m, \
                   fE, fEb, fI, fIb, n_e_pops, n_i_pops):   
    
    
    Jab, Cab, Jab_ext, Cab_ext, tau_m_mat = fcn_compute_weight_degree_mat(
                                           tau_m_e, tau_m_i, \
                                           Cee, Cei, Cie, Cii, Cext, \
                                           Jee, Jei, Jie, Jii, Jee_ext, Jie_ext, \
                                           Jee_p, Jee_m, Jei_p, Jei_m, Jie_p, Jie_m, Jii_p, Jii_m, \
                                           fE, fEb, fI, fIb, n_e_pops, n_i_pops)
    
    
    # MEAN OF INPUT TO EACH POPULATION
    nu_recurrent_vec = np.concatenate((nu_e,nu_i))
    nu_external_vec = np.concatenate((nu_ext_e*np.ones(n_e_pops),nu_ext_i*np.ones(n_i_pops))) 
    mu_recurrent_vec = np.matmul(np.matmul(tau_m_mat,(Cab*Jab)), nu_recurrent_vec)
    mu_external_vec = Jab_ext*Cab_ext*np.diag(tau_m_mat)*nu_external_vec
    mu_vec =  mu_recurrent_vec + mu_external_vec
    Mu_e = mu_vec[:n_e_pops]
    Mu_i = mu_vec[n_e_pops:]
                      
    return Mu_e, Mu_i           
    

#%%

#------------------------------------------------------------------------------  
#------------------------------------------------------------------------------        
# Compute the variance of the input current
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------  
def fcn_compute_Sigma2(nu_e, nu_i, nu_ext_e, nu_ext_i, \
                       tau_m_e, tau_m_i, \
                       Cee, Cei, Cie, Cii, Cext, \
                       Jee, Jei, Jie, Jii, Jee_ext, Jie_ext, \
                       Jee_p, Jee_m, Jei_p, Jei_m, Jie_p, Jie_m, Jii_p, Jii_m, \
                       fE, fEb, fI, fIb, n_e_pops, n_i_pops, \
                       ext_variance, additional_externalVariance):   
    
    
    Jab, Cab, Jab_ext, Cab_ext, tau_m_mat = fcn_compute_weight_degree_mat(
                                           tau_m_e, tau_m_i, \
                                           Cee, Cei, Cie, Cii, Cext, \
                                           Jee, Jei, Jie, Jii, Jee_ext, Jie_ext, \
                                           Jee_p, Jee_m, Jei_p, Jei_m, Jie_p, Jie_m, Jii_p, Jii_m, \
                                           fE, fEb, fI, fIb, n_e_pops, n_i_pops)    
    
    # VARIANCE OF INPUT TO EACH POPULATION
    nu_recurrent_vec = np.concatenate((nu_e,nu_i))
    nu_external_vec = np.concatenate((nu_ext_e*np.ones(n_e_pops),nu_ext_i*np.ones(n_i_pops))) 
    sig2_recurrent_vec = np.matmul(np.matmul(tau_m_mat,(Cab*Jab*Jab)), nu_recurrent_vec)
    sig2_external_vec = Jab_ext*Jab_ext*Cab_ext*np.diag(tau_m_mat)*nu_external_vec*ext_variance
    sig2_additional_externalVariance = additional_externalVariance*sig2_external_vec
    sig2_vec =  sig2_recurrent_vec + sig2_external_vec + sig2_additional_externalVariance
    Sigma2_e = sig2_vec[:n_e_pops]
    Sigma2_i = sig2_vec[n_e_pops:]
   
    return Sigma2_e, Sigma2_i

#%%       
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# COMPUTE STATIONARY RATES BY SOLVING DYNAMICAL EQUATIONS
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def fcn_compute_MFT_rates_DynEqs(nSteps, dt, Te, Ti, stop_thresh, plot, \
                                 tau_r, tau_m_e, tau_m_i, tau_s_e, tau_s_i, \
                                 Vr_e, Vr_i, Vth_e, Vth_i, nu_ext_e, nu_ext_i, \
                                 Cee, Cei, Cie, Cii, Cext, \
                                 Jee, Jei, Jie, Jii, Jee_ext, Jie_ext, \
                                 Jee_p, Jee_m, Jei_p, Jei_m, Jie_p, Jie_m, Jii_p, Jii_m, \
                                 fE, fEb, fI, fIb, n_e_pops, n_i_pops, \
                                 ext_variance, additional_externalVariance, nu_vec_in):
    
    
#------------------------------------------------------------------------------
# SETUP
#------------------------------------------------------------------------------
    
    # initial conditions
    nu_eo = nu_vec_in[:n_e_pops]
    nu_io = nu_vec_in[n_e_pops:]
    
    
    # initialize
    nu_e = np.zeros((n_e_pops, nSteps+1))
    nu_i = np.zeros((n_i_pops, nSteps+1))
    
              
      
#------------------------------------------------------------------------------
# MAIN LOOP
#------------------------------------------------------------------------------     

    # set initial conditions
    nu_e[:,0] = nu_eo
    nu_i[:,0] = nu_io
    
    
    # time loop
    for i in range(0,nSteps,1):
        
        # compute information for next time step
        
        # compute mean of inputs
        Mu_e, Mu_i = fcn_compute_Mu(nu_e[:,i], nu_i[:,i], nu_ext_e, nu_ext_i, \
                                    tau_m_e, tau_m_i, \
                                    Cee, Cei, Cie, Cii, Cext, \
                                    Jee, Jei, Jie, Jii, Jee_ext, Jie_ext, \
                                    Jee_p, Jee_m, Jei_p, Jei_m, Jie_p, Jie_m, Jii_p, Jii_m,\
                                    fE, fEb, fI, fIb, n_e_pops, n_i_pops)
    
        # compute variance of inputs
        Sigma2_e, Sigma2_i = fcn_compute_Sigma2(nu_e[:,i], nu_i[:,i], nu_ext_e, nu_ext_i, \
                                                tau_m_e, tau_m_i, \
                                                Cee, Cei, Cie, Cii, Cext, \
                                                Jee, Jei, Jie, Jii, Jee_ext, Jie_ext, \
                                                Jee_p, Jee_m, Jei_p, Jei_m, Jie_p, Jie_m, Jii_p, Jii_m, \
                                                fE, fEb, fI, fIb, n_e_pops, n_i_pops,\
                                                ext_variance, additional_externalVariance)
            
        # compute standard deviations
        sigma_e = np.sqrt(Sigma2_e)
        sigma_i = np.sqrt(Sigma2_i)        
        
        # compute output rates
        phi_e = np.zeros(n_e_pops)
        phi_i = np.zeros(n_i_pops) 
        
        for epop_ind in range(0,n_e_pops,1):
            phi_e[epop_ind] = fcn_compute_rate(Vr_e, Vth_e, \
                                               Mu_e[epop_ind], sigma_e[epop_ind], \
                                               tau_r, tau_m_e, tau_s_e)
            
        for ipop_ind in range(0,n_i_pops,1):
            phi_i[ipop_ind] = fcn_compute_rate(Vr_i, Vth_i, \
                                               Mu_i[ipop_ind], sigma_i[ipop_ind], \
                                               tau_r, tau_m_i, tau_s_i)
                       
        # update rates
        nu_e[:,i+1] = nu_e[:,i] + (-nu_e[:,i]/Te + phi_e/Te)*dt
        nu_i[:,i+1] = nu_i[:,i] + (-nu_i[:,i]/Ti + phi_i/Ti)*dt
        
        # check tolerances
        nu_e_check = all(abs(nu_e[:,i+1]-nu_e[:,i]) < stop_thresh)
        nu_i_check = all(abs(nu_i[:,i+1]-nu_i[:,i]) < stop_thresh)
        
        if (nu_e_check == True and nu_i_check == True):
            
            # delete remaining elements
            nu_e = np.delete(nu_e, np.arange(i+1,nSteps+1), 1)
            nu_i = np.delete(nu_i, np.arange(i+1,nSteps+1), 1)
            
            # return final estimates of the rates
            final_rate_e = nu_e[:,-1]
            final_rate_i = nu_i[:,-1]
            
            # end loop
            break
        
    else:
        print('ERROR: solution did not converge!')  
        final_rate_e = np.nan*np.ones(n_e_pops)
        final_rate_i = np.nan*np.ones(n_i_pops)                     

    # plot to see convergence
    if plot == 1:
        plt.figure()
        for i in range(0,n_e_pops,1):
            plt.plot(nu_e[i],label=('Epop %d' % i))
        for i in range(0,n_i_pops,1):
            plt.plot(nu_i[i],label=('Ipop %d' % i))
        plt.ylabel(r'$\nu^\mathrm{mft} \mathrm{\ [spks/sec]}$',fontsize=16)
        plt.xlabel(r'$ \mathrm{iteration \ step,} n}$',fontsize=16)
        plt.legend([r'$ \nu_{E,c1}$', r'$ \nu_{E,c2}$', r'$ \nu_{E,b}$', r'$\nu_{I}$'])
        
        
    return final_rate_e, final_rate_i
 


#%%
#------------------------------------------------------------------------------  
#------------------------------------------------------------------------------  
# COMPUTE WITH ROOT FINDING (JACOBIAN NUMERICALLY ESTIMATED)
#------------------------------------------------------------------------------  
#------------------------------------------------------------------------------ 

# DEFINE ROOT EQUATION
def fcn_root_eqs(nu_vec, \
                 tau_r, tau_m_e, tau_m_i, tau_s_e, tau_s_i, \
                 Vr_e, Vr_i, Vth_e, Vth_i, nu_ext_e, nu_ext_i, \
                 Cee, Cei, Cie, Cii, Cext, \
                 Jee, Jei, Jie, Jii, Jee_ext, Jie_ext, \
                 Jee_p, Jee_m, Jei_p, Jei_m, Jie_p, Jie_m, Jii_p, Jii_m, \
                 fE, fEb, fI, fIb, n_e_pops, n_i_pops, \
                 ext_variance, additional_externalVariance):
    
      
    # compute mean and sd               
    Mu_e, Mu_i = fcn_compute_Mu(nu_vec[:n_e_pops], nu_vec[n_e_pops:], \
                                nu_ext_e, nu_ext_i, \
                                tau_m_e, tau_m_i, \
                                Cee, Cei, Cie, Cii, Cext, \
                                Jee, Jei, Jie, Jii, Jee_ext, Jie_ext, \
                                Jee_p, Jee_m, Jei_p, Jei_m, Jie_p, Jie_m, Jii_p, Jii_m, \
                                fE, fEb, fI, fIb, n_e_pops, n_i_pops)
                 
        
    Sigma2_e, Sigma2_i = fcn_compute_Sigma2(nu_vec[:n_e_pops], nu_vec[n_e_pops:], \
                                            nu_ext_e, nu_ext_i, \
                                            tau_m_e, tau_m_i, \
                                            Cee, Cei, Cie, Cii, Cext, \
                                            Jee, Jei, Jie, Jii, Jee_ext, Jie_ext, \
                                            Jee_p, Jee_m, Jei_p, Jei_m, Jie_p, Jie_m, Jii_p, Jii_m, \
                                            fE, fEb, fI, fIb, n_e_pops, n_i_pops,\
                                            ext_variance, additional_externalVariance)
        
    
    sigma_e = np.sqrt(Sigma2_e)
    sigma_i = np.sqrt(Sigma2_i)
      
    F = np.empty((n_e_pops + n_i_pops))
    
    for i in range(0,n_e_pops,1):
        F[i] = nu_vec[i] - fcn_compute_rate(Vr_e, Vth_e, Mu_e[i], sigma_e[i], \
                                            tau_r, tau_m_e, tau_s_e)
    for i in range(0,n_i_pops,1):            
        F[i+n_e_pops] = nu_vec[i+n_e_pops] - fcn_compute_rate(Vr_i, Vth_i, \
                                                              Mu_i[i], \
                                                              sigma_i[i], \
                                                              tau_r, tau_m_i, tau_s_i)
    
    Fvec = np.ndarray.tolist(F)
        
    return Fvec


# SOLVE ROOT EQUATION    
def fcn_MFT_rate_roots(tau_r, tau_m_e, tau_m_i, tau_s_e, tau_s_i, \
                       Vr_e, Vr_i, Vth_e, Vth_i, nu_ext_e, nu_ext_i, \
                        Cee, Cei, Cie, Cii, Cext, \
                        Jee, Jei, Jie, Jii, Jee_ext, Jie_ext, \
                        Jee_p, Jee_m, Jei_p, Jei_m, Jie_p, Jie_m, Jii_p, Jii_m,
                        fE, fEb, fI, fIb, n_e_pops, n_i_pops, \
                        ext_variance, additional_externalVariance, \
                        nu_vec_in):
    
    # solve self-consistent equations
    sol = optimize.root(fcn_root_eqs, nu_vec_in, \
                        args=(tau_r, tau_m_e, tau_m_i, tau_s_e, tau_s_i, \
                              Vr_e, Vr_i, Vth_e, Vth_i, nu_ext_e, nu_ext_i, \
                              Cee, Cei, Cie, Cii, Cext, \
                              Jee, Jei, Jie, Jii, Jee_ext, Jie_ext, \
                              Jee_p, Jee_m, Jei_p, Jei_m, Jie_p, Jie_m, Jii_p, Jii_m,
                              fE, fEb, fI, fIb, n_e_pops, n_i_pops, \
                              ext_variance, additional_externalVariance),\
                        jac=False, method='hybr',
                        tol=1e-12,options={'xtol':1e-12})
        
        
    # return solution    
    return sol




#%% STABILITY CALCULATION


def fcn_stability_matrix(nu_fixed_point, \
                         tau_r, tau_m_e, tau_m_i, tau_s_e, tau_s_i, \
                         Vr_e, Vr_i, Vth_e, Vth_i, \
                         nu_ext_e, nu_ext_i, \
                         Cee, Cei, Cie, Cii, Cext, \
                         Jee, Jei, Jie, Jii, Jee_ext, Jie_ext, \
                         Jee_p, Jee_m, Jei_p, Jei_m, Jie_p, Jie_m, Jii_p, Jii_m,
                         fE, fEb, fI, fIb, n_e_pops, n_i_pops, \
                         ext_variance, additional_externalVariance):
    
        
    
    # COMPUTE WEIGHT AND DEGREE MATRICES Jab and Cab
    Jab, Cab, _, _, _ = fcn_compute_weight_degree_mat(
                                               tau_m_e, tau_m_i, \
                                               Cee, Cei, Cie, Cii, Cext, \
                                               Jee, Jei, Jie, Jii, Jee_ext, Jie_ext, \
                                               Jee_p, Jee_m, Jei_p, Jei_m, Jie_p, Jie_m, Jii_p, Jii_m, \
                                               fE, fEb, fI, fIb, n_e_pops, n_i_pops)
            
   
    # COMPUTE MEAN AND STANDARD DEVIATION OF ALL POPS EVALUATED AT FIXED POINTS
    Mu_e, Mu_i = fcn_compute_Mu(nu_fixed_point[:n_e_pops], nu_fixed_point[n_e_pops:], \
                                nu_ext_e, nu_ext_i, \
                                tau_m_e, tau_m_i, \
                                Cee, Cei, Cie, Cii, Cext, \
                                Jee, Jei, Jie, Jii, Jee_ext, Jie_ext, \
                                Jee_p, Jee_m, Jei_p, Jei_m, Jie_p, Jie_m, Jii_p, Jii_m, \
                                fE, fEb, fI, fIb, n_e_pops, n_i_pops)
                 
        
    Sigma2_e, Sigma2_i = fcn_compute_Sigma2(nu_fixed_point[:n_e_pops], nu_fixed_point[n_e_pops:], \
                                            nu_ext_e, nu_ext_i, \
                                            tau_m_e, tau_m_i, \
                                            Cee, Cei, Cie, Cii, Cext, \
                                            Jee, Jei, Jie, Jii, Jee_ext, Jie_ext, \
                                            Jee_p, Jee_m, Jei_p, Jei_m, Jie_p, Jie_m, Jii_p, Jii_m, \
                                            fE, fEb, fI, fIb, n_e_pops, n_i_pops,\
                                            ext_variance, additional_externalVariance)
    
    Mu_vec = np.concatenate((Mu_e, Mu_i))
    Sig_vec = np.sqrt(np.concatenate((Sigma2_e, Sigma2_i)))
    tau_m_vec = np.concatenate((np.ones(n_e_pops)*tau_m_e,np.ones(n_i_pops)*tau_m_i))
        
    # COMPUTE STABILITY MATRIX ELEMENTS
    
    # dphi_m/dnu_n
    dphi_m_dnu_n = np.zeros((n_e_pops+n_i_pops, n_e_pops+n_i_pops))
    dphi_m_dsig2_m = np.zeros((n_e_pops+n_i_pops, n_e_pops+n_i_pops))
    dsig2_m_dnu_n = np.zeros((n_e_pops+n_i_pops, n_e_pops+n_i_pops))
    delta_m_n = np.zeros((n_e_pops+n_i_pops, n_e_pops+n_i_pops))
    np.fill_diagonal(delta_m_n,1)
    
    S = np.zeros((n_e_pops+n_i_pops,n_e_pops+n_i_pops))
    
    # E POPULATIONS
    for m in range(0,n_e_pops,1):

        
        phi = fcn_compute_rate(Vr_e, Vth_e, Mu_vec[m], Sig_vec[m], \
                                       tau_r, tau_m_vec[m], tau_s_e)
            
        BS = fcn_BrunelSergi_correction(tau_m_vec[m], tau_s_e)
                    
                    
        bm = (Vth_e - Mu_vec[m])/Sig_vec[m] + BS
        am = (Vr_e - Mu_vec[m])/Sig_vec[m] + BS
                
        gm_b = fcn_TF_integrand(bm)
        gm_a = fcn_TF_integrand(am)
                
        dbm_dsigm = -(Vth_e - Mu_vec[m])/(Sig_vec[m]**2)
        dam_dsigm = -(Vr_e - Mu_vec[m])/(Sig_vec[m]**2)
        
        dphi_m_dsig_m = -(phi**2)*tau_m_vec[m]*( gm_b*dbm_dsigm - gm_a*dam_dsigm )

        
        # take derivatives with respect to all others       
        for n in range(0,n_e_pops+n_i_pops,1):
                        
                
            dbm_dnun = -tau_m_vec[m]*( (Cab[m,n]*Jab[m,n])/Sig_vec[m] + \
                                       (Vth_e-Mu_vec[m])*(Cab[m,n]*Jab[m,n]**2)/(2*Sig_vec[m]**3) )
            
            dam_dnun = -tau_m_vec[m]*( (Cab[m,n]*Jab[m,n])/Sig_vec[m] + \
                                       (Vr_e-Mu_vec[m])*(Cab[m,n]*Jab[m,n]**2)/(2*Sig_vec[m]**3) )      
                    
                                               
            dphi_m_dsig2_m[m,n] = dphi_m_dsig_m/(2*Sig_vec[m])
            dphi_m_dnu_n[m,n] = -(phi**2)*tau_m_vec[m]*(gm_b*dbm_dnun - gm_a*dam_dnun)
            dsig2_m_dnu_n[m,n] = tau_m_vec[m]*Jab[m,n]*Jab[m,n]*Cab[m,n] 

            S[m,n] = (1/tau_m_vec[m])*( dphi_m_dnu_n[m,n] -  \
                                        dphi_m_dsig2_m[m,n]*dsig2_m_dnu_n[m,n] - \
                                        delta_m_n[m,n] ) 
                
                
                
    # I POPULATIONS
    for m in range(n_e_pops,n_e_pops+n_i_pops,1):
        
        phi = fcn_compute_rate(Vr_i, Vth_i, Mu_vec[m], Sig_vec[m], \
                                       tau_r, tau_m_vec[m], tau_s_i)
            
        BS = fcn_BrunelSergi_correction(tau_m_vec[m], tau_s_i)            
                    
                    
        bm = (Vth_i - Mu_vec[m])/Sig_vec[m] + BS
        am = (Vr_i - Mu_vec[m])/Sig_vec[m] + BS
                
        gm_b = fcn_TF_integrand(bm)
        gm_a = fcn_TF_integrand(am)
                
        dbm_dsigm = -(Vth_i - Mu_vec[m])/(Sig_vec[m]**2)
        dam_dsigm = -(Vr_i - Mu_vec[m])/(Sig_vec[m]**2)
        
        dphi_m_dsig_m = -(phi**2)*tau_m_vec[m]*( gm_b*dbm_dsigm - gm_a*dam_dsigm )

        
        # take derivatives with respect to all others       
        for n in range(0,n_e_pops+n_i_pops,1):
                
            dbm_dnun = -tau_m_vec[m]*( (Cab[m,n]*Jab[m,n])/Sig_vec[m] + \
                                       (Vth_i-Mu_vec[m])*(Cab[m,n]*Jab[m,n]**2)/(2*Sig_vec[m]**3) )
            
            dam_dnun = -tau_m_vec[m]*( (Cab[m,n]*Jab[m,n])/Sig_vec[m] + \
                                       (Vr_i-Mu_vec[m])*(Cab[m,n]*Jab[m,n]**2)/(2*Sig_vec[m]**3) )      
                    
                                               
            dphi_m_dsig2_m[m,n] = dphi_m_dsig_m/(2*Sig_vec[m])
            dphi_m_dnu_n[m,n] = -(phi**2)*tau_m_vec[m]*(gm_b*dbm_dnun - gm_a*dam_dnun)
            dsig2_m_dnu_n[m,n] = tau_m_vec[m]*Jab[m,n]*Jab[m,n]*Cab[m,n] 

            S[m,n] = (1/tau_m_vec[m])*( dphi_m_dnu_n[m,n] -  \
                                        dphi_m_dsig2_m[m,n]*dsig2_m_dnu_n[m,n] - \
                                        delta_m_n[m,n] )                 
                
                
    
    # compute eigenvalues
    eigenvals_S = np.linalg.eigvals(S)
    realPart_eigvals_S = np.real(eigenvals_S)
            
    return S, eigenvals_S, realPart_eigvals_S       



#%% MASTER FUNCTION FOR MFT CALCULATIONS

def fcn_master_MFT(s_params, a_params):    
    
    
    tau_r = s_params.tau_r            # refractory period
    tau_m_e = s_params.tau_m_e        # membrane time constant E
    tau_m_i = s_params.tau_m_i        # membrane time constant I
    tau_s_e = s_params.tau_s_e        # synaptic time constant E
    tau_s_i = s_params.tau_s_i        # synaptic time constant I
    
    Vr_e = s_params.Vr_e              # reset potential E
    Vr_i = s_params.Vr_i              # reset potential E
    
    Vth_e = s_params.Vth_e            # threshold potential E
    Vth_i = s_params.Vth_i            # threshold potential I
    
    nu_ext_e = s_params.nu_ext_e[0]   # avg baseline afferent rate to E 
    nu_ext_i = s_params.nu_ext_i[0] 
    
    Cee = s_params.Cee 
    Cei = s_params.Cei 
    Cii = s_params.Cii 
    Cie = s_params.Cie 
    Cext = s_params.Cext 

    Jee = s_params.Jee 
    Jei = s_params.Jei 
    Jii = s_params.Jii 
    Jie = s_params.Jie 
    Jee_ext = s_params.Jee_ext        
    Jie_ext = s_params.Jie_ext 
    
    p = s_params.p
    popsizeE = s_params.popsizeE
    popsizeI = s_params.popsizeI
    
    bgrE = s_params.bgrE
    bgrI = s_params.bgrI
    
    
    JplusEE = s_params.JplusEE
    JplusEI = s_params.JplusEI    
    JplusIE = s_params.JplusIE
    JplusII = s_params.JplusII
           
    externalNoise = s_params.extCurrent_poisson
    
    additional_externalVariance = a_params.additional_externalVariance
    
    nu_vec = a_params.nu_vec
    
    # check that number of clusters !=0
    if p==0:
        sys.exit('ERROR: # CLUSTERS CANT BE ZERO. SET BACKGROUND FRACTION=1 IF YOU WANT NO CLUSTER LIMIT')        
    # check that external inputs are all the same; if not, thrown an error
    if np.all(s_params.nu_ext_e == s_params.nu_ext_e[0]) == False:
        sys.exit('ERROR: not all external inputs are the same --> should not use this function')
    if np.all(s_params.nu_ext_i == s_params.nu_ext_i[0]) == False:
        sys.exit('ERROR: not all external inputs are the same --> should not use this function')
    
    # get depression factors
    JminusEE, JminusEI, JminusIE, JminusII = fcn_compute_depressFactors(\
                                            p, bgrE, bgrI, \
                                            JplusEE, JplusII, \
                                            JplusEI, JplusIE)
       
    
    n_e_pops = np.size(popsizeE)
    n_i_pops = np.size(popsizeI)
    
    fE = (1-bgrE)/p 
    fI = (1-bgrI)/p
    fEb = bgrE
    fIb = bgrI
    Jee_p = JplusEE*Jee
    Jee_m = JminusEE*Jee 
    Jei_p = JplusEI*Jei
    Jei_m = JminusEI*Jei 
    Jie_p = JplusIE*Jie
    Jie_m = JminusIE*Jie   
    Jii_p = JplusII*Jii
    Jii_m = JminusII*Jii  
    
    # SOLVE SELF CONSISTENT EQUATIONS VIA ROOT FINDING    
    sol = fcn_MFT_rate_roots(tau_r, tau_m_e, tau_m_i, tau_s_e, tau_s_i, \
                             Vr_e, Vr_i, Vth_e, Vth_i, nu_ext_e, nu_ext_i, \
                             Cee, Cei, Cie, Cii, Cext, \
                             Jee, Jei, Jie, Jii, Jee_ext, Jie_ext, \
                             Jee_p, Jee_m, Jei_p, Jei_m, Jie_p, Jie_m, Jii_p, Jii_m,
                             fE, fEb, fI, fIb, n_e_pops, n_i_pops, \
                             externalNoise, additional_externalVariance, \
                             nu_vec) 
    
    
    if sol.success == False:
        print('error in root finding!')
        return None, None, None, None

    # OUTPUT THE RATES
    nu_e = sol.x[:n_e_pops]
    nu_i = sol.x[n_e_pops:]
    
    
    # COMPUTE SELF-CONSISTENT MU AND SIGMA (SEE ER MFT CODE)
        
    Mu_e, Mu_i = fcn_compute_Mu(nu_e, nu_i, nu_ext_e, nu_ext_i, \
                   tau_m_e, tau_m_i, \
                   Cee, Cei, Cie, Cii, Cext, \
                   Jee, Jei, Jie, Jii, Jee_ext, Jie_ext, \
                   Jee_p, Jee_m, Jei_p, Jei_m, Jie_p, Jie_m, Jii_p, Jii_m, \
                   fE, fEb, fI, fIb, n_e_pops, n_i_pops)
    
    Sig2_e, Sig2_i = fcn_compute_Sigma2(nu_e, nu_i, nu_ext_e, nu_ext_i, \
                       tau_m_e, tau_m_i, \
                       Cee, Cei, Cie, Cii, Cext, \
                       Jee, Jei, Jie, Jii, Jee_ext, Jie_ext, \
                       Jee_p, Jee_m, Jei_p, Jei_m, Jie_p, Jie_m, Jii_p, Jii_m, \
                       fE, fEb, fI, fIb, n_e_pops, n_i_pops, \
                       externalNoise, additional_externalVariance)
    
    # COMPUTE RATES USING SELF-CONSISTENT MEAN AND VARIANCE
    
    nu_e_sc = np.zeros(n_e_pops)
    nu_i_sc = np.zeros(n_i_pops)
    for i in range(0,n_e_pops,1):
        nu_e_sc[i] = fcn_compute_rate(Vr_e, Vth_e, Mu_e[i], np.sqrt(Sig2_e[i]), \
                                   tau_r, tau_m_e, tau_s_e)
    for i in range(0,n_i_pops,1):
        nu_i_sc[i] = fcn_compute_rate(Vr_i, Vth_i, Mu_i[i], np.sqrt(Sig2_i[i]), \
                                   tau_r, tau_m_i, tau_s_i)    
    
    # VERIFY THAT THESE AGREE WITH THE OUTPUT RATES
    nu_e_check = all(abs(nu_e_sc-nu_e) < 1e-4)
    nu_i_check = all(abs(nu_i_sc-nu_i) < 1e-4)
        
    if (nu_e_check == True and nu_i_check == True):
        print('verified solution is self consistent.')
    else:
        sys.exit('ERROR: Solution is not self-consistent!')
    
   
    
    # COMPUTE STABILITY
    nu_fixed_point = np.hstack((nu_e, nu_i))
    
    S, eigenvals_S, realPart_eigvals_S  = \
                    fcn_stability_matrix(nu_fixed_point, \
                                         tau_r, tau_m_e, tau_m_i, tau_s_e, tau_s_i, \
                                         Vr_e, Vr_i, Vth_e, Vth_i, \
                                         nu_ext_e, nu_ext_i, \
                                         Cee, Cei, Cie, Cii, Cext, \
                                         Jee, Jei, Jie, Jii, Jee_ext, Jie_ext, \
                                         Jee_p, Jee_m, Jei_p, Jei_m, Jie_p, Jie_m, Jii_p, Jii_m,
                                         fE, fEb, fI, fIb, n_e_pops, n_i_pops, \
                                         externalNoise, additional_externalVariance)
                        
                        
    # RESULTS DICTIONARY
    results = {}
    results['nu_e'] = nu_e
    results['nu_i'] = nu_i
    results['Mu_e'] = Mu_e
    results['Sig2_e'] = Sig2_e
    results['Mu_i'] = Mu_i
    results['Sig2_i'] = Sig2_i
    results['realPart_eigvals_S'] = realPart_eigvals_S
    results['S'] = S
    results['a_params'] = a_params
                        
            
    return results



#%%
# MASTER MFT FUNCTION WHEN SOLVING WITH DYNAMICAL EQUATIONS    
def fcn_master_MFT_DynEqs(s_params, a_params):    
    
    
    tau_r = s_params.tau_r            # refractory period
    tau_m_e = s_params.tau_m_e        # membrane time constant E
    tau_m_i = s_params.tau_m_i        # membrane time constant I
    tau_s_e = s_params.tau_s_e        # synaptic time constant E
    tau_s_i = s_params.tau_s_i        # synaptic time constant I
    
    Vr_e = s_params.Vr_e              # reset potential E
    Vr_i = s_params.Vr_i              # reset potential E
    
    Vth_e = s_params.Vth_e            # threshold potential E
    Vth_i = s_params.Vth_i            # threshold potential I
    
    nu_ext_e = s_params.nu_ext_e[0]      # avg baseline afferent rate to E 
    nu_ext_i = s_params.nu_ext_i[0] 
    
    Cee = s_params.Cee 
    Cei = s_params.Cei 
    Cii = s_params.Cii 
    Cie = s_params.Cie 
    Cext = s_params.Cext 

    Jee = s_params.Jee 
    Jei = s_params.Jei 
    Jii = s_params.Jii 
    Jie = s_params.Jie 
    Jee_ext = s_params.Jee_ext        
    Jie_ext = s_params.Jie_ext 
    
    popsizeE = s_params.popsizeE
    popsizeI = s_params.popsizeI
    
    p = s_params.p
    bgrE = s_params.bgrE
    bgrI = s_params.bgrI
    JplusEE = s_params.JplusEE
    JplusEI = s_params.JplusEI    
    JplusIE = s_params.JplusIE
    JplusII = s_params.JplusII
           
    externalNoise = s_params.extCurrent_poisson
    
    additional_externalVariance = a_params.additional_externalVariance

    nu_vec = a_params.nu_vec
    
    # check that number of clusters !=0
    if p==0:
        sys.exit('ERROR: # CLUSTERS CANT BE ZERO. SET BACKGROUND FRACTION=1 IF YOU WANT NO CLUSTER LIMIT')
    
    # get depression factors
    JminusEE, JminusEI, JminusIE, JminusII = fcn_compute_depressFactors(\
                                            p, bgrE, bgrI, \
                                            JplusEE, s_params.JplusII, \
                                            s_params.JplusEI, s_params.JplusIE)
       
    n_e_pops = np.size(popsizeE)
    n_i_pops = np.size(popsizeI)

        
    fE = (1-bgrE)/p 
    fI = (1-bgrI)/p
    fEb = bgrE
    fIb = bgrI
    Jee_p = JplusEE*Jee
    Jee_m = JminusEE*Jee 
    Jei_p = JplusEI*Jei
    Jei_m = JminusEI*Jei 
    Jie_p = JplusIE*Jie
    Jie_m = JminusIE*Jie   
    Jii_p = JplusII*Jii
    Jii_m = JminusII*Jii  
    
    
    nSteps = a_params.nSteps_MFT_DynEqs
    dt = a_params.dt_MFT_DynEqs
    Te = a_params.tau_e_MFT_DynEqs
    Ti = a_params.tau_i_MFT_DynEqs
    stop_thresh = a_params.stopThresh_MFT_DynEqs
    plot = a_params.plot_MFT_DynEqs

    
    # COMPUTE RATES OF EACH E AND I POPULATION 
    nu_e, nu_i = fcn_compute_MFT_rates_DynEqs(nSteps, dt, Te, Ti, stop_thresh, plot, \
                                 tau_r, tau_m_e, tau_m_i, tau_s_e, tau_s_i, \
                                 Vr_e, Vr_i, Vth_e, Vth_i, nu_ext_e, nu_ext_i, \
                                 Cee, Cei, Cie, Cii, Cext, \
                                 Jee, Jei, Jie, Jii, Jee_ext, Jie_ext, \
                                 Jee_p, Jee_m, Jei_p, Jei_m, Jie_p, Jie_m, Jii_p, Jii_m, \
                                 fE, fEb, fI, fIb, n_e_pops, n_i_pops, \
                                 externalNoise, additional_externalVariance, nu_vec)
        
        
    # COMPUTE SELF-CONSISTENT MU AND SIGMA (SEE ER MFT CODE)
        
    Mu_e, Mu_i = fcn_compute_Mu(nu_e, nu_i, nu_ext_e, nu_ext_i, \
                   tau_m_e, tau_m_i, \
                   Cee, Cei, Cie, Cii, Cext, \
                   Jee, Jei, Jie, Jii, Jee_ext, Jie_ext, \
                   Jee_p, Jee_m, Jei_p, Jei_m, Jie_p, Jie_m, Jii_p, Jii_m, \
                   fE, fEb, fI, fIb, n_e_pops, n_i_pops)
    
    Sig2_e, Sig2_i = fcn_compute_Sigma2(nu_e, nu_i, nu_ext_e, nu_ext_i, \
                       tau_m_e, tau_m_i, \
                       Cee, Cei, Cie, Cii, Cext, \
                       Jee, Jei, Jie, Jii, Jee_ext, Jie_ext, \
                       Jee_p, Jee_m, Jei_p, Jei_m, Jie_p, Jie_m, Jii_p, Jii_m, \
                       fE, fEb, fI, fIb, n_e_pops, n_i_pops, \
                       externalNoise, additional_externalVariance)
    
    # COMPUTE RATES USING SELF-CONSISTENT MEAN AND VARIANCE
    
    nu_e_sc = np.zeros(n_e_pops)
    nu_i_sc = np.zeros(n_i_pops)
    for i in range(0,n_e_pops,1):
        nu_e_sc[i] = fcn_compute_rate(Vr_e, Vth_e, Mu_e[i], np.sqrt(Sig2_e[i]), \
                                   tau_r, tau_m_e, tau_s_e)
    for i in range(0,n_i_pops,1):
        nu_i_sc[i] = fcn_compute_rate(Vr_i, Vth_i, Mu_i[i], np.sqrt(Sig2_i[i]), \
                                   tau_r, tau_m_i, tau_s_i)    
    
    # VERIFY THAT THESE AGREE WITH THE OUTPUT RATES
    nu_e_check = all(abs(nu_e_sc-nu_e) < 1e-4)
    nu_i_check = all(abs(nu_i_sc-nu_i) < 1e-4)
        
    if (nu_e_check == True and nu_i_check == True):
        print('verified solution is self consistent.')
    else:
        sys.exit('ERROR: Solution is not self-consistent!')


    # COMPUTE STABILITY
    nu_fixed_point = np.hstack((nu_e, nu_i))
    
    S, eigenvals_S, realPart_eigvals_S  = \
                    fcn_stability_matrix(nu_fixed_point, \
                                         tau_r, tau_m_e, tau_m_i, tau_s_e, tau_s_i, \
                                         Vr_e, Vr_i, Vth_e, Vth_i, \
                                         nu_ext_e, nu_ext_i, \
                                         Cee, Cei, Cie, Cii, Cext, \
                                         Jee, Jei, Jie, Jii, Jee_ext, Jie_ext, \
                                         Jee_p, Jee_m, Jei_p, Jei_m, Jie_p, Jie_m, Jii_p, Jii_m,
                                         fE, fEb, fI, fIb, n_e_pops, n_i_pops, \
                                         externalNoise, additional_externalVariance)
    
    # RESULTS DICTIONARY
    results = {}
    results['nu_e'] = nu_e
    results['nu_i'] = nu_i
    results['realPart_eigvals_S'] = realPart_eigvals_S
    results['S'] = S
    results['a_params'] = a_params
                        
            
    return results
