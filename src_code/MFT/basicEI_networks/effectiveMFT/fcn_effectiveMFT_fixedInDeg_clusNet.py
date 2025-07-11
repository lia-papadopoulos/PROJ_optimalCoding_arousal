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
                       ext_variance):   
    
    
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
    sig2_vec =  sig2_recurrent_vec + sig2_external_vec
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
                                 ext_variance, \
                                 nu_vec_in, dynamicPops_e, dynamicPops_i):
    
    
    
    
#------------------------------------------------------------------------------
# SETUP
#------------------------------------------------------------------------------
    
    # initial conditions
    nu_eo = nu_vec_in[:n_e_pops]
    nu_io = nu_vec_in[n_e_pops:]
    
    
    # initialize
    nu_e = np.zeros((n_e_pops, nSteps+1))
    nu_i = np.zeros((n_i_pops, nSteps+1))
    
    converged = False
      
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
                                                ext_variance)
            
            
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
                  
                
        # update rates of dynamic populations
        for j in range(0, n_e_pops):
            
            if j in dynamicPops_e:
                
                nu_e[j,i+1] = nu_e[j,i] + (1/Te[j])*(-nu_e[j,i] + phi_e[j])*dt
                
            else:
                
                nu_e[j,i+1] = nu_e[j,i]

        for j in range(0, n_i_pops):
            
            if j in dynamicPops_i:
                
                nu_i[j,i+1] = nu_i[j,i] + (-nu_i[j,i]/Ti[j] + phi_i[j]/Ti[j])*dt
                
            else:
                
                nu_i[j,i+1] = nu_i[j,i]       
        
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
            
            converged = True
            
            # end loop
            break
        

    if converged == False:
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
# COMPUTE STATIONARY RATES BY SOLVING DYNAMICAL EQUATIONS FOR THE INPUT
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def fcn_compute_MFT_rates_DynEqs_inputs(nSteps, dt, Te, Ti, stop_thresh, plot, \
                                 tau_r, tau_m_e, tau_m_i, tau_s_e, tau_s_i, \
                                 Vr_e, Vr_i, Vth_e, Vth_i, nu_ext_e, nu_ext_i, \
                                 Cee, Cei, Cie, Cii, Cext, \
                                 Jee, Jei, Jie, Jii, Jee_ext, Jie_ext, \
                                 Jee_p, Jee_m, Jei_p, Jei_m, Jie_p, Jie_m, Jii_p, Jii_m, \
                                 fE, fEb, fI, fIb, n_e_pops, n_i_pops, \
                                 ext_variance, nu_vec_in, dynamicPops_e, dynamicPops_i):
    
    
    
    
#------------------------------------------------------------------------------
# SETUP
#------------------------------------------------------------------------------
    
    # initial conditions
    m_e = np.zeros((n_e_pops, nSteps+1))
    m_i = np.zeros((n_i_pops, nSteps+1))
    s2_e = np.zeros((n_e_pops, nSteps+1))
    s2_i = np.zeros((n_i_pops, nSteps+1))
    
    
    # initialize
    nu_e = np.zeros((n_e_pops, nSteps+1))
    nu_i = np.zeros((n_i_pops, nSteps+1))
    
    
      
#------------------------------------------------------------------------------
# MAIN LOOP
#------------------------------------------------------------------------------     

    # set initial conditions
    nu_e[:,0] = nu_vec_in[:n_e_pops]
    nu_i[:,0] = nu_vec_in[n_e_pops:]
    

    # compute mean of inputs
    Mu_e, Mu_i = fcn_compute_Mu(nu_e[:,0], nu_i[:,0], nu_ext_e, nu_ext_i, \
                                    tau_m_e, tau_m_i, \
                                    Cee, Cei, Cie, Cii, Cext, \
                                    Jee, Jei, Jie, Jii, Jee_ext, Jie_ext, \
                                    Jee_p, Jee_m, Jei_p, Jei_m, Jie_p, Jie_m, Jii_p, Jii_m,\
                                    fE, fEb, fI, fIb, n_e_pops, n_i_pops)
    
    
        
    # compute variance of inputs
    Sigma2_e, Sigma2_i = fcn_compute_Sigma2(nu_e[:,0], nu_i[:,0], nu_ext_e, nu_ext_i, \
                                                tau_m_e, tau_m_i, \
                                                Cee, Cei, Cie, Cii, Cext, \
                                                Jee, Jei, Jie, Jii, Jee_ext, Jie_ext, \
                                                Jee_p, Jee_m, Jei_p, Jei_m, Jie_p, Jie_m, Jii_p, Jii_m, \
                                                fE, fEb, fI, fIb, n_e_pops, n_i_pops,\
                                                ext_variance)
            
        
    # initial conditions for m, s2
    m_e[:, 0] = Mu_e
    m_i[:, 0] = Mu_i
    s2_e[:, 0] = 1/2*Sigma2_e
    s2_i[:,0] = 1/2*Sigma2_i
    
        
    
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
                                                ext_variance)
                       
                
        # update m and s2 of dynamic populations
        for j in range(0, n_e_pops):
            
            if j in dynamicPops_e:
                
                m_e[j,i+1] = m_e[j,i] + (1/tau_m_e)*(-m_e[j,i] + Mu_e[j])*dt
                s2_e[j,i+1] = s2_e[j,i] + (1/tau_m_e)*(-2*s2_e[j,i] + Sigma2_e[j])*dt
                
            else:
                
                m_e[j,i+1] = m_e[j,i]
                s2_e[j,i+1] = s2_e[j,i]
                

        for j in range(0, n_i_pops):
            
            if j in dynamicPops_i:
                
                m_i[j,i+1] = m_i[j,i] + (1/tau_m_i)*(-m_i[j,i] + Mu_i[j])*dt
                s2_i[j,i+1] = s2_i[j,i] + (1/tau_m_i)*(-2*s2_i[j,i] + Sigma2_i[j])*dt
                
                
            else:
                
                m_i[j,i+1] = m_i[j,i]
                s2_i[j,i+1] = s2_i[j,i]
        
        
        # compute output rates given m, s2
        
        phi_e = np.zeros(n_e_pops)
        phi_i = np.zeros(n_i_pops) 
        
        for epop_ind in range(0,n_e_pops,1):
            
            if epop_ind in dynamicPops_e:
                
                phi_e[epop_ind] = fcn_compute_rate(Vr_e, Vth_e, \
                                               m_e[epop_ind,i+1], np.sqrt(2*s2_e[epop_ind,i+1]), \
                                               tau_r, tau_m_e, tau_s_e)
                    
            else:
                
                phi_e[epop_ind] = nu_e[epop_ind, 0]
            
        for ipop_ind in range(0,n_i_pops,1):
            
            if ipop_ind in dynamicPops_i:
                
                phi_i[ipop_ind] = fcn_compute_rate(Vr_i, Vth_i, \
                                               m_i[ipop_ind,i+1], np.sqrt(2*s2_i[ipop_ind,i+1]), \
                                               tau_r, tau_m_i, tau_s_i)    
            else:
                
                phi_i[ipop_ind] = nu_i[ipop_ind, 0]
        
        
        
        
        # update rates
        nu_e[:,i+1] = phi_e
        nu_i[:,i+1] = phi_i        

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
def fcn_root_eqs(nu_vec_dynamicPops, nu_vec, \
                 tau_r, tau_m_e, tau_m_i, tau_s_e, tau_s_i, \
                 Vr_e, Vr_i, Vth_e, Vth_i, nu_ext_e, nu_ext_i, \
                 Cee, Cei, Cie, Cii, Cext, \
                 Jee, Jei, Jie, Jii, Jee_ext, Jie_ext, \
                 Jee_p, Jee_m, Jei_p, Jei_m, Jie_p, Jie_m, Jii_p, Jii_m, \
                 fE, fEb, fI, fIb, n_e_pops, n_i_pops, \
                 ext_variance, dynamicPops_e, dynamicPops_i):
    
    
    # update dynamic elements of nu_vec to match nu_vec_dynamicPops
    # other elements stay fixed
    for i in range(0, np.size(dynamicPops_e)):
        nu_vec[dynamicPops_e[i]] = nu_vec_dynamicPops[i]
        
    for i in range(0, np.size(dynamicPops_i)):
        nu_vec[n_e_pops + dynamicPops_i[i]] = nu_vec_dynamicPops[np.size(dynamicPops_e) + i]
    
      
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
                                            ext_variance)
        
    
        
    
    sigma_e = np.sqrt(Sigma2_e)
    sigma_i = np.sqrt(Sigma2_i)
    

          
    n_dynamicPops = np.size(dynamicPops_e) + np.size(dynamicPops_i)
    F = np.empty((n_dynamicPops))
    
    for count, i in enumerate(dynamicPops_e):
        F[count] = nu_vec[i] - fcn_compute_rate(Vr_e, Vth_e, Mu_e[i], sigma_e[i], \
                                                tau_r, tau_m_e, tau_s_e)
            
    for count, i in enumerate(dynamicPops_i):            
        F[count + np.size(dynamicPops_e)] = nu_vec[i+n_e_pops] - fcn_compute_rate(Vr_i, Vth_i, \
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
                        ext_variance, \
                        nu_vec_dynamicPops_in, nu_vec_in, dynamicPops_e, dynamicPops_i):
    
    
    # solve self-consistent equations
    sol = optimize.root(fcn_root_eqs, nu_vec_dynamicPops_in, \
                        args=(nu_vec_in, tau_r, tau_m_e, tau_m_i, tau_s_e, tau_s_i, \
                              Vr_e, Vr_i, Vth_e, Vth_i, nu_ext_e, nu_ext_i, \
                              Cee, Cei, Cie, Cii, Cext, \
                              Jee, Jei, Jie, Jii, Jee_ext, Jie_ext, \
                              Jee_p, Jee_m, Jei_p, Jei_m, Jie_p, Jie_m, Jii_p, Jii_m,
                              fE, fEb, fI, fIb, n_e_pops, n_i_pops, \
                              ext_variance, dynamicPops_e, dynamicPops_i),\
                        jac=False, method='hybr',
                        tol=1e-12,options={'xtol':1e-12})
                
        
    # return solution    
    return sol



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
    
    nu_vec = a_params.nu_vec
    nu_vec_original = nu_vec.copy()
    
    
    inFocus_pops_e = a_params.inFocus_pops_e
    outFocus_pops_e = a_params.outFocus_pops_e
    
    inFocus_pops_i = a_params.inFocus_pops_i
    outFocus_pops_i = a_params.outFocus_pops_i
    
    
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
    
    

    ##### COMPUTE OUT-OF-FOCUS RATES SELF-CONSISTENTLY #####
    
    
    # set initial guess for dynamic (out-of-focus populations)
    nu_vec_dynamicPops_in = np.zeros( np.size(outFocus_pops_e) + np.size(outFocus_pops_i) )
    
    for i in range(0, np.size(outFocus_pops_e)):
        nu_vec_dynamicPops_in[i] = nu_vec[outFocus_pops_e[i]]
        
    for i in range(0, np.size(outFocus_pops_i)):
        nu_vec_dynamicPops_in[i + np.size(outFocus_pops_e)] = nu_vec[n_e_pops + outFocus_pops_i[i]]
        
    
    #print('starting rates:') 
    #print(nu_vec[0],nu_vec[1],nu_vec[2],nu_vec[3])
    
        
    # SOLVE FOR SELF CONSISTENT RATES OF OUT-OF-FOCUS POPS FOR FIXED IN-FOCUS POPS
    
    sol = fcn_MFT_rate_roots(tau_r, tau_m_e, tau_m_i, tau_s_e, tau_s_i, \
                             Vr_e, Vr_i, Vth_e, Vth_i, nu_ext_e, nu_ext_i, \
                             Cee, Cei, Cie, Cii, Cext, \
                             Jee, Jei, Jie, Jii, Jee_ext, Jie_ext, \
                             Jee_p, Jee_m, Jei_p, Jei_m, Jie_p, Jie_m, Jii_p, Jii_m,
                             fE, fEb, fI, fIb, n_e_pops, n_i_pops, \
                             externalNoise, \
                             nu_vec_dynamicPops_in, nu_vec, outFocus_pops_e, outFocus_pops_i) 
    
    
    if sol.success == False:
        print('error in root finding outFocus!')
        return np.nan, np.nan, np.nan, np.nan
    
    print('out-of-focus rates found')

    # OUTPUT THE RATES
    if np.size(outFocus_pops_e)>0:
        outFocus_nu_e = sol.x[:np.size(outFocus_pops_e)]
    else:
        outFocus_nu_e = np.nan
    if np.size(outFocus_pops_i)>0:
        outFocus_nu_i = sol.x[np.size(outFocus_pops_e):]
    else:
        outFocus_nu_i = np.nan
    
    print(outFocus_nu_e, outFocus_nu_i)
    
    
    # SOLVE FOR NEW IN-FOCUS POPS DUE TO FEEDBACK FROM OUT-FOCUS
       
    # initial in-focus rates set to input values
    for i in range(0, np.size(inFocus_pops_e)):
        nu_vec[inFocus_pops_e[i]] = nu_vec_original[inFocus_pops_e[i]]
        
    for i in range(0, np.size(inFocus_pops_i)):
        nu_vec[n_e_pops + inFocus_pops_i[i]] = nu_vec_original[n_e_pops + inFocus_pops_i[i]]
    
        
    # update nu_vec with new out-of focus rates
    for i in range(0, np.size(outFocus_pops_e)):
        nu_vec[outFocus_pops_e[i]] = outFocus_nu_e[i]
        
    for i in range(0, np.size(outFocus_pops_i)):
        nu_vec[n_e_pops + outFocus_pops_i[i]] = outFocus_nu_i[i]
    
    #print('starting rates for in-focus:')
    #print(nu_vec[0],nu_vec[1],nu_vec[2],nu_vec[3])  
    
    
    # get e and i rates
    nu_e = nu_vec[:n_e_pops]
    nu_i = nu_vec[n_e_pops:]
    
    # compute mean & variance of input
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
                                                                externalNoise)
        
        
    # compute in-focus rates
    
    inFocus_nu_e_out = np.zeros(len(inFocus_pops_e))
    inFocus_nu_i_out = np.zeros(len(inFocus_pops_i))
    
    for i in range(0, len(inFocus_pops_e)):
    
        popInd = inFocus_pops_e[i]
        
        inFocus_nu_e_out[popInd] = fcn_compute_rate(Vr_e, Vth_e, Mu_e[popInd], np.sqrt(Sig2_e[popInd]), \
                                                    tau_r, tau_m_e, tau_s_e)
    
        
    for i in range(0, len(inFocus_pops_i)):
    
        popInd = inFocus_pops_i[i]
        
        inFocus_nu_i_out[popInd] = fcn_compute_rate(Vr_i, Vth_i, Mu_i[popInd], np.sqrt(Sig2_i[popInd]), \
                                                    tau_r, tau_m_i, tau_s_i)         



            
    # return all the rates        
    return inFocus_nu_e_out, inFocus_nu_i_out, outFocus_nu_e, outFocus_nu_i



#%% MASTER MFT FUNCTION WHEN SOLVING WITH DYNAMICAL EQUATIONS    

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
    
    nu_vec = a_params.nu_vec
    nu_vec_original = nu_vec.copy()
    
    inFocus_pops_e = a_params.inFocus_pops_e
    outFocus_pops_e = a_params.outFocus_pops_e
    
    inFocus_pops_i = a_params.inFocus_pops_i
    outFocus_pops_i = a_params.outFocus_pops_i
    
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
    
    
    # SOLVE FOR SELF CONSISTENT RATES OF OUT-OF-FOCUS POPS FOR FIXED IN-FOCUS POPS


    ##### COMPUTE OUT-OF-FOCUS RATES SELF-CONSISTENTLY #####

    print('starting rates:') 
    print(nu_vec)
    
    nu_e, nu_i = fcn_compute_MFT_rates_DynEqs(nSteps, dt, Te, Ti, stop_thresh, plot, \
                                 tau_r, tau_m_e, tau_m_i, tau_s_e, tau_s_i, \
                                 Vr_e, Vr_i, Vth_e, Vth_i, nu_ext_e, nu_ext_i, \
                                 Cee, Cei, Cie, Cii, Cext, \
                                 Jee, Jei, Jie, Jii, Jee_ext, Jie_ext, \
                                 Jee_p, Jee_m, Jei_p, Jei_m, Jie_p, Jie_m, Jii_p, Jii_m, \
                                 fE, fEb, fI, fIb, n_e_pops, n_i_pops, externalNoise, \
                                 nu_vec, outFocus_pops_e, outFocus_pops_i)
        

    
    # OUTPUT THE RATES
    if np.size(outFocus_pops_e)>0:
        outFocus_nu_e = nu_e[outFocus_pops_e]
    else:
        outFocus_nu_e = np.nan
    if np.size(outFocus_pops_i)>0:
        outFocus_nu_i = nu_i[outFocus_pops_i]
    else:
        outFocus_nu_i = np.nan
    
    
    print('out of focus rates found:')
    print(outFocus_nu_e, outFocus_nu_i)
        
    

    # SOLVE FOR NEW IN-FOCUS POPS DUE TO FEEDBACK FROM OUT-FOCUS
        

    # in-focus rates set to input values
    for i in range(0, np.size(inFocus_pops_e)):
        nu_vec[inFocus_pops_e[i]] = nu_vec_original[inFocus_pops_e[i]]
        
    for i in range(0, np.size(inFocus_pops_i)):
        nu_vec[n_e_pops + inFocus_pops_i[i]] = nu_vec_original[n_e_pops + inFocus_pops_i[i]]
    
        
    # update nu_vec with new out-of focus rates
    for i in range(0, np.size(outFocus_pops_e)):
        nu_vec[outFocus_pops_e[i]] = outFocus_nu_e[i]
        
    for i in range(0, np.size(outFocus_pops_i)):
        nu_vec[n_e_pops + outFocus_pops_i[i]] = outFocus_nu_i[i]
        
    
    print('starting rates for in-focus:')
    print(nu_vec[0],nu_vec[1],nu_vec[2],nu_vec[3])


    # get e and i rates
    nu_e = nu_vec[:n_e_pops]
    nu_i = nu_vec[n_e_pops:]


    # compute mean & variance of input
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
                                                                externalNoise)

    # compute in-focus rates
    inFocus_nu_e_out = np.zeros(len(inFocus_pops_e))
    inFocus_nu_i_out = np.zeros(len(inFocus_pops_i))
    
    for i in range(0, len(inFocus_pops_e)):
    
        popInd = inFocus_pops_e[i]
        
        inFocus_nu_e_out[popInd] = fcn_compute_rate(Vr_e, Vth_e, Mu_e[popInd], np.sqrt(Sig2_e[popInd]), \
                                                    tau_r, tau_m_e, tau_s_e)
    
        
    for i in range(0, len(inFocus_pops_i)):
    
        popInd = inFocus_pops_i[i]
        
        inFocus_nu_i_out[popInd] = fcn_compute_rate(Vr_i, Vth_i, Mu_i[popInd], np.sqrt(Sig2_i[popInd]), \
                                                    tau_r, tau_m_i, tau_s_i) 


    # return rates
    return inFocus_nu_e_out, inFocus_nu_i_out, outFocus_nu_e, outFocus_nu_i


