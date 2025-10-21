
"""
set of functions to run MFT for E-I networks with potentially clustered architecture
full mean field theory for a clustered network with p clusters
looks for solutions where q clusters are active and q-p are inactive
"""

import sys
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

import fcn_MFT_fixedInDeg_generalNet
import fcn_MFT_clusteredEINetworks_tools

import settings


sys.path.append(settings.network_generation_path)  
import fcn_make_network_cluster



#%%
#------------------------------------------------------------------------------  
#------------------------------------------------------------------------------  
# COMPUTE WITH ROOT FINDING (JACOBIAN NUMERICALLY ESTIMATED)
#------------------------------------------------------------------------------  
#------------------------------------------------------------------------------ 

# DEFINE ROOT EQUATION
def fcn_root_eqs_reduced(nu_vec, \
                         tau_r, tau_m, tau_s, \
                             Vr, Vth,  nu_ext, \
                                 Jab_reduced, Cab_reduced, \
                                     Jab_ext_reduced, Cab_ext_reduced, \
                                         ext_variance):
      
        
    # total number of populations
    n_pops = np.size(nu_vec)
    
    # compute augmented version of state vectors for computing correct mean and variance of reduced system
    nu_augmented = fcn_MFT_clusteredEINetworks_tools.fcn_compute_augmented_nuVec(nu_vec)
    
    # compute mean and sd               
    Mu = fcn_MFT_fixedInDeg_generalNet.fcn_compute_Mu(nu_augmented, nu_ext, \
                        Jab_reduced, Cab_reduced, \
                            Jab_ext_reduced, Cab_ext_reduced, tau_m)
        
    
    Sigma2 = fcn_MFT_fixedInDeg_generalNet.fcn_compute_Sigma2(nu_augmented, nu_ext, Jab_reduced, Cab_reduced, \
                                Jab_ext_reduced, Cab_ext_reduced, tau_m, ext_variance)

    sigma = np.sqrt(Sigma2)
      
    
    F = np.empty((n_pops))
    
    for i in range(0,n_pops):
        F[i] = nu_vec[i] - fcn_MFT_fixedInDeg_generalNet.fcn_compute_rate(Vr[i], Vth[i], Mu[i], sigma[i], \
                                            tau_r[i], tau_m[i], tau_s[i])

    
    Fvec = np.ndarray.tolist(F)
        
    return Fvec



# SOLVE ROOT EQUATION    
def fcn_MFT_rate_roots_reduced(nu_vec_in, nu_ext, \
                               Cab_reduced, Jab_reduced, \
                                   Jab_ext_reduced, Cab_ext_reduced, ext_variance, \
                                       tau_r, tau_m, tau_s, Vr, Vth):
    
    # solve self-consistent equations
    sol = optimize.root(fcn_root_eqs_reduced, nu_vec_in, \
                        args=(tau_r, tau_m, tau_s, Vr, Vth, nu_ext, \
                                  Jab_reduced, Cab_reduced, \
                                      Jab_ext_reduced, Cab_ext_reduced, ext_variance),\
                        jac=False, method='hybr',
                        tol=1e-12,options={'xtol':1e-12})
        
    
        
        
    # return solution    
    return sol


#%%       
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# COMPUTE STATIONARY RATES BY SOLVING DYNAMICAL EQUATIONS
#
# all parameters (eg time constants, threshold voltages etc should be vectors of length = # dynamical populations) 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def fcn_compute_MFT_rates_DynEqs_reduced(nSteps, dt, T, stop_thresh, plot, \
                                             tau_r, tau_m, tau_s, Vr,  Vth,  \
                                                     Jab_reduced, Cab_reduced, \
                                                         Jab_ext_reduced, Cab_ext_reduced, \
                                                             ext_variance, nu_ext, nu_vec_in):
    
    
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
        
        # compute augmented version of state vectors for computing correct mean and variance of reduced system
        nu_augmented = fcn_MFT_clusteredEINetworks_tools.fcn_compute_augmented_nuVec(nu[:, i])
        
        # compute mean of inputs
        Mu = fcn_MFT_fixedInDeg_generalNet.fcn_compute_Mu(nu_augmented, nu_ext, Jab_reduced, Cab_reduced, Jab_ext_reduced, Cab_ext_reduced, tau_m)
    
        # compute variance of inputs
        Sigma2 = fcn_MFT_fixedInDeg_generalNet.fcn_compute_Sigma2(nu_augmented, nu_ext, Jab_reduced, Cab_reduced, Jab_ext_reduced, Cab_ext_reduced, tau_m, ext_variance)
            
        # compute standard deviations
        sigma = np.sqrt(Sigma2)
        
        # compute output rates
        phi = np.zeros(n_pops)
        
        
        for pop_ind in range(0,n_pops):
            
            phi[pop_ind] = fcn_MFT_fixedInDeg_generalNet.fcn_compute_rate(Vr[pop_ind], Vth[pop_ind], \
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

'''
master function for computing mean-field solution
'''

def fcn_master_MFT_fixedInDeg_EI_cluster_net(s_params, a_params):    
    
    
    #-------------------------------------------------------------------------#            
    #---------------- LIF PARAMETERS -----------------------------------------#
    #-------------------------------------------------------------------------#        

    
    tau_r = s_params.tau_r              # refractory period
    tau_m_e = s_params.tau_m_e          # membrane time constant E
    tau_m_i = s_params.tau_m_i          # membrane time constant I
    tau_s_e = s_params.tau_s_e          # synaptic time constant E
    tau_s_i = s_params.tau_s_i          # synaptic time constant I
    
    Vr_e = s_params.Vr_e                # reset potential E
    Vr_i = s_params.Vr_i                # reset potential E
    
    Vth_e = s_params.Vth_e              # threshold potential E
    Vth_i = s_params.Vth_i              # threshold potential I
    
    if hasattr(s_params, 'nu_ext_e'):
        nu_ext_e = s_params.nu_ext_e[0]      # avg baseline afferent rate to E 
        if np.all(s_params.nu_ext_e == s_params.nu_ext_e[0]) == False:
            sys.exit('ERROR: not all external inputs are the same --> should not use this function')

    elif hasattr(s_params, 'nu_ext_ee'):
        nu_ext_e = s_params.nu_ext_ee[0]
        if np.all(s_params.nu_ext_ee == s_params.nu_ext_ee[0]) == False:
            sys.exit('ERROR: not all external inputs are the same --> should not use this function')
    else:
        sys.exit('exiting')
        
    
    if hasattr(s_params, 'nu_ext_i'):
        nu_ext_i = s_params.nu_ext_i[0]      # avg baseline afferent rate to E 
        if np.all(s_params.nu_ext_i == s_params.nu_ext_i[0]) == False:
            sys.exit('ERROR: not all external inputs are the same --> should not use this function')

    elif hasattr(s_params, 'nu_ext_ie'):
        nu_ext_i = s_params.nu_ext_ie[0]
        if np.all(s_params.nu_ext_ie == s_params.nu_ext_ie[0]) == False:
            sys.exit('ERROR: not all external inputs are the same --> should not use this function')
    else:
        sys.exit('exiting')
    
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
    
    bgrE = s_params.bgrE
    bgrI = s_params.bgrI
    
    
    JplusEE = s_params.JplusEE
    JplusEI = s_params.JplusEI    
    JplusIE = s_params.JplusIE
    JplusII = s_params.JplusII
           
    if hasattr(s_params, 'extCurrent_poisson'):
        externalNoise = s_params.extCurrent_poisson
    elif hasattr(s_params, 'base_extCurrent_poisson'):
        externalNoise = s_params.base_extCurrent_poisson
    else:
        sys.exit()    
    #-------------------------------------------------------------------------#        
    #---------------- MFT PARAMETERS -----------------------------------------#
    #-------------------------------------------------------------------------#        
   
    nu_vec = a_params.nu_vec
    n_activeClusters = a_params.n_active_clusters
    stability_tau_e = a_params.stability_tau_e
    stability_tau_i = a_params.stability_tau_i
    
    #-------------------------------------------------------------------------#            
    #---------------- MFT CHECKS ---------------------------------------------#
    #-------------------------------------------------------------------------#        
    
    if p < 2:
        sys.exit('ERROR: number of clusters must be >=2. Set Jplus=1 if you want no cluster limit')


    #-------------------------------------------------------------------------#        
    #---------------- J PLUS AND MINUS PARAMETERS ----------------------------#
    #-------------------------------------------------------------------------#        


    # depression factors
    if hasattr(s_params, 'JminusEE'):
        JminusEE = s_params.JminusEE
    if hasattr(s_params, 'JminusEI'):
        JminusEI = s_params.JminusEI
    if hasattr(s_params, 'JminusIE'):
        JminusIE = s_params.JminusIE
    if hasattr(s_params, 'JminusII'):
        JminusII = s_params.JminusII       
    else:
        JminusEE, JminusEI, JminusIE, JminusII = fcn_make_network_cluster.fcn_compute_depressFactors(s_params)
        
    # total number of E and I pops
    n_e_pops = p+1
    n_i_pops = p+1
    
    # cluster information
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
            
    
    #-------------------------------------------------------------------------#        
    #---------------- SET NUMBER OF DYNAMICAL POPULATIONS --------------------#
    #-------------------------------------------------------------------------#
    
    
    # REDUCED METHOD
    if ( (hasattr(a_params, 'solve_reduced')) and (a_params.solve_reduced==True) ):
        
        n_dPops_e = 3
        n_dPops_i = 3
        n_dPops = n_dPops_e + n_dPops_i
                
        if np.size(nu_vec) > n_dPops:
            nu_vec = nu_vec[ np.array([0, n_activeClusters, n_e_pops-1, n_e_pops, n_e_pops+n_activeClusters, -1]) ]        

            
    # FULL METHOD
    else:

        # total number of dynamical populations
        n_dPops_e = n_e_pops
        n_dPops_i = n_i_pops
        n_dPops = n_e_pops + n_i_pops
    
    
    # vectorize all parameters
    tau_r_vec = tau_r*np.ones(n_dPops)
    tau_m_vec = np.append(tau_m_e*np.ones(n_dPops_e), tau_m_i*np.ones(n_dPops_i))
    tau_s_vec = np.append(tau_s_e*np.ones(n_dPops_e), tau_s_i*np.ones(n_dPops_i))
    Vr_vec = np.append(Vr_e*np.ones(n_dPops_e), Vr_i*np.ones(n_dPops_i))
    Vth_vec = np.append(Vth_e*np.ones(n_dPops_e), Vth_i*np.ones(n_dPops_i))    
    nu_ext = np.append(nu_ext_e*np.ones(n_dPops_e), nu_ext_i*np.ones(n_dPops_i))
    
    stability_tau_vec = np.append(stability_tau_e*np.ones(n_e_pops), stability_tau_i*np.ones(n_i_pops)) 
    
    
    #-------------------------------------------------------------------------#        
    #---------------- SOLVE SELF-CONSISTENT EQUATIONS VIA ROOT FINDING -------#
    #-------------------------------------------------------------------------# 
    
    # REDUCED METHOD
    if ( (hasattr(a_params, 'solve_reduced')) and (a_params.solve_reduced==True) ):
        
        # compute Jab and Cab matrices
        Jab, Cab, Jab_ext, Cab_ext = fcn_MFT_clusteredEINetworks_tools.fcn_compute_weight_degree_mat_reduced(\
                                                                   Cee, Cei, Cie, Cii, Cext, \
                                                                       Jee, Jei, Jie, Jii, Jee_ext, Jie_ext, \
                                                                           Jee_p, Jee_m, Jei_p, Jei_m, Jie_p, Jie_m, Jii_p, Jii_m, \
                                                                               fE, fEb, fI, fIb, p, n_activeClusters)    
        
            
        
           
    
        # solution
        sol = fcn_MFT_rate_roots_reduced(nu_vec, nu_ext, \
                                             Cab, Jab, Jab_ext, Cab_ext, externalNoise, \
                                                 tau_r_vec, tau_m_vec, tau_s_vec, Vr_vec, Vth_vec)        
            
        
    
    
    # FULL METHOD
    else:
    
        # compute Jab and Cab matrices
        Jab, Cab, Jab_ext, Cab_ext = fcn_MFT_clusteredEINetworks_tools.fcn_compute_weight_degree_mat(\
                                                                   Cee, Cei, Cie, Cii, Cext, \
                                                                   Jee, Jei, Jie, Jii, Jee_ext, Jie_ext, \
                                                                   Jee_p, Jee_m, Jei_p, Jei_m, Jie_p, Jie_m, Jii_p, Jii_m, \
                                                                   fE, fEb, fI, fIb, n_dPops_e, n_dPops_i)    
        
            
        
           
    
        # solution
        sol = fcn_MFT_fixedInDeg_generalNet.fcn_MFT_rate_roots(nu_vec, nu_ext, \
                                                               Cab, Jab, Jab_ext, Cab_ext, externalNoise, \
                                                               tau_r_vec, tau_m_vec, tau_s_vec, Vr_vec, Vth_vec) 
           
            
    # check solution
    if sol.success == False:
    
        print('error in root finding!')
        
        # results dictionary
        results = {}
        results['nu_out'] = np.nan*np.ones(n_dPops)
        results['realPart_eigvals_S'] = np.nan
        results['S'] = np.nan
        results['Jab'] = Jab
        results['Cab'] = Cab
        results['Jab_ext'] = Jab_ext
        results['Cab_ext'] = Cab_ext
        results['a_params'] = a_params
        
        return results
    

    # output the rates
    nu_e_out = sol.x[:n_dPops_e]
    nu_i_out = sol.x[n_dPops_e:]
    nu_out = np.append(nu_e_out, nu_i_out)
    
    
    #-------------------------------------------------------------------------#        
    #----------------COMPUTE SELF-CONSISTENT MU AND SIGMA --------------------#
    #-------------------------------------------------------------------------#
    
    # REDUCED METHOD
    if ( (hasattr(a_params,'solve_reduced')) and (a_params.solve_reduced==True) ):

        nu_augmented = fcn_MFT_clusteredEINetworks_tools.fcn_compute_augmented_nuVec(nu_out)
        Mu = fcn_MFT_fixedInDeg_generalNet.fcn_compute_Mu(nu_augmented, nu_ext, Jab, Cab, Jab_ext, Cab_ext, tau_m_vec)
        Sigma2 = fcn_MFT_fixedInDeg_generalNet.fcn_compute_Sigma2(nu_augmented, nu_ext, Jab, Cab, Jab_ext, Cab_ext, tau_m_vec, externalNoise)
        
    
    
    # FULL METHOD
    else:
    
        Mu = fcn_MFT_fixedInDeg_generalNet.fcn_compute_Mu(nu_out, nu_ext, Jab, Cab, Jab_ext, Cab_ext, tau_m_vec)
        Sigma2 = fcn_MFT_fixedInDeg_generalNet.fcn_compute_Sigma2(nu_out, nu_ext, Jab, Cab, Jab_ext, Cab_ext, tau_m_vec, externalNoise)
        
        
    #-------------------------------------------------------------------------#        
    #----------------COMPUTE RATES USING SELF-CONSISTENT MU AND SIGMA2 -------#
    #-------------------------------------------------------------------------#
    
    # self-consistent rates
    nu_sc = np.zeros(n_dPops)
    
    for i in range(0,n_dPops):
        
        nu_sc[i] = fcn_MFT_fixedInDeg_generalNet.fcn_compute_rate(Vr_vec[i], Vth_vec[i], Mu[i], np.sqrt(Sigma2[i]), \
                                                                  tau_r_vec[i], tau_m_vec[i], tau_s_vec[i])  
    
    # verify that rates are consistent
    nu_check = all(abs(nu_sc - nu_out) < 1e-4)
        
    if (nu_check == True):
        print('verified solution is self consistent.')
    else:
        sys.exit('ERROR: Solution is not self-consistent!')
        

    #-------------------------------------------------------------------------#        
    #----------------IF USING REDUCED METHOD, EXTEND TO FULL SOLUTION --------#
    #-------------------------------------------------------------------------#
    if ( (a_params.solve_reduced==True) ):
        
    
        nu_out = fcn_MFT_clusteredEINetworks_tools.fcn_compute_full_vector_from_reduced(nu_out, p, n_activeClusters)
        Mu = fcn_MFT_clusteredEINetworks_tools.fcn_compute_full_vector_from_reduced(Mu, p, n_activeClusters)
        Sigma2 = fcn_MFT_clusteredEINetworks_tools.fcn_compute_full_vector_from_reduced(Sigma2, p, n_activeClusters)

    
    #-------------------------------------------------------------------------#        
    #----------------COMPUTE STABILITY ---------------------------------------#
    #-------------------------------------------------------------------------#
    if ( (a_params.solve_reduced==True) ):
        
        
        n_dPops_e = n_e_pops
        n_dPops_i = n_i_pops
        n_dPops = n_e_pops + n_i_pops
        
        # vectorize all parameters
        tau_r_vec = tau_r*np.ones(n_dPops)
        tau_m_vec = np.append(tau_m_e*np.ones(n_dPops_e), tau_m_i*np.ones(n_dPops_i))
        tau_s_vec = np.append(tau_s_e*np.ones(n_dPops_e), tau_s_i*np.ones(n_dPops_i))
        Vr_vec = np.append(Vr_e*np.ones(n_dPops_e), Vr_i*np.ones(n_dPops_i))
        Vth_vec = np.append(Vth_e*np.ones(n_dPops_e), Vth_i*np.ones(n_dPops_i))    
        nu_ext = np.append(nu_ext_e*np.ones(n_dPops_e), nu_ext_i*np.ones(n_dPops_i))
        
            
        # compute Jab and Cab matrices
        Jab_full, Cab_full, Jab_ext_full, Cab_ext_full = fcn_MFT_clusteredEINetworks_tools.fcn_compute_weight_degree_mat(\
                                                                   Cee, Cei, Cie, Cii, Cext, \
                                                                   Jee, Jei, Jie, Jii, Jee_ext, Jie_ext, \
                                                                   Jee_p, Jee_m, Jei_p, Jei_m, Jie_p, Jie_m, Jii_p, Jii_m, \
                                                                   fE, fEb, fI, fIb, n_dPops_e, n_dPops_i)  
            
        
        # stability    
        S, eigenvals_S, realPart_eigvals_S  = fcn_MFT_fixedInDeg_generalNet.fcn_stability_matrix(nu_out, \
                                                                                                 tau_r_vec, tau_m_vec, tau_s_vec, \
                                                                                                     Vr_vec, Vth_vec, nu_ext, \
                                                                                                         Jab_full, Cab_full, Jab_ext_full, Cab_ext_full, externalNoise)
            
        # stability alternate   
        S_alt, eigenvals_S_alt, realPart_eigvals_S_alt  = fcn_MFT_fixedInDeg_generalNet.fcn_stability_matrix_alternate(nu_out, \
                                                                                                 tau_r_vec, tau_m_vec, tau_s_vec, \
                                                                                                     Vr_vec, Vth_vec, nu_ext, \
                                                                                                         Jab_full, Cab_full, Jab_ext_full, Cab_ext_full, externalNoise, \
                                                                                                             stability_tau_vec)            
    
    
    else:
    
        S, eigenvals_S, realPart_eigvals_S  = fcn_MFT_fixedInDeg_generalNet.fcn_stability_matrix(nu_out, \
                                                                                                 tau_r_vec, tau_m_vec, tau_s_vec, \
                                                                                                     Vr_vec, Vth_vec, nu_ext, \
                                                                                                         Jab, Cab, Jab_ext, Cab_ext, externalNoise)


        S_alt, eigenvals_S_alt, realPart_eigvals_S_alt  = fcn_MFT_fixedInDeg_generalNet.fcn_stability_matrix_alternate(nu_out, \
                                                                                                 tau_r_vec, tau_m_vec, tau_s_vec, \
                                                                                                     Vr_vec, Vth_vec, nu_ext, \
                                                                                                         Jab, Cab, Jab_ext, Cab_ext, externalNoise, \
                                                                                                             stability_tau_vec)
                        
                        
    # RESULTS DICTIONARY
    results = {}
    results['Jab'] = Jab
    results['Cab'] = Cab
    results['Jab_ext'] = Jab_ext
    results['Cab_ext'] = Cab_ext
    results['nu_out'] = nu_out
    results['realPart_eigvals_S'] = realPart_eigvals_S
    results['S'] = S
    results['realPart_eigvals_S_alt'] = realPart_eigvals_S_alt
    results['S_alt'] = S_alt
    results['Mu'] = Mu
    results['Sigma2'] = Sigma2
    results['a_params'] = a_params
                        
            
    return results


#%%


# MASTER MFT FUNCTION WHEN SOLVING WITH DYNAMICAL EQUATIONS    
def fcn_master_MFT_DynEqs_fixedInDeg_EI_cluster_net(s_params, a_params):    
    
    
    tau_r = s_params.tau_r            # refractory period
    tau_m_e = s_params.tau_m_e        # membrane time constant E
    tau_m_i = s_params.tau_m_i        # membrane time constant I
    tau_s_e = s_params.tau_s_e        # synaptic time constant E
    tau_s_i = s_params.tau_s_i        # synaptic time constant I
    
    Vr_e = s_params.Vr_e              # reset potential E
    Vr_i = s_params.Vr_i              # reset potential E
    
    Vth_e = s_params.Vth_e            # threshold potential E
    Vth_i = s_params.Vth_i            # threshold potential I
    
    if hasattr(s_params, 'nu_ext_e'):
        nu_ext_e = s_params.nu_ext_e[0]      # avg baseline afferent rate to E 
        if np.all(s_params.nu_ext_e == s_params.nu_ext_e[0]) == False:
            sys.exit('ERROR: not all external inputs are the same --> should not use this function')

    elif hasattr(s_params, 'nu_ext_ee'):
        nu_ext_e = s_params.nu_ext_ee[0]
        if np.all(s_params.nu_ext_ee == s_params.nu_ext_ee[0]) == False:
            sys.exit('ERROR: not all external inputs are the same --> should not use this function')
    else:
        sys.exit()
        
    
    if hasattr(s_params, 'nu_ext_i'):
        nu_ext_i = s_params.nu_ext_i[0]      # avg baseline afferent rate to E 
        if np.all(s_params.nu_ext_i == s_params.nu_ext_i[0]) == False:
            sys.exit('ERROR: not all external inputs are the same --> should not use this function')

    elif hasattr(s_params, 'nu_ext_ie'):
        nu_ext_i = s_params.nu_ext_ie[0]
        if np.all(s_params.nu_ext_ie == s_params.nu_ext_ie[0]) == False:
            sys.exit('ERROR: not all external inputs are the same --> should not use this function')
    else:
        sys.exit()
        
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
    bgrE = s_params.bgrE
    bgrI = s_params.bgrI
    JplusEE = s_params.JplusEE
    JplusEI = s_params.JplusEI    
    JplusIE = s_params.JplusIE
    JplusII = s_params.JplusII
           
    if hasattr(s_params, 'extCurrent_poisson'):
        externalNoise = s_params.extCurrent_poisson
    elif hasattr(s_params, 'base_extCurrent_poisson'):
        externalNoise = s_params.base_extCurrent_poisson
    else:
        sys.exit()    
    #-------------------------------------------------------------------------#        
    #---------------- MFT PARAMETERS -----------------------------------------#
    #-------------------------------------------------------------------------#        
   
    nu_vec = a_params.nu_vec
    n_activeClusters = a_params.n_active_clusters
    stability_tau_e = a_params.stability_tau_e
    stability_tau_i = a_params.stability_tau_i

    nSteps = a_params.nSteps_MFT_DynEqs
    dt = a_params.dt_MFT_DynEqs
    Te = a_params.tau_e_MFT_DynEqs
    Ti = a_params.tau_i_MFT_DynEqs
    stop_thresh = a_params.stopThresh_MFT_DynEqs
    plot = a_params.plot_MFT_DynEqs
    

    #-------------------------------------------------------------------------#            
    #---------------- MFT CHECKS ---------------------------------------------#
    #-------------------------------------------------------------------------#        
    
    if p < 2:
        sys.exit('ERROR: number of clusters must be >=2. Set Jplus=1 if you want no cluster limit')


    #-------------------------------------------------------------------------#        
    #---------------- SET UP Jab and Cab matrices -----------------------------#
    #-------------------------------------------------------------------------#        


    # depression factors
    # depression factors
    if hasattr(s_params, 'JminusEE'):
        JminusEE = s_params.JminusEE
    if hasattr(s_params, 'JminusEI'):
        JminusEI = s_params.JminusEI
    if hasattr(s_params, 'JminusIE'):
        JminusIE = s_params.JminusIE
    if hasattr(s_params, 'JminusII'):
        JminusII = s_params.JminusII       
    else:
        JminusEE, JminusEI, JminusIE, JminusII = fcn_make_network_cluster.fcn_compute_depressFactors(s_params)    # total number of E and I pops
        
    n_e_pops = p+1
    n_i_pops = p+1
    
    # cluster information
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
            
    
    #-------------------------------------------------------------------------#        
    #---------------- SET NUMBER OF DYNAMICAL POPULATIONS --------------------#
    #-------------------------------------------------------------------------#
    
    
    # REDUCED METHOD
    if ( (hasattr(a_params, 'solve_reduced')) and (a_params.solve_reduced==True) ):
        
        n_dPops_e = 3
        n_dPops_i = 3
        n_dPops = n_dPops_e + n_dPops_i
        
        
        if np.size(nu_vec) > n_dPops:
            nu_vec = nu_vec[ np.array([0, n_activeClusters, n_e_pops-1, n_e_pops, n_e_pops+n_activeClusters, -1]) ]        

            
    # FULL METHOD
    else:

        # total number of dynamical populations
        n_dPops_e = n_e_pops
        n_dPops_i = n_i_pops
        n_dPops = n_e_pops + n_i_pops
        
        
    
    # vectorize all parameters
    tau_r_vec = tau_r*np.ones(n_dPops)
    tau_m_vec = np.append(tau_m_e*np.ones(n_dPops_e), tau_m_i*np.ones(n_dPops_i))
    tau_s_vec = np.append(tau_s_e*np.ones(n_dPops_e), tau_s_i*np.ones(n_dPops_i))
    Vr_vec = np.append(Vr_e*np.ones(n_dPops_e), Vr_i*np.ones(n_dPops_i))
    Vth_vec = np.append(Vth_e*np.ones(n_dPops_e), Vth_i*np.ones(n_dPops_i))    
    nu_ext = np.append(nu_ext_e*np.ones(n_dPops_e), nu_ext_i*np.ones(n_dPops_i))
    
    stability_tau_vec = np.append(stability_tau_e*np.ones(n_e_pops), stability_tau_i*np.ones(n_i_pops)) 

    
    if ( ( np.size(Te) == 1 ) ):
        Te = Te*np.ones(n_dPops_e)
    if ( ( np.size(Ti) == 1 ) ):
        Ti = Ti*np.ones(n_dPops_i)
    if ( ( np.size(Te)>n_dPops ) ):
        Te = Te[0]*np.ones(n_dPops_e)
    if ( ( np.size(Ti)>n_dPops ) ):
        Ti = Ti[0]*np.ones(n_dPops_i)
        
    T = np.append(Te, Ti)
    
    
    #-------------------------------------------------------------------------#        
    #---------------- SOLVE SELF-CONSISTENT EQUATIONS VIA DYNAMICAL EQS ------#
    #-------------------------------------------------------------------------# 
        
    # REDUCED METHOD
    if ( (hasattr(a_params, 'solve_reduced')) and (a_params.solve_reduced==True) ):
        
        # compute Jab and Cab matrices
        Jab, Cab, Jab_ext, Cab_ext = fcn_MFT_clusteredEINetworks_tools.fcn_compute_weight_degree_mat_reduced(\
                                                                   Cee, Cei, Cie, Cii, Cext, \
                                                                       Jee, Jei, Jie, Jii, Jee_ext, Jie_ext, \
                                                                           Jee_p, Jee_m, Jei_p, Jei_m, Jie_p, Jie_m, Jii_p, Jii_m, \
                                                                               fE, fEb, fI, fIb, p, n_activeClusters)    
        
            
        
                   
        # solution
        nu_out = fcn_compute_MFT_rates_DynEqs_reduced(nSteps, dt, T, stop_thresh, plot, \
                                                      tau_r_vec, tau_m_vec, tau_s_vec, Vr_vec,  Vth_vec,  \
                                                          Jab, Cab, Jab_ext, Cab_ext, externalNoise, nu_ext, nu_vec)    
            

            
        
    
    
    # FULL METHOD
    else:
    
        # compute Jab and Cab matrices
        Jab, Cab, Jab_ext, Cab_ext = fcn_MFT_clusteredEINetworks_tools.fcn_compute_weight_degree_mat(\
                                                                   Cee, Cei, Cie, Cii, Cext, \
                                                                   Jee, Jei, Jie, Jii, Jee_ext, Jie_ext, \
                                                                   Jee_p, Jee_m, Jei_p, Jei_m, Jie_p, Jie_m, Jii_p, Jii_m, \
                                                                   fE, fEb, fI, fIb, n_dPops_e, n_dPops_i)    
            
        
        # solution
        nu_out = fcn_MFT_fixedInDeg_generalNet.fcn_compute_MFT_rates_DynEqs(nSteps, dt, T, stop_thresh, plot, \
                                                                                tau_r_vec, tau_m_vec, tau_s_vec, Vr_vec,  Vth_vec,  \
                                                                                    Jab, Cab, Jab_ext, Cab_ext, externalNoise, nu_ext, nu_vec)              
        
           

    #-------------------------------------------------------------------------#        
    #----------------COMPUTE SELF-CONSISTENT MU AND SIGMA --------------------#
    #-------------------------------------------------------------------------#
    
    # REDUCED METHOD
    if ( (hasattr(a_params,'solve_reduced')) and (a_params.solve_reduced==True) ):

        nu_augmented = fcn_MFT_clusteredEINetworks_tools.fcn_compute_augmented_nuVec(nu_out)
        Mu = fcn_MFT_fixedInDeg_generalNet.fcn_compute_Mu(nu_augmented, nu_ext, Jab, Cab, Jab_ext, Cab_ext, tau_m_vec)
        Sigma2 = fcn_MFT_fixedInDeg_generalNet.fcn_compute_Sigma2(nu_augmented, nu_ext, Jab, Cab, Jab_ext, Cab_ext, tau_m_vec, externalNoise)
        
    
    
    # FULL METHOD
    else:
    
        Mu = fcn_MFT_fixedInDeg_generalNet.fcn_compute_Mu(nu_out, nu_ext, Jab, Cab, Jab_ext, Cab_ext, tau_m_vec)
        Sigma2 = fcn_MFT_fixedInDeg_generalNet.fcn_compute_Sigma2(nu_out, nu_ext, Jab, Cab, Jab_ext, Cab_ext, tau_m_vec, externalNoise)
    
    
    
    #-------------------------------------------------------------------------#        
    #----------------COMPUTE RATES USING SELF-CONSISTENT MU AND SIGMA2 -------#
    #-------------------------------------------------------------------------#
    
    # self-consistent rates
    nu_sc = np.zeros(n_dPops)
    
    for i in range(0,n_dPops):
        
        nu_sc[i] = fcn_MFT_fixedInDeg_generalNet.fcn_compute_rate(Vr_vec[i], Vth_vec[i], Mu[i], np.sqrt(Sigma2[i]), \
                                                                  tau_r_vec[i], tau_m_vec[i], tau_s_vec[i])  
    
    # verify that rates are consistent
    nu_check = all(abs(nu_sc - nu_out) < 1e-4)
        
    if (nu_check == True):
        print('verified solution is self consistent.')
        
    else:
        print('ERROR: Solution is not self-consistent!')
        # RESULTS DICTIONARY    
        results = {}
        results['nu_out'] = np.nan*np.ones(n_dPops)
        results['realPart_eigvals_S'] = np.nan
        results['S'] = np.nan
        results['realPart_eigvals_S_alt'] = np.nan
        results['S_alt'] = np.nan
        results['Jab'] = Jab
        results['Cab'] = Cab
        results['Jab_ext'] = Jab_ext
        results['Cab_ext'] = Cab_ext
        results['a_params'] = a_params
        results['Mu'] = Mu
        results['Sigma2'] = Sigma2
        
        return results
        
    


    #-------------------------------------------------------------------------#        
    #----------------IF USING REDUCED METHOD, EXTEND TO FULL SOLUTION --------#
    #-------------------------------------------------------------------------#
    if ( (a_params.solve_reduced==True) ):
        
    
        nu_out = fcn_MFT_clusteredEINetworks_tools.fcn_compute_full_vector_from_reduced(nu_out, p, n_activeClusters)
        Mu = fcn_MFT_clusteredEINetworks_tools.fcn_compute_full_vector_from_reduced(Mu, p, n_activeClusters)
        Sigma2 = fcn_MFT_clusteredEINetworks_tools.fcn_compute_full_vector_from_reduced(Sigma2, p, n_activeClusters)
        
    
    #-------------------------------------------------------------------------#        
    #----------------COMPUTE STABILITY ---------------------------------------#
    #-------------------------------------------------------------------------#
    if ( (a_params.solve_reduced==True) ):
        
        
        n_dPops_e = n_e_pops
        n_dPops_i = n_i_pops
        n_dPops = n_e_pops + n_i_pops
        
        # vectorize all parameters
        tau_r_vec = tau_r*np.ones(n_dPops)
        tau_m_vec = np.append(tau_m_e*np.ones(n_dPops_e), tau_m_i*np.ones(n_dPops_i))
        tau_s_vec = np.append(tau_s_e*np.ones(n_dPops_e), tau_s_i*np.ones(n_dPops_i))
        Vr_vec = np.append(Vr_e*np.ones(n_dPops_e), Vr_i*np.ones(n_dPops_i))
        Vth_vec = np.append(Vth_e*np.ones(n_dPops_e), Vth_i*np.ones(n_dPops_i))    
        nu_ext = np.append(nu_ext_e*np.ones(n_dPops_e), nu_ext_i*np.ones(n_dPops_i))
        
            
        # compute Jab and Cab matrices
        Jab_full, Cab_full, Jab_ext_full, Cab_ext_full = fcn_MFT_clusteredEINetworks_tools.fcn_compute_weight_degree_mat(\
                                                                   Cee, Cei, Cie, Cii, Cext, \
                                                                   Jee, Jei, Jie, Jii, Jee_ext, Jie_ext, \
                                                                   Jee_p, Jee_m, Jei_p, Jei_m, Jie_p, Jie_m, Jii_p, Jii_m, \
                                                                   fE, fEb, fI, fIb, n_dPops_e, n_dPops_i)  
            
        
        # stability    
        S, eigenvals_S, realPart_eigvals_S  = fcn_MFT_fixedInDeg_generalNet.fcn_stability_matrix(nu_out, \
                                                                                                 tau_r_vec, tau_m_vec, tau_s_vec, \
                                                                                                     Vr_vec, Vth_vec, nu_ext, \
                                                                                                         Jab_full, Cab_full, Jab_ext_full, Cab_ext_full, externalNoise)

        # stability alternate   
        S_alt, eigenvals_S_alt, realPart_eigvals_S_alt  = fcn_MFT_fixedInDeg_generalNet.fcn_stability_matrix_alternate(nu_out, \
                                                                                                 tau_r_vec, tau_m_vec, tau_s_vec, \
                                                                                                     Vr_vec, Vth_vec, nu_ext, \
                                                                                                         Jab_full, Cab_full, Jab_ext_full, Cab_ext_full, externalNoise, \
                                                                                                             stability_tau_vec)

    
    
    else:
    
        S, eigenvals_S, realPart_eigvals_S  = fcn_MFT_fixedInDeg_generalNet.fcn_stability_matrix(nu_out, \
                                                                                                 tau_r_vec, tau_m_vec, tau_s_vec, \
                                                                                                     Vr_vec, Vth_vec, nu_ext, \
                                                                                                         Jab, Cab, Jab_ext, Cab_ext, externalNoise)
            
        S_alt, eigenvals_S_alt, realPart_eigvals_S_alt  = fcn_MFT_fixedInDeg_generalNet.fcn_stability_matrix_alternate(nu_out, \
                                                                                                 tau_r_vec, tau_m_vec, tau_s_vec, \
                                                                                                     Vr_vec, Vth_vec, nu_ext, \
                                                                                                         Jab, Cab, Jab_ext, Cab_ext, externalNoise, \
                                                                                                             stability_tau_vec)
                                    
                        
    # RESULTS DICTIONARY
    results = {}
    results['nu_out'] = nu_out
    results['realPart_eigvals_S'] = realPart_eigvals_S
    results['S'] = S
    results['realPart_eigvals_S_alt'] = realPart_eigvals_S_alt
    results['S_alt'] = S_alt
    results['Mu'] = Mu
    results['Sigma2'] = Sigma2
    results['Jab'] = Jab
    results['Cab'] = Cab
    results['Jab_ext'] = Jab_ext
    results['Cab_ext'] = Cab_ext
    results['a_params'] = a_params
                        
            
    return results

#%% FUNCTION THAT SWEEPS OVER JEE+ AND COMPUTES MFT SOLUTION
### BACKWARDS


def fcn_JeePlus_sweep_backwards(sim_params, mft_params):
    
        
    # number of clusters
    nClu = sim_params.p
    
    # Jplus values
    minJplus = mft_params.min_JplusEE
    maxJplus = mft_params.max_JplusEE
    deltaJplus = mft_params.delta_JplusEE

    # number of active clusters to look for in solution
    n_activeClusters_sweep = mft_params.n_active_clusters_sweep
    
    # high and low rates to begin at
    nu_high_E = mft_params.nu_high_E
    nu_high_I = mft_params.nu_high_I
    nu_low_E = mft_params.nu_low_E
    nu_low_I = mft_params.nu_low_I
    
    # number of E and I pops
    n_e_pops = nClu + 1
    n_i_pops = nClu + 1
    n_pops = n_e_pops + n_i_pops

    
    # sanity checks
    if np.any(n_activeClusters_sweep > nClu):
        sys.exit('# of active clusters cannot be larger than the number of clusters')
        
    if n_e_pops != n_i_pops:
        sys.exit('this function assumes that the number of E and I pops is the same')
        
    if n_e_pops != nClu + 1:
        sys.exit('this function assumes that there are p clusters and 1 background population')
        
    
    # initialize backwards sweep quantities
    JplusEE_back = np.flip(np.arange(minJplus, maxJplus, deltaJplus))
    nu_e_back = np.zeros((n_e_pops, len(JplusEE_back), len(n_activeClusters_sweep)))
    nu_i_back = np.zeros((n_i_pops, len(JplusEE_back), len(n_activeClusters_sweep)))
    MaxReEig_back = np.zeros((len(JplusEE_back), len(n_activeClusters_sweep)))
    stabilityMatrix_back = np.zeros(( n_pops, n_pops, len(JplusEE_back), len(n_activeClusters_sweep) ))
    MaxReEig_back_alt = np.zeros((len(JplusEE_back), len(n_activeClusters_sweep)))
    stabilityMatrix_back_alt = np.zeros(( n_pops, n_pops, len(JplusEE_back), len(n_activeClusters_sweep) ))
    n_activeClustersE_back = np.zeros((len(JplusEE_back), len(n_activeClusters_sweep)))
    n_activeClustersI_back = np.zeros((len(JplusEE_back), len(n_activeClusters_sweep)))
        
    
    # loop over number of active clusters in solution
    for ind_nActive in range(0, len(n_activeClusters_sweep)):
        
        # number of active clusters in solution
        n_activeClusters = n_activeClusters_sweep[ind_nActive]
        mft_params.n_active_clusters = n_activeClusters

        # make initial and final rate vectors
        nu_vec_e_highJ = np.ones(nClu+1)
        nu_vec_e_highJ[:n_activeClusters] = nu_high_E
        nu_vec_e_highJ[n_activeClusters:] = nu_low_E
        
        nu_vec_i_highJ = np.ones(nClu+1)
        nu_vec_i_highJ[:n_activeClusters] = nu_high_I
        nu_vec_i_highJ[n_activeClusters:] = nu_low_I
        
        nu_vec_e_lowJ = np.ones(nClu+1)*nu_low_E
        nu_vec_i_lowJ = np.ones(nClu+1)*nu_low_I
        
        nu_vec_highJ = np.append(nu_vec_e_highJ, nu_vec_i_highJ)
        nu_vec_lowJ = np.append(nu_vec_e_lowJ, nu_vec_i_lowJ)
          
    
        # set initial rate vector
        mft_params.nu_vec = nu_vec_highJ
    
    
        # loop over Jee+
        for Jind in range(0,len(JplusEE_back),1):
            
            # update value of Jplus
            sim_params.JplusEE = JplusEE_back[Jind]
            
            # find fixed point and stability    
            
            # if first Jee+ value, solve using dynamical equations
            if Jind == 0:
                mft_results = fcn_master_MFT_DynEqs_fixedInDeg_EI_cluster_net(sim_params, mft_params)
            else:
                mft_results = fcn_master_MFT_fixedInDeg_EI_cluster_net(sim_params, mft_params)

                
            # firing rate
            nu_vec = mft_results['nu_out'].copy()

  
            # try low activity solution with dynamical equations
            if np.isnan(nu_vec[0]) == True:
                
                print('trying low activity fixed point; dynamical equations')
                    
                # update initial rate vector
                mft_params.nu_vec = nu_vec_highJ
                    
                # run MFT
                mft_results = fcn_master_MFT_DynEqs_fixedInDeg_EI_cluster_net(sim_params, mft_params)
                    
                # output rates
                nu_vec = mft_results['nu_out'].copy()
                
            
            
            # if no solution found, try low activity fixed point solving with dynamical equations
            if np.isnan(nu_vec[0]) == True:
                
                print('trying low activity fixed point; dynamical equations')

                # update initial rate vector
                mft_params.nu_vec = nu_vec_lowJ
                
                mft_results = fcn_master_MFT_DynEqs_fixedInDeg_EI_cluster_net(sim_params, mft_params)
            
                nu_vec = mft_results['nu_out'].copy()
            
            
            if np.isnan(nu_vec[0]) == True:
                sys.exit('could not find solution')
            
        
        
            # check that we found the solution we are looking for
            n_activeClustersE_back[Jind, ind_nActive] = np.size(np.nonzero( nu_vec[:nClu] >= np.max(nu_vec[:nClu])-1e-6 )[0])
            n_activeClustersI_back[Jind, ind_nActive] = np.size(np.nonzero( nu_vec[nClu+1:-1] >= np.max(nu_vec[nClu+1:-1])-1e-6 )[0])
            
            
            if ( ~( (n_activeClustersE_back[Jind, ind_nActive] == n_activeClusters) | (n_activeClustersE_back[Jind, ind_nActive] == nClu) ) ):
                
                print(nu_vec)
                print(Jind, ind_nActive, n_activeClustersE_back[Jind, ind_nActive])
                sys.exit('solution does not have correct # of active E clusters') 
            
            if ( ~((n_activeClustersI_back[Jind, ind_nActive] == n_activeClusters) | (n_activeClustersI_back[Jind, ind_nActive] == nClu)) ):
                print(nu_vec)
                print(Jind, ind_nActive, n_activeClustersI_back[Jind, ind_nActive])                
                #sys.exit('solution does not have correct # of active I clusters') 
        
        
            # save solution and stability
            nu_e_back[:,Jind, ind_nActive] = nu_vec[:n_e_pops].copy()
            nu_i_back[:,Jind, ind_nActive] = nu_vec[n_e_pops:].copy()
            MaxReEig_back[Jind, ind_nActive] = np.max(mft_results['realPart_eigvals_S'])
            stabilityMatrix_back[:,:, Jind, ind_nActive] = mft_results['S']
            MaxReEig_back_alt[Jind, ind_nActive] = np.max(mft_results['realPart_eigvals_S_alt'])
            stabilityMatrix_back_alt[:,:, Jind, ind_nActive] = mft_results['S_alt']
            
            
            # update initial guess at solution           
            mft_params.nu_vec = nu_vec
        
            # next value of Jplus        
            print(Jind)
        
        
        # next value of n_activeClusters
        print(ind_nActive)
    
    
    results = {}
    results['JplusEE_back'] = JplusEE_back
    results['nu_e_backSweep'] = nu_e_back
    results['nu_i_backSweep'] = nu_i_back
    results['stabilityMatrix_backSweep'] = stabilityMatrix_back
    results['maxRealEig_backSweep'] = MaxReEig_back
    results['stabilityMatrix_alt_backSweep'] = stabilityMatrix_back_alt
    results['maxRealEig_alt_backSweep'] = MaxReEig_back_alt
    results['n_activeClustersE_back'] = n_activeClustersE_back
    results['n_activeClustersI_back'] = n_activeClustersI_back

    return results



#%% FUNCTION THAT SWEEPS OVER JEE+ AND COMPUTES MFT SOLUTION
### FORWARDS

def fcn_JeePlus_sweep_forwards(sim_params, mft_params):
    
    # number of clusters
    nClu = sim_params.p
    
    # Jplus values
    minJplus = mft_params.min_JplusEE
    maxJplus = mft_params.max_JplusEE
    deltaJplus = mft_params.delta_JplusEE

    # number of active clusters to look for in solution
    n_activeClusters_sweep = mft_params.n_active_clusters_sweep    
    
    # high and low rates to begin at
    nu_high_E = mft_params.nu_high_E
    nu_high_I = mft_params.nu_high_I
    nu_low_E = mft_params.nu_low_E
    nu_low_I = mft_params.nu_low_I
    
    # number of E and I pops
    n_e_pops = nClu + 1
    n_i_pops = nClu + 1
    n_pops = n_e_pops + n_i_pops
    
        
        
    # sanity checks
    if np.any(n_activeClusters_sweep > nClu):
        sys.exit('# of active clusters cannot be larger than the number of clusters')
        
    if n_e_pops != n_i_pops:
        sys.exit('this function assumes that the number of E and I pops is the same')
        
    if n_e_pops != nClu + 1:
        sys.exit('this function assumes that there are p clusters and 1 background population')
        
        
    # initialize forwards sweep quantities
    JplusEE_for = np.arange(minJplus, maxJplus, deltaJplus)
    nu_e_for = np.zeros((n_e_pops, len(JplusEE_for), len(n_activeClusters_sweep)))
    nu_i_for = np.zeros((n_i_pops, len(JplusEE_for), len(n_activeClusters_sweep)))
    MaxReEig_for = np.zeros((len(JplusEE_for), len(n_activeClusters_sweep)))
    stabilityMatrix_for = np.zeros(( n_pops, n_pops, len(JplusEE_for), len(n_activeClusters_sweep) ))
    MaxReEig_for_alt = np.zeros((len(JplusEE_for), len(n_activeClusters_sweep)))
    stabilityMatrix_for_alt = np.zeros(( n_pops, n_pops, len(JplusEE_for), len(n_activeClusters_sweep) ))
    n_activeClustersE_for = np.zeros((len(JplusEE_for), len(n_activeClusters_sweep)))
    n_activeClustersI_for = np.zeros((len(JplusEE_for), len(n_activeClusters_sweep)))      
        
    
    # loop over number of active clusters in solution
    for ind_nActive in range(0, len(n_activeClusters_sweep)):
        
        # number of active clusters in solution
        n_activeClusters = n_activeClusters_sweep[ind_nActive]
        mft_params.n_active_clusters = n_activeClusters    
        
        
        # make initial and final rate vectors
        nu_vec_e_highJ = np.ones(nClu+1)
        nu_vec_e_highJ[:n_activeClusters] = nu_high_E
        nu_vec_e_highJ[n_activeClusters:] = nu_low_E
        
        nu_vec_i_highJ = np.ones(nClu+1)
        nu_vec_i_highJ[:n_activeClusters] = nu_high_I
        nu_vec_i_highJ[n_activeClusters:] = nu_low_I
        
        nu_vec_e_lowJ = np.ones(nClu+1)*nu_low_E
        nu_vec_i_lowJ = np.ones(nClu+1)*nu_low_I
        
        nu_vec_highJ = np.append(nu_vec_e_highJ, nu_vec_i_highJ)
        nu_vec_lowJ = np.append(nu_vec_e_lowJ, nu_vec_i_lowJ)
      
        
        # set initial rate vector
        mft_params.nu_vec = nu_vec_lowJ
    
    
        # loop over Jee+
        for Jind in range(0,len(JplusEE_for),1):
            
            # update value of Jplus
            sim_params.JplusEE = JplusEE_for[Jind]
            
            # find fixed point and stability    
            
            # if first Jee+ value, solve using dynamical equations
            if Jind == 0:
                mft_results = fcn_master_MFT_DynEqs_fixedInDeg_EI_cluster_net(sim_params, mft_params)
            else:
                mft_results = fcn_master_MFT_fixedInDeg_EI_cluster_net(sim_params, mft_params)
                
                
            # firing rate
            nu_vec = mft_results['nu_out'].copy()

            # if no solution found try solving with dynamical equations
            if np.isnan(nu_vec[0]) == True:
                print('trying low activity fixed point with dynamical equations')
                
                # update initial rate vector
                mft_params.nu_vec = nu_vec_lowJ
                
                # run MFT
                mft_results = fcn_master_MFT_DynEqs_fixedInDeg_EI_cluster_net(sim_params, mft_params)
                
                # output rate
                nu_vec = mft_results['nu_out'].copy()
                
            
            # if no solution found try solution where some clusters are active
            if np.isnan(nu_vec[0]) == True:
                
                print('trying high activity fixed point; dynamical equations')
                
                # update initial rate vector
                mft_params.nu_vec = nu_vec_highJ
                
                # run MFT
                mft_results = fcn_master_MFT_DynEqs_fixedInDeg_EI_cluster_net(sim_params, mft_params)
                
                # output rate
                nu_vec = mft_results['nu_out'].copy()
                
            if np.isnan(nu_vec[0]) == True:
                sys.exit('could not find solution')
        
        
            # check that we found the solution we are looking for

            n_activeClustersE_for[Jind, ind_nActive] = np.size(np.nonzero( nu_vec[:nClu] >= np.max(nu_vec[:nClu])-1e-6 )[0])
            n_activeClustersI_for[Jind, ind_nActive] = np.size(np.nonzero( nu_vec[nClu+1:-1] >= np.max(nu_vec[nClu+1:-1])-1e-6 )[0])
            
            
            if ( ~((n_activeClustersE_for[Jind, ind_nActive] == n_activeClusters) | (n_activeClustersE_for[Jind, ind_nActive] == nClu)) ):
                print(Jind, ind_nActive, n_activeClustersE_for[Jind, ind_nActive])
                sys.exit('solution does not have correct # of active E clusters') 
            
            
            if ( ~((n_activeClustersI_for[Jind, ind_nActive] == n_activeClusters) | (n_activeClustersI_for[Jind, ind_nActive] == nClu)) ):
                print(nu_vec)
                print(Jind, ind_nActive, n_activeClustersI_for[Jind, ind_nActive])
                sys.exit('solution does not have correct # of active I clusters') 
        
        
            # save solution and stability
            nu_e_for[:,Jind, ind_nActive] = nu_vec[:n_e_pops].copy()
            nu_i_for[:,Jind, ind_nActive] = nu_vec[n_e_pops:].copy()
            MaxReEig_for[Jind, ind_nActive] = np.max(mft_results['realPart_eigvals_S'])
            stabilityMatrix_for[:,:, Jind, ind_nActive] = mft_results['S']
            MaxReEig_for_alt[Jind, ind_nActive] = np.max(mft_results['realPart_eigvals_S_alt'])
            stabilityMatrix_for_alt[:,:, Jind, ind_nActive] = mft_results['S_alt']

            
            # update initial guess at solution           
            mft_params.nu_vec = nu_vec
        
            # next value of Jplus        
            print(Jind)
        
        
        # next value of n_activeClusters
        print(ind_nActive)
        
        
    results = {}
    results['JplusEE_for'] = JplusEE_for
    results['nu_e_forSweep'] = nu_e_for
    results['nu_i_forSweep'] = nu_i_for
    results['maxRealEig_forSweep'] = MaxReEig_for
    results['stabilityMatrix_forSweep'] = stabilityMatrix_for
    results['maxRealEig_alt_forSweep'] = MaxReEig_for_alt
    results['stabilityMatrix_alt_forSweep'] = stabilityMatrix_for_alt
    results['n_activeClustersE_for'] = n_activeClustersE_for
    results['n_activeClustersI_for'] = n_activeClustersI_for
    
    
    return results




#%% FUNCTION THAT SWEEPS OVER A GENERAL SET OF PARAMETERS AND COMPUTES MFT SOLUTION
### HIGH TO LOW RATE


def fcn_paramSweep_high_to_low_rate(sim_params, mft_params):
    
    swept_params_dict = {}
    
    if ( (sim_params.sweep_param_name == 'Jee_reduction_nu_ext_e_uniform_spread_nu_ext_i_uniform_spread')  or \
         (sim_params.sweep_param_name == 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread') ):
        swept_params_dict['Jee_sweep_vals'] = sim_params.Jee_sweep_vals.copy()
        swept_params_dict['nu_ext_e_sweep_vals']= sim_params.nu_ext_e_sweep_vals.copy()
        swept_params_dict['nu_ext_i_sweep_vals'] = sim_params.nu_ext_i_sweep_vals.copy()
    elif sim_params.sweep_param_name == 'pert_mean_nu_ext_ee':
        swept_params_dict['nu_ext_e_sweep_vals']= sim_params.nu_ext_e_sweep_vals.copy()
    else:
        sys.exit('unrecognized sweep param name')
    
        
    # number of clusters
    nClu = sim_params.p

    # number of active clusters to look for in solution
    n_activeClusters_sweep = mft_params.n_active_clusters_sweep
    
    # number of sampled values for swept parameters
    n_sampledValues = sim_params.n_paramVals_mft
    
    # high and low rates to begin at
    nu_high_E = mft_params.nu_high_E
    nu_high_I = mft_params.nu_high_I
    nu_low_E = mft_params.nu_low_E
    nu_low_I = mft_params.nu_low_I
    
    # number of E and I pops
    n_e_pops = nClu + 1
    n_i_pops = nClu + 1
    n_pops = n_e_pops + n_i_pops
    

    # sanity checks
    if np.any(n_activeClusters_sweep > nClu):
        sys.exit('# of active clusters cannot be larger than the number of clusters')
        
    if n_e_pops != n_i_pops:
        sys.exit('this function assumes that the number of E and I pops is the same')
        
    if n_e_pops != nClu + 1:
        sys.exit('this function assumes that there are p clusters and 1 background population')
        
    
    # initialize backwards sweep quantities
    nu_e_back = np.zeros((n_e_pops, n_sampledValues, len(n_activeClusters_sweep)))
    nu_i_back = np.zeros((n_i_pops, n_sampledValues, len(n_activeClusters_sweep)))
    MaxReEig_back = np.zeros((n_sampledValues, len(n_activeClusters_sweep)))
    stabilityMatrix_back = np.zeros(( n_pops, n_pops, n_sampledValues, len(n_activeClusters_sweep) ))
    MaxReEig_back_alt = np.zeros((n_sampledValues, len(n_activeClusters_sweep)))
    stabilityMatrix_back_alt = np.zeros(( n_pops, n_pops, n_sampledValues, len(n_activeClusters_sweep) ))
    n_activeClustersE_back = np.zeros((n_sampledValues, len(n_activeClusters_sweep)))
    n_activeClustersI_back = np.zeros((n_sampledValues, len(n_activeClusters_sweep)))
        
    
    # loop over number of active clusters in solution
    for ind_nActive in range(0, len(n_activeClusters_sweep)):
        
        # number of active clusters in solution
        n_activeClusters = n_activeClusters_sweep[ind_nActive]
        mft_params.n_active_clusters = n_activeClusters

        # make initial and final rate vectors
        nu_vec_e_high = np.ones(nClu+1)
        nu_vec_e_high[:n_activeClusters] = nu_high_E
        nu_vec_e_high[n_activeClusters:] = nu_low_E
        
        nu_vec_i_high = np.ones(nClu+1)
        nu_vec_i_high[:n_activeClusters] = nu_high_I
        nu_vec_i_high[n_activeClusters:] = nu_low_I
        
        nu_vec_e_low = np.ones(nClu+1)*nu_low_E
        nu_vec_i_low = np.ones(nClu+1)*nu_low_I
        
        nu_vec_high = np.append(nu_vec_e_high, nu_vec_i_high)
        nu_vec_low = np.append(nu_vec_e_low, nu_vec_i_low)
          
    
        # set initial rate vector
        mft_params.nu_vec = nu_vec_high
    
    
        # loop over values of swept parameters
        for indParamSweep in range(0,n_sampledValues,1):
            
            # update value of swept parameters
            if ( (sim_params.sweep_param_name == 'Jee_reduction_nu_ext_e_uniform_spread_nu_ext_i_uniform_spread')  or \
                 (sim_params.sweep_param_name == 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread') ):
                sim_params.Jee = sim_params.Jee_sweep_vals[indParamSweep]
                sim_params.nu_ext_e[:] = sim_params.nu_ext_e_sweep_vals[indParamSweep]
                sim_params.nu_ext_i[:] = sim_params.nu_ext_i_sweep_vals[indParamSweep]
            elif sim_params.sweep_param_name == 'pert_mean_nu_ext_ee':
                sim_params.nu_ext_e[:] = sim_params.nu_ext_e_sweep_vals[indParamSweep]  
            else:
                sys.exit('unknown swept parameter combination')
           
            
            # find fixed point and stability    
            
            # if first parameter value, solve using dynamical equations
            if indParamSweep == 0:
                mft_results = fcn_master_MFT_DynEqs_fixedInDeg_EI_cluster_net(sim_params, mft_params)
            else:
                mft_results = fcn_master_MFT_fixedInDeg_EI_cluster_net(sim_params, mft_params)

                
            # firing rate
            nu_vec = mft_results['nu_out'].copy()
            
            
            # low activity solution
            if np.isnan(nu_vec[0]) == True:
                    
                print('trying low activity fixed point')

                # update initial rate vector
                mft_params.nu_vec = nu_vec_low
                    
                # run MFT
                mft_results = fcn_master_MFT_DynEqs_fixedInDeg_EI_cluster_net(sim_params, mft_params)
                    
                # output rates
                nu_vec = mft_results['nu_out'].copy()
                
            if np.isnan(nu_vec[0]) == True:
                sys.exit('could not find solution')
            
        
        
            # check that we found the solution we are looking for
            n_activeClustersE_back[indParamSweep, ind_nActive] = np.size(np.nonzero( nu_vec[:nClu] >= np.max(nu_vec[:nClu])-1e-6 )[0])
            n_activeClustersI_back[indParamSweep, ind_nActive] = np.size(np.nonzero( nu_vec[nClu+1:-1] >= np.max(nu_vec[nClu+1:-1])-1e-6 )[0])
            
            
            if ( ~( (n_activeClustersE_back[indParamSweep, ind_nActive] == n_activeClusters) | (n_activeClustersE_back[indParamSweep, ind_nActive] == nClu) ) ):
                
                print(nu_vec)
                print(indParamSweep, ind_nActive, n_activeClustersE_back[indParamSweep, ind_nActive])
                sys.exit('solution does not have correct # of active E clusters') 
            
            if ( ~((n_activeClustersI_back[indParamSweep, ind_nActive] == n_activeClusters) | (n_activeClustersI_back[indParamSweep, ind_nActive] == nClu)) ):
                print(nu_vec)
                print(indParamSweep, ind_nActive, n_activeClustersI_back[indParamSweep, ind_nActive])                
                #sys.exit('solution does not have correct # of active I clusters') 
        
        
            # save solution and stability
            nu_e_back[:,indParamSweep, ind_nActive] = nu_vec[:n_e_pops].copy()
            nu_i_back[:,indParamSweep, ind_nActive] = nu_vec[n_e_pops:].copy()
            MaxReEig_back[indParamSweep, ind_nActive] = np.max(mft_results['realPart_eigvals_S'])
            stabilityMatrix_back[:,:, indParamSweep, ind_nActive] = mft_results['S']
            MaxReEig_back_alt[indParamSweep, ind_nActive] = np.max(mft_results['realPart_eigvals_S_alt'])
            stabilityMatrix_back_alt[:,:, indParamSweep, ind_nActive] = mft_results['S_alt']
            
            
            # update initial guess at solution           
            mft_params.nu_vec = nu_vec
        
            # next parameter value index  
            print(indParamSweep)
        
        
        # next value of n_activeClusters
        print(ind_nActive)
    
    
    results = {}
    results['swept_params_dict'] = swept_params_dict
    results['nu_e_backSweep'] = nu_e_back
    results['nu_i_backSweep'] = nu_i_back
    results['stabilityMatrix_backSweep'] = stabilityMatrix_back
    results['maxRealEig_backSweep'] = MaxReEig_back
    results['stabilityMatrix_alt_backSweep'] = stabilityMatrix_back_alt
    results['maxRealEig_alt_backSweep'] = MaxReEig_back_alt
    results['n_activeClustersE_back'] = n_activeClustersE_back
    results['n_activeClustersI_back'] = n_activeClustersI_back

    return results



#%% FUNCTION THAT SWEEPS OVER A GENERAL SET OF PARAMETERS AND COMPUTES MFT SOLUTION
### LOW TO HIGH RATE


def fcn_paramSweep_low_to_high_rate(sim_params, mft_params):
    
    
    swept_params_dict = {}
    if ( (sim_params.sweep_param_name == 'Jee_reduction_nu_ext_e_uniform_spread_nu_ext_i_uniform_spread')  or \
         (sim_params.sweep_param_name == 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread') ):
        swept_params_dict['Jee_sweep_vals'] = sim_params.Jee_sweep_vals.copy()
        swept_params_dict['nu_ext_e_sweep_vals']= sim_params.nu_ext_e_sweep_vals.copy()
        swept_params_dict['nu_ext_i_sweep_vals'] = sim_params.nu_ext_i_sweep_vals.copy()
    elif sim_params.sweep_param_name == 'pert_mean_nu_ext_ee':
        swept_params_dict['nu_ext_e_sweep_vals']= sim_params.nu_ext_e_sweep_vals.copy()
    else:
        sys.exit('unrecognized sweep param name')
    
        
    # number of clusters
    nClu = sim_params.p

    # number of active clusters to look for in solution
    n_activeClusters_sweep = mft_params.n_active_clusters_sweep
    
    # number of sampled values for swept parameters
    n_sampledValues = sim_params.n_paramVals_mft
    
    # high and low rates to begin at
    nu_high_E = mft_params.nu_high_E
    nu_high_I = mft_params.nu_high_I
    nu_low_E = mft_params.nu_low_E
    nu_low_I = mft_params.nu_low_I
    
    # number of E and I pops
    n_e_pops = nClu + 1
    n_i_pops = nClu + 1
    n_pops = n_e_pops + n_i_pops

        
    
    # sanity checks
    if np.any(n_activeClusters_sweep > nClu):
        sys.exit('# of active clusters cannot be larger than the number of clusters')
        
    if n_e_pops != n_i_pops:
        sys.exit('this function assumes that the number of E and I pops is the same')
        
    if n_e_pops != nClu + 1:
        sys.exit('this function assumes that there are p clusters and 1 background population')
        
    
    # initialize backwards sweep quantities
    nu_e_for = np.zeros((n_e_pops, n_sampledValues, len(n_activeClusters_sweep)))
    nu_i_for  = np.zeros((n_i_pops, n_sampledValues, len(n_activeClusters_sweep)))
    MaxReEig_for  = np.zeros((n_sampledValues, len(n_activeClusters_sweep)))
    stabilityMatrix_for  = np.zeros(( n_pops, n_pops, n_sampledValues, len(n_activeClusters_sweep) ))
    MaxReEig_for_alt = np.zeros((n_sampledValues, len(n_activeClusters_sweep)))
    stabilityMatrix_for_alt = np.zeros(( n_pops, n_pops, n_sampledValues, len(n_activeClusters_sweep) ))
    n_activeClustersE_for  = np.zeros((n_sampledValues, len(n_activeClusters_sweep)))
    n_activeClustersI_for  = np.zeros((n_sampledValues, len(n_activeClusters_sweep)))
        
    
    # loop over number of active clusters in solution
    for ind_nActive in range(0, len(n_activeClusters_sweep)):
        
        # number of active clusters in solution
        n_activeClusters = n_activeClusters_sweep[ind_nActive]
        mft_params.n_active_clusters = n_activeClusters

        # make initial and final rate vectors
        nu_vec_e_high = np.ones(nClu+1)
        nu_vec_e_high[:n_activeClusters] = nu_high_E
        nu_vec_e_high[n_activeClusters:] = nu_low_E
        
        nu_vec_i_high = np.ones(nClu+1)
        nu_vec_i_high[:n_activeClusters] = nu_high_I
        nu_vec_i_high[n_activeClusters:] = nu_low_I
        
        nu_vec_e_low = np.ones(nClu+1)*nu_low_E
        nu_vec_i_low = np.ones(nClu+1)*nu_low_I
        
        nu_vec_high = np.append(nu_vec_e_high, nu_vec_i_high)
        nu_vec_low = np.append(nu_vec_e_low, nu_vec_i_low)
          
    
        # set initial rate vector
        mft_params.nu_vec = nu_vec_low
    
    
        # loop over values of swept parameters
        for indParamSweep in range(n_sampledValues-1,-1,-1):
            
            # update value of swept parameters
            if ( (sim_params.sweep_param_name == 'Jee_reduction_nu_ext_e_uniform_spread_nu_ext_i_uniform_spread')  or \
                 (sim_params.sweep_param_name == 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread') ):
                sim_params.Jee = sim_params.Jee_sweep_vals[indParamSweep]
                sim_params.nu_ext_e[:] = sim_params.nu_ext_e_sweep_vals[indParamSweep]
                sim_params.nu_ext_i[:] = sim_params.nu_ext_i_sweep_vals[indParamSweep]
            elif sim_params.sweep_param_name == 'pert_mean_nu_ext_ee':
                sim_params.nu_ext_e[:] = sim_params.nu_ext_e_sweep_vals[indParamSweep]                
            else:
                sys.exit('unknown swept parameter combination')
           
            
            # find fixed point and stability    
            
            # if first parameter value, solve using dynamical equations
            if indParamSweep == n_sampledValues-1:
                mft_results = fcn_master_MFT_DynEqs_fixedInDeg_EI_cluster_net(sim_params, mft_params)
            else:
                mft_results = fcn_master_MFT_fixedInDeg_EI_cluster_net(sim_params, mft_params)

                
            # firing rate
            nu_vec = mft_results['nu_out'].copy()
            
            
            # try high activity solution
            if np.isnan(nu_vec[0]) == True:
                    
                print('trying high activity fixed point')

                # update initial rate vector
                mft_params.nu_vec = nu_vec_high
                    
                # run MFT
                mft_results = fcn_master_MFT_DynEqs_fixedInDeg_EI_cluster_net(sim_params, mft_params)
                    
                # output rates
                nu_vec = mft_results['nu_out'].copy()
            
            if np.isnan(nu_vec[0]) == True:
                sys.exit('could not find solution')
        
        
            # check that we found the solution we are looking for
            n_activeClustersE_for[indParamSweep, ind_nActive] = np.size(np.nonzero( nu_vec[:nClu] >= np.max(nu_vec[:nClu])-1e-6 )[0])
            n_activeClustersI_for[indParamSweep, ind_nActive] = np.size(np.nonzero( nu_vec[nClu+1:-1] >= np.max(nu_vec[nClu+1:-1])-1e-6 )[0])
            
            
            if ( ~( (n_activeClustersE_for[indParamSweep, ind_nActive] == n_activeClusters) | (n_activeClustersE_for[indParamSweep, ind_nActive] == nClu) ) ):
                
                print(nu_vec)
                print(indParamSweep, ind_nActive, n_activeClustersE_for[indParamSweep, ind_nActive])
                sys.exit('solution does not have correct # of active E clusters') 
            
            if ( ~((n_activeClustersI_for[indParamSweep, ind_nActive] == n_activeClusters) | (n_activeClustersI_for[indParamSweep, ind_nActive] == nClu)) ):
                print(nu_vec)
                print(indParamSweep, ind_nActive, n_activeClustersI_for[indParamSweep, ind_nActive])                
                #sys.exit('solution does not have correct # of active I clusters') 
        
        
            # save solution and stability
            nu_e_for[:,indParamSweep, ind_nActive] = nu_vec[:n_e_pops].copy()
            nu_i_for[:,indParamSweep, ind_nActive] = nu_vec[n_e_pops:].copy()
            MaxReEig_for[indParamSweep, ind_nActive] = np.max(mft_results['realPart_eigvals_S'])
            stabilityMatrix_for[:,:, indParamSweep, ind_nActive] = mft_results['S']
            MaxReEig_for_alt[indParamSweep, ind_nActive] = np.max(mft_results['realPart_eigvals_S_alt'])
            stabilityMatrix_for_alt[:,:, indParamSweep, ind_nActive] = mft_results['S_alt']
            
            
            # update initial guess at solution           
            mft_params.nu_vec = nu_vec
        
            # next parameter value index  
            print(indParamSweep)
        
        
        # next value of n_activeClusters
        print(ind_nActive)
    
    
    results = {}
    results['swept_params_dict'] = swept_params_dict
    results['nu_e_forSweep'] = nu_e_for
    results['nu_i_forSweep'] = nu_i_for
    results['stabilityMatrix_forSweep'] = stabilityMatrix_for
    results['maxRealEig_forSweep'] = MaxReEig_for
    results['stabilityMatrix_alt_forSweep'] = stabilityMatrix_for_alt
    results['maxRealEig_alt_forSweep'] = MaxReEig_for_alt
    results['n_activeClustersE_for'] = n_activeClustersE_for
    results['n_activeClustersI_for'] = n_activeClustersI_for

    return results

