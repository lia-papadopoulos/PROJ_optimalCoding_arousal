import sys

def fcn_update_params_forMFT(s_params):
    
    # external connections
    if s_params.pext_ee != s_params.pext_ie:
        sys.exit('external connection probability not the same for E->E and E->I; not handled by MFT')
        
    if hasattr(s_params, 'Cext') == False:
        s_params.Cext = s_params.pext_ee*s_params.N_e
        
    # check external input rates for heterogeneity
    
    if s_params.nu_ext_ee_beta_spread != 0:
        s_params.nu_ext_ee[:] = s_params.mean_nu_ext_ee + s_params.nu_ext_ee_beta_spread/2
            
        if s_params.nu_ext_ee_uniform_spread != 0:
            sys.exit('cant have both beta and uniform spread nonzero')
            
    elif s_params.nu_ext_ee_uniform_spread != 0:
        s_params.nu_ext_ee[:] = s_params.mean_nu_ext_ee + s_params.nu_ext_ee_uniform_spread/2
        

    if s_params.nu_ext_ie_beta_spread != 0:
        s_params.nu_ext_ie[:] = s_params.mean_nu_ext_ie + s_params.nu_ext_ie_beta_spread/2
            
        if s_params.nu_ext_ie_uniform_spread != 0:
            sys.exit('cant have both beta and uniform spread nonzero')    
            
    elif s_params.nu_ext_ie_uniform_spread != 0:
        s_params.nu_ext_ie[:] = s_params.mean_nu_ext_ie + s_params.nu_ext_ie_uniform_spread/2    


    # check inhibitory inputs
    if ( (s_params.mean_nu_ext_ei != 0) or (s_params.pert_mean_nu_ext_ei != 0) or (s_params.nu_ext_ei_beta_spread !=0 ) or (s_params.nu_ext_ei_uniform_spread != 0) ):
        sys.exit('mft function cant handle inhibitory external input')
        
    if ( (s_params.mean_nu_ext_ii != 0) or (s_params.pert_mean_nu_ext_ii != 0) or (s_params.nu_ext_ii_beta_spread !=0 ) or (s_params.nu_ext_ii_uniform_spread != 0) ):
        sys.exit('mft function cant handle inhibitory external input')
        
    return s_params