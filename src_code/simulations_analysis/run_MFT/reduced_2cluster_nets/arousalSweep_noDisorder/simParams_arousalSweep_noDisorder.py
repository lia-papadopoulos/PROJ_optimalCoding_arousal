
import numpy as np
import sys

class sim_params:

    
    def __init__(self):     
        
        
        self.synType = 'exp'
        self.N = 800                      
        self.ne = 0.8                      
        self.Vth_e = 4.86                    
        self.Vth_i = 5.98                  
        self.Vr_e = 0.                     
        self.Vr_i = 0.                      
        self.tau_m_e = 20e-3                
        self.tau_m_i = 20e-3                
        self.tau_s_e = 5e-3
        self.tau_s_i = 5e-3         
        self.tau_r = 5e-3                   
        self.extCurrent_poisson = True
        self.nu_ext_e = 7.0
        self.nu_ext_i = 7.0
        self.Jee_reduction = 0.
        self.nu_ext_e_delta = 0.
        self.nu_ext_i_delta = 0.
        self.net_type = 'cluster'
        self.connType = 'fixed_InDegree'         
        self.pext = 0.2
        self.pee = 0.2
        self.pei = 0.5
        self.pii = 0.5
        self.pie = 0.5        
        self.Jee = 0.8/np.sqrt(self.N)
        self.Jie = 2.5/np.sqrt(self.N)
        self.Jei = 10.6/np.sqrt(self.N)
        self.Jii = 9.7/np.sqrt(self.N)
        self.Jie_ext = 12.9/np.sqrt(self.N)
        self.Jee_ext = 14.5/np.sqrt(self.N)        
        self.p = 2
        self.bgrE = 0.75
        self.bgrI = 1.
        self.clusters = ['E']
        self.clusterWeights = ['EE']
        self.Ecluster_weightSize = False
        self.clustE = 'hom' 
        self.clustI = 'hom' 
        if self.clustE == 'hom' or self.clustI == 'hom':
            self.clust_std = 0.0 
        else:
            self.clust_std = 1.0
        self.depress_interCluster = False
        self.JplusEE = 20.0                  
        self.JplusII = 1.0                  
        self.JplusEI = 1.0                  
        self.JplusIE = 1.0                 
        self.deltaEE = 0
        self.deltaEI = 0
        self.deltaIE = 0
        self.deltaII = 0
        

    #---------------------------------------------------------------------------------
    # SET AROUSAL VARIABLES
    #--------------------------------------------------------------------------------- 
    def fcn_set_arousalVars(self, sweep_param_name, swept_params_dict, param_indx):


        if sweep_param_name == 'Jee_reduction_nu_ext_ee_nu_ext_ie':
    
            self.Jee_reduction = swept_params_dict['param_vals1'][param_indx]
            self.nu_ext_e_delta = swept_params_dict['param_vals2'][param_indx]
            self.nu_ext_i_delta = swept_params_dict['param_vals3'][param_indx]
    
        else:
    
            sys.exit('unknown arousal type')


    #---------------------------------------------------------------------------------
    # SET ANY VARIABLES THAT ARE COMPLETELY DETERMINED BY MAIN INPUTS
    #---------------------------------------------------------------------------------       
    def set_dependent_vars(self):
                   
        
        self.N_e = int(self.N*self.ne)                   
        self.N_i = int(self.N - self.N_e)       
        
        if self.net_type == 'hom':
            
            self.JplusEE = 1.0       
            self.JplusII = 1.0       
            self.JplusEI = 1.0       
            self.JplusIE = 1.0       
        
        self.Cee = self.pee*self.N_e
        self.Cei = self.pei*self.N_i
        self.Cii = self.pii*self.N_i
        self.Cie = self.pie*self.N_e
        self.Cext = self.N_e*self.pext 
        
        self.nu_ext_e = self.nu_ext_e*np.ones(self.N_e)
        self.nu_ext_i = self.nu_ext_i*np.ones(self.N_i)
        
        self.Jee = self.Jee - self.Jee*self.Jee_reduction
        self.nu_ext_e = self.nu_ext_e + self.nu_ext_e_delta
        self.nu_ext_i = self.nu_ext_i + self.nu_ext_i_delta
        
                        



        