
# basic imports
import numpy as np
import sys

#-----------------------------------------------------------------------------
# CLASS FILE THAT SETS PARAMETERS FOR SIMULATIONS
# USE FOR SIMULATIONS OF HETEROGENEOUS RANDOM NETWORKS
#-----------------------------------------------------------------------------

class sim_params:

    
    # INIT METHOD
    def __init__(self):     
        
        
        self.synType = 'exp'

        self.N = 800                       # total number of neurons
        self.ne = 0.8                       # total fraction of excitatory neurons
        
        self.Vth_e = 4.86                    # excitatory threshold
        self.Vth_i = 5.98                   # inhibitory threshold
        self.Vr_e = 0.                      # reset potential E
        self.Vr_i = 0.                      # reset potential I
        
        self.tau_m_e = 20e-3                # membrane time constant E
        self.tau_m_i = 20e-3                # membrane time constant I  
        self.tau_s_e = 5e-3
        self.tau_s_i = 5e-3         
        self.tau_r = 5e-3                   # refractory period
        
        self.extCurrent_poisson = True
        
        #-----------------------------------------------------------------------------
        # EXTERNAL INPUTS
        #----------------------------------------------------------------------------- 
        
        # external input level
        self.nu_ext_e = 7.0
        self.nu_ext_i = 7.0
        
        
        #-----------------------------------------------------------------------------
        # AROUSAL
        #----------------------------------------------------------------------------- 
        self.Jee_reduction = 0.
        self.nu_ext_e_delta = 0.
        self.nu_ext_i_delta = 0.

        #-----------------------------------------------------------------------------
        # NETWORK PROPERTIES
        #-----------------------------------------------------------------------------        
        
        # cluster or homogeneous
        self.net_type = 'cluster'
        
        # network connection type        
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

        
        # clusters
        self.p = 2
        self.bgrE = 0.75
        self.bgrI = 1.
        # which neurons & weights are clustered
        self.clusters = ['E']
        self.clusterWeights = ['EE']
        
        # other cluster properties (probably wont change much)
        self.Ecluster_weightSize = False
        # cluster size heterogeneity ('hom' or 'het')
        self.clustE = 'hom' # E clusters 
        self.clustI = 'hom' # I clusters 
        if self.clustE == 'hom' or self.clustI == 'hom':
            # std of cluster size (as a fraction of mean)
            self.clust_std = 0.0 
        else:
            self.clust_std = 1.0
            
            
        self.depress_interCluster = False
        
        # cluster depression & potentiation
        self.JplusEE = 20.0                  # EE intra-cluster potentiation factor
        self.JplusII = 1.0                  # II intra-cluster potentiation factor
        self.JplusEI = 1.0                  # EI intra-cluster potentiation factor
        self.JplusIE = 1.0                  # IE intra-cluster potentiation factor
        
        # variance in synaptic weights
        self.deltaEE = 0
        self.deltaEI = 0
        self.deltaIE = 0
        self.deltaII = 0
        
#-----------------------------------------------------------------------------
# CLASS FUNCTIONS
#-----------------------------------------------------------------------------
 
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
                   
        
        # total numbers of E and I neurons
        self.N_e = int(self.N*self.ne)                   
        self.N_i = int(self.N - self.N_e)       
        
        # if homogeneous networks
        if self.net_type == 'hom':
            
            self.JplusEE = 1.0       # EE intra-cluster potentiation factor
            self.JplusII = 1.0       # II intra-cluster potentiation factor
            self.JplusEI = 1.0       # EI intra-cluster potentiation factor
            self.JplusIE = 1.0       # IE intra-cluster potentiation factor

        
        # number of connections
        self.Cee = self.pee*self.N_e
        self.Cei = self.pei*self.N_i
        self.Cii = self.pii*self.N_i
        self.Cie = self.pie*self.N_e
        self.Cext = self.N_e*self.pext 
        
        
        # external inputs
        self.nu_ext_e = self.nu_ext_e*np.ones(self.N_e)
        self.nu_ext_i = self.nu_ext_i*np.ones(self.N_i)
        
        # update simulation parameters based on arousal parameters
        self.Jee = self.Jee - self.Jee*self.Jee_reduction
        self.nu_ext_e = self.nu_ext_e + self.nu_ext_e_delta
        self.nu_ext_i = self.nu_ext_i + self.nu_ext_i_delta
        
                        



        