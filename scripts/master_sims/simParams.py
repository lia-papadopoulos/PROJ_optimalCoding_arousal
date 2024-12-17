"""
notes:
    could compute baseline external and perturbation external currents
    separately, and then feed those into fcn_simulation.py

"""


import numpy as np
import sys

sys.path.append('../functions') 
from fcn_make_network_cluster import fcn_compute_cluster_assignments

#-----------------------------------------------------------------------------
# CLASS FILE THAT SETS PARAMETERS FOR SIMULATIONS
# USE FOR SIMULATIONS OF HETEROGENEOUS RANDOM NETWORKS
#-----------------------------------------------------------------------------

class sim_params:

    
    # INIT METHOD
    def __init__(self):     
        
        #-----------------------------------------------------------------------------
        # DEFAULTS
        #-----------------------------------------------------------------------------  
        

     
        self.writeNetwork_to_file = False       # write network to file?
        self.writeSimulation_to_file = True     # write simulation to file?
        self.save_voltage = False               # whether or not to save membrane potential, input current 
                    
        self.T0 = 0.                            # simulation start time
        self.TF = 1.5                         # simulation end time
        self.dt = 0.05e-3                       # time step
        
        self.N = 2000                       # total number of neurons
        self.ne = 0.8                       # total fraction of excitatory neurons
        
        self.Vth_e = 1.5                    # excitatory threshold
        self.Vth_i = 0.75                   # inhibitory threshold
        self.Vr_e = 0.                      # reset potential E
        self.Vr_i = 0.                      # reset potential I
        
        self.tau_m_e = 20e-3                # membrane time constant E
        self.tau_m_i = 20e-3                # membrane time constant I  
        self.tau_s_e = 5e-3
        self.tau_s_i = 5e-3         
        self.tau_r = 5e-3                   # refractory period
        self.t_delay = 0.             # delay
        
        self.synType = 'exp'                # synapse type
        
            
        # whether or not external inputs are poisson
        self.base_extCurrent_poisson = True
        self.pert_extCurrent_poisson = True
        self.pert_toVoltage = False
        
    
        #-----------------------------------------------------------------------------
        # EXTERNAL INPUTS
        #----------------------------------------------------------------------------- 
        
        # external input mean and spatial standard dev 
        self.mean_nu_ext_ee = 7.0
        self.mean_nu_ext_ie = 7.0
        self.mean_nu_ext_ei = 0.
        self.mean_nu_ext_ii = 0.
        
        # perturbations
        self.Jee_reduction = 0.
        self.Jie_reduction = 0.
        
        
        self.pert_mean_nu_ext_ee = 0.
        self.pert_mean_nu_ext_ie = 0.
        self.pert_mean_nu_ext_ei = 0.
        self.pert_mean_nu_ext_ii = 0.
        

        
        #-----------------------------------------------------------------------------
        # NETWORK PROPERTIES
        #-----------------------------------------------------------------------------        
        
        # cluster or hom
        self.net_type = 'cluster'
        
        # network connection type        
        self.connType = 'fixed_InDegree' 
        
        # whether or not to depress inter-cluster connections
        self.depress_interCluster = True
               
        self.pee = 0.2
        self.pei = 0.5
        self.pii = 0.5
        self.pie = 0.5
        
        self.pext_ee = 0.2
        self.pext_ie = 0.2
        self.pext_ei = 0.8
        self.pext_ii = 0.8
        
        self.jee = 0.63
        self.jie = 0.63
        self.jei = 1.9
        self.jii = 3.8
        
        self.jie_ext = 2.3
        self.jee_ext = 2.3
        self.jii_ext = 2.3
        self.jei_ext = 2.3
        
        # clusters
        self.p = 18
        self.bgrE = 0.1
        self.bgrI = 0.1
        # which neurons & weights are clustered
        self.clusters = ['E','I']
        self.clusterWeights = ['EE','EI','IE','II']
        
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
            
        # cluster depression & potentiation
        self.JplusEE = 15.75                 # EE intra-cluster potentiation factor (poisson)
        self.JplusII = 5.0                   # II intra-cluster potentiation factor
        self.JplusEI = 6.25                  # EI intra-cluster potentiation factor
        self.JplusIE = 5.45                  # IE intra-cluster potentiation factor
        # variance in synaptic weights
        self.deltaEE = 0
        self.deltaEI = 0
        self.deltaIE = 0
        self.deltaII = 0
        
        #-----------------------------------------------------------------------------
        # STIMULUS PROPERTIES
        #-----------------------------------------------------------------------------    
        # for stimuli, specify:
        self.stim_type = ''           # type of stimulus ['' or 'noStim']
        self.nStim = 5                      # number of different stimuli to run
        self.mixed_selectivity = True       # allow different stimuli to target same clusters
        self.stim_shape = 'diff2exp'        # type of stimulus
        self.stim_onset = self.T0 + 1.0     # stimulus onset
        self.f_selectiveClus = 0.5          # fraction of clusters that are selective to each stimulus
        self.f_Ecells_target = 0.5          # fraction E cells targeted in selective clusters
        self.f_Icells_target = 0.0          # fraction of I cells targeted in selective clsuters
        self.stim_rel_amp = 0.05            # relative strength (fraction above baseline)
        if self.stim_type == 'noStim':      # set stim strength to zero if stim type is 'noStim'
            self.nStim = 1
            self.stim_rel_amp = 0
        
        # for box and linear
        self.stim_duration = 2.             # duration of stimulus in seconds
        # for difference of exponentials
        self.stim_taur = 0.075
        self.stim_taud = 0.1
        
        #-----------------------------------------------------------------------------
        # FILENAMES
        #----------------------------------------------------------------------------- 
        self.parameters_fileName = ('simulationData')
        self.network_name = 'network'
        
        # PRINT
        print('sim_params class initialized')
        
#-----------------------------------------------------------------------------
# CLASS FUNCTIONS
#-----------------------------------------------------------------------------
    
    #---------------------------------------------------------------------------------
    # GET ANY VALUES THAT COULD HAVE BEEN PASSED THROUGH ARGPARSE
    #---------------------------------------------------------------------------------
    def get_argparse_vals(self, args):
        
        # get any values from argparser
        
        # perturbations
        self.pert_mean_nu_ext_ee = args.pert_mean_nu_ext_ee
        self.pert_mean_nu_ext_ie = args.pert_mean_nu_ext_ie
        self.pert_mean_nu_ext_ii = args.pert_mean_nu_ext_ii
        self.pert_mean_nu_ext_ei = args.pert_mean_nu_ext_ei

        # network type
        self.net_type = args.net_type

        self.Jee_reduction = args.Jee_reduction
        self.Jie_reduction = args.Jie_reduction
        
        
    #---------------------------------------------------------------------------------
    # SET Js
    #---------------------------------------------------------------------------------   
    def set_Je_reduction(self):
        
        self.jee = self.jee - self.jee*self.Jee_reduction
        self.jie = self.jie - self.jie*self.Jie_reduction

        

    #---------------------------------------------------------------------------------
    # SET JplusAB BASED ON NETWORK TYPE
    #--------------------------------------------------------------------------------- 
    def update_JplusAB(self):
        
        if self.net_type == 'hom':
            
            self.JplusEE = 1.0       # EE intra-cluster potentiation factor
            self.JplusII = 1.0       # II intra-cluster potentiation factor
            self.JplusEI = 1.0       # EI intra-cluster potentiation factor
            self.JplusIE = 1.0       # IE intra-cluster potentiation factor
            
    #---------------------------------------------------------------------------------
    # SET ANY VARIABLES THAT ARE COMPLETELY DETERMINED BY MAIN INPUTS
    #---------------------------------------------------------------------------------       
    def set_dependent_vars(self):
                   
        # variables that depend on main inputs
        
        # total numbers of E and I neurons
        self.N_e = int(self.N*self.ne)                   
        self.N_i = int(self.N - self.N_e)       
        
        # number of connections
        self.Cee = self.pee*self.N_e
        self.Cei = self.pei*self.N_i
        self.Cii = self.pii*self.N_i
        self.Cie = self.pie*self.N_e
        
        self.Jee = self.jee/np.sqrt(self.N)
        self.Jie = self.jie/np.sqrt(self.N)
        self.Jei = self.jei/np.sqrt(self.N)
        self.Jii = self.jii/np.sqrt(self.N)
        
        self.Jie_ext = self.jie_ext/np.sqrt(self.N) 
        self.Jee_ext = self.jee_ext/np.sqrt(self.N) 
        self.Jii_ext = self.jii_ext/np.sqrt(self.N) 
        self.Jei_ext = self.jei_ext/np.sqrt(self.N)        
        
        if ((self.stim_shape == 'diff2exp')):
            
            self.stim_duration = []

 
    #---------------------------------------------------------------------------------
    # ADD CLUSTER SIZES
    #--------------------------------------------------------------------------------- 
    def set_popSizes(self, popSize_E, popSize_I):
        
        self.popSize_E = popSize_E.copy()
        self.popSize_I = popSize_I.copy()

    

    def set_external_inputs_ei(self, random_seed):
        
        print('currently no randomness in external inputs; seed not used')
        
        nu_ext_ee_base = self.mean_nu_ext_ee*self.pext_ee*self.N_e
        nu_ext_ee_pert = self.pert_mean_nu_ext_ee*self.pext_ee*self.N_e
        
        self.nu_ext_ee = nu_ext_ee_base*np.ones(self.N_e)
        self.pert_nu_ext_ee = nu_ext_ee_pert*np.ones(self.N_e)
        

        nu_ext_ie_base = self.mean_nu_ext_ie*self.pext_ie*self.N_e
        nu_ext_ie_pert = self.pert_mean_nu_ext_ie*self.pext_ie*self.N_e
        
        self.nu_ext_ie = (nu_ext_ie_base)*np.ones(self.N_i)
        self.pert_nu_ext_ie = (nu_ext_ie_pert)*np.ones(self.N_i)
        
        
        nu_ext_ei_base = self.mean_nu_ext_ei*self.pext_ei*self.N_i
        nu_ext_ei_pert = self.pert_mean_nu_ext_ei*self.pext_ei*self.N_i
        
        self.nu_ext_ei = (nu_ext_ei_base)*np.ones(self.N_e)
        self.pert_nu_ext_ei = (nu_ext_ei_pert)*np.ones(self.N_e)
        
        
        nu_ext_ii_base = self.mean_nu_ext_ii*self.pext_ii*self.N_i
        nu_ext_ii_pert = self.pert_mean_nu_ext_ii*self.pext_ii*self.N_i
        
        self.nu_ext_ii = (nu_ext_ii_base)**np.ones(self.N_i)
        self.pert_nu_ext_ii = (nu_ext_ii_pert)**np.ones(self.N_i)
   
              
        
    #---------------------------------------------------------------------------------
    # COMPUTE WHICH NEURONS ARE STIMULATED 
    #---------------------------------------------------------------------------------    
    def get_stimulated_neurons(self, random_seed, clust_sizeE, clust_sizeI):       
        
        # boolean arrays that denote which neurons receive stimulus
        self.stim_Ecells = np.zeros(self.N_e)
        self.stim_Icells = np.zeros(self.N_i)
        
        # set random number generator using the specified seed
        if random_seed == 'random':
            random_seed = np.random.choice(10000,1)
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng(random_seed)

        # get selective cluster ids
        selectiveClusters = self.selectiveClusters        
        
        # get assignment of neurons to clusters
        Ecluster_inds, Icluster_inds = fcn_compute_cluster_assignments(clust_sizeE,clust_sizeI)
        
        # loop over selective clusters
        for cluInd in selectiveClusters:
            
            #---------- Ecells -----------#
            
            # cells in this cluster
            cells_in_clu = np.where(Ecluster_inds == cluInd)[0]
            
            # number to select
            nstim = np.round(self.f_Ecells_target*np.size(cells_in_clu),0).astype(int)
            
            # randomly select fraction of them
            stim_cells = rng.choice(cells_in_clu, \
                                    size = nstim, \
                                    replace=False)
            
            # update array
            self.stim_Ecells[stim_cells] = True
            
            
            #---------- Icells -----------#
            
            # cells in this cluster
            cells_in_clu = np.where(Icluster_inds == cluInd)[0]
            
            # number to select
            nstim = np.round(self.f_Icells_target*np.size(cells_in_clu),0).astype(int)
            
            # randomly select fraction of them
            stim_cells = rng.choice(cells_in_clu, \
                                    size = nstim, \
                                    replace=False)
            
            # update array
            self.stim_Icells[stim_cells] = True       
           

    #---------------------------------------------------------------------------------
    # COMPUTE MAX STIMULUS STRENGTH
    # SET TO BE SOME FRACTION OF THE BASELINE EXTERNAL RATE mean_nu_ext
    #--------------------------------------------------------------------------------- 
    def set_max_stim_rate(self):
        
        self.stimRate_E = self.stim_rel_amp*self.mean_nu_ext_ee
        self.stimRate_I = self.stim_rel_amp*self.mean_nu_ext_ie
        
        
    #---------------------------------------------------------------------------------
    # SET INITIAL VOLTAGE
    #---------------------------------------------------------------------------------       
    def fcn_set_initialVoltage(self):
        
        print('initial voltages uniformly distributed between reset and threshold; could add rng seed as input here')
        iVe = np.random.uniform(self.Vr_e, self.Vth_e, self.N_e)
        iVi = np.random.uniform(self.Vr_i, self.Vth_i, self.N_i)
        self.iV = np.append(iVe, iVi)
