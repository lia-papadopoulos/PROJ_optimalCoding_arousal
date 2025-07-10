
"""
mftParams_2cluster_effective

"""

import numpy as np
import sys

sys.path.append('../../../../')
import global_settings

class mft_params:

    
    # INIT METHOD
    def __init__(self):     
        
        #-----------------------------------------------------------------------------
        # DEFAULTS
        #-----------------------------------------------------------------------------   
        self.newRun = False
        self.n_Epops = 3
        self.n_Ipops = 1
        self.additional_externalVariance = 0.0
        self.dt_MFT_DynEqs = 1e-4
        self.tau_e_MFT_DynEqs = np.array([1e-3,1e-3,1e-3])
        self.tau_i_MFT_DynEqs = np.array([1e-3])
        self.stopThresh_MFT_DynEqs = 1e-8
        self.plot_MFT_DynEqs = False
        self.nSteps_MFT_DynEqs = 30000
        self.nu_vec_selective = [60, 0, 1, 16]
        self.nu_vec_nonselective = [30, 30, 1, 16]
    
        print('mft_params class initialized')
                
        # JpluseEE values
        self.JplusEE_array = np.arange(15, 22.05, 0.05)

        # paths to functions
        self.func_path1 = global_settings.path_to_src_code + 'functions/'
        self.func_path2 = global_settings.path_to_src_code + 'MFT/funcs_MFT/basicEI_networks/'
        
        # path to simParams (used when newRun = True)
        self.simParams_path = global_settings.path_to_src_code + 'simulations_analysis/master_MFT/reduced_2cluster_nets/arousalSweep_noDisorder/'

        # filenames
        self.fName_begin = '051300002025_clu'

        # path and filenames when we want to load parameters from file
        self.fName_begin_paramData = '051300002025_clu'
        self.pathName_end_paramData = 'effectiveMFT_sweep_%s_2Ecluster' % self.fName_begin_paramData
        self.pathName_paramData = (global_settings.path_to_2clusterMFT_output + '%s/' % (self.pathName_end_paramData))

        # paths        
        self.fig_outpath = global_settings.path_to_2clusterMFT_figures + 'MFT_sweep_JeePlus_2Ecluster/'
        self.data_outpath = global_settings.path_to_2clusterMFT_output + 'MFT_sweep_JeePlus_2Ecluster/'
        
        
