

import numpy as np

class mft_params:
    
    def __init__(self):     
        
        self.additional_externalVariance = 0.
        self.nSteps_MFT_DynEqs = 100000
        self.dt_MFT_DynEqs = 1e-4
        self.tau_e_MFT_DynEqs = np.array([1e-3,1e-3,1e-3])
        self.tau_i_MFT_DynEqs = np.array([1e-3])
        self.stopThresh_MFT_DynEqs = 1e-8
        self.plot_MFT_DynEqs = False
        self.nu_vec_uniform = np.array([30,30,1,16])
        self.nu_vec_cluster = np.array([60,0,1,16])
        self.inFocus_pops_e = np.array([0,1])
        self.inFocus_pops_i = np.array([])
        self.outFocus_pops_e = np.array([2])
        self.outFocus_pops_i = np.array([0])
