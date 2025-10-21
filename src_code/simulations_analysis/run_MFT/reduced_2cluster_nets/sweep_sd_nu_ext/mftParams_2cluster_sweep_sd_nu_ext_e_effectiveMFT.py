#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mftParams_2cluster_effective

"""

import numpy as np

class mft_params:

    
    # INIT METHOD
    def __init__(self):     
        
        #-----------------------------------------------------------------------------
        # DEFAULTS
        #-----------------------------------------------------------------------------   
     
        self.nSteps_MFT_DynEqs = 1000
        self.dt_MFT_DynEqs = 1e-4
        self.tau_e_MFT_DynEqs = np.array([1e-3,1e-3,1e-3])
        self.tau_i_MFT_DynEqs = np.array([1e-3])
        self.stopThresh_MFT_DynEqs = 1e-8
        self.plot_MFT_DynEqs = False
        self.low_w_lim = -20
        self.high_w_lim = 20.
        self.nu_bar_vec_selective = [60,1,1,14]
        self.nu_bar_vec_nonselective = [30,30,1,14]
        
        self.inFocus_pops_e = np.array([0,1])
        self.inFocus_pops_i = np.array([])
        self.outFocus_pops_e = np.array([2])
        self.outFocus_pops_i = np.array([0])

    
        # PRINT
        print('mft_params class initialized')