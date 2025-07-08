
"""
settings file
"""

import numpy as np

# for task spooler
maxCores = 48
cores_per_job = 4 # needs to be set ahead of time using OMP_NUM_THREADS

# paths for loading and saving data

sim_params_path = '/home/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/scripts/master_sims/'

load_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/'\
             'PROJ_VariabilityGainMod/data_files/test_stim_expSyn/')

save_path = '/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/test_stim_expSyn/num_active_targeted_nontargeted_clusters/'
    
fig_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/' \
            'data_files/Figures/test_stim_expSyn/clusterRates_numActiveClusters/')
    
# path to functions
func_path = '/home/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/scripts/functions/'
func_path1 = '/home/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/scripts/master_sims/'

        
# simulation parameters
simParams_fname = 'simParams_051325_clu'
sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'
net_type = 'baseEIclu'
nNetworks = 10



# analysis parameters
zscore = False
preStim_burn = 200e-3
window_length = 100e-3
window_step = 1e-3
window_std = 25e-3
gain_thresh_array = np.array([0, 1.0, 2, 3])
rate_thresh_array = np.array([8, 12, 15, 20, 30])
lifetimeThresh = 25e-3

# decoding 
decoding_path = '/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/test_stim_expSyn/decoding_analysis/'   
decode_ensembleSize = 160
decode_windowSize = 100e-3
decode_type = 'LinearSVC'
decode_rateThresh = 0.

# plotting
gain_based = True

