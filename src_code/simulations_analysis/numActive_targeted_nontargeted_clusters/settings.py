
#%% imports
import numpy as np
import sys

sys.path.append('../../')
import global_settings

#%% for task spooler
maxCores = 48 # total number of cores to use
cores_per_job = 4 # number of cores/job; needs to be set ahead of time using OMP_NUM_THREADS

#%% paths
sim_params_path = global_settings.path_to_src_code + 'run_simulations/'
func_path = global_settings.path_to_src_code + 'functions/'
func_path1 = global_settings.path_to_src_code + 'run_simulations/'
load_path = global_settings.path_to_sim_output + ''
decoding_path = global_settings.path_to_sim_output + 'decoding_analysis/'   
save_path = global_settings.path_to_sim_output + 'num_active_targeted_nontargeted_clusters/'

#%% simulation parameters
simParams_fname = 'simParams_051325_clu'
sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'
net_type = 'baseEIclu'
nNetworks = 10

#%% analysis parameters
zscore = False
preStim_burn = 200e-3
window_length = 100e-3
window_step = 1e-3
window_std = 25e-3
gain_thresh_array = np.array([0, 1.0, 2, 3])
rate_thresh_array = np.array([8, 12, 15, 20, 30])
lifetimeThresh = 25e-3

#%% decoding params
decode_ensembleSize = 160
decode_windowSize = 100e-3
decode_type = 'LinearSVC'
decode_rateThresh = 0.

#%% plotting
gain_based = True

