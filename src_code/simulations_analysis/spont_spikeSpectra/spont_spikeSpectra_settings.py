
#%% imports
import numpy as np
import sys

sys.path.append('../../')
import global_settings

#%% paths
sim_params_path = global_settings.path_to_src_code + 'run_simulations/'
func_path0 = global_settings.path_to_src_code + 'run_simulations/'
func_path = global_settings.path_to_src_code + 'functions/'
load_path = global_settings.path_to_sim_output + ''
save_path = global_settings.path_to_sim_output + 'spont_spikeSpectra_vsPerturbation/'
    
#%% simulation parameters
simParams_fname = 'simParams_051325_clu_spont'
sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'
net_type = 'baseEIclu'
nNetworks = 2

#%% analysis parameters

# window length for analysis
windL = 2500e-3
# time resolution
dt = 1e-3
# frequency resolution
df_array = np.array([0.8, 1.6, 4])

#%% cluster
maxCores = 48 # max number of cores to use
cores_per_job = 4 # number of cores/job; needs to be set ahead of time using OMP_NUM_THREADS
