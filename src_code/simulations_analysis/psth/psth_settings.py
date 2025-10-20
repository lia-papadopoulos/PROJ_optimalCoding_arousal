
#%% imports

import sys

sys.path.append('../../')
import global_settings

#%% paths

sim_params_path = global_settings.path_to_src_code + 'run_simulations/'
func_path1 = global_settings.path_to_src_code + 'functions/'
func_path0 = global_settings.path_to_src_code + 'run_simulations/'
load_path = global_settings.path_to_sim_output + ''
save_path = global_settings.path_to_sim_output + 'psth/'
    
#%% simulations params

simParams_fname = 'simParams_051325_clu'
#simParams_fname = 'simParams_051325_hom'
sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'
net_type = 'baseEIclu'
#net_type = 'baseHOM'
nNetworks = 10
nStim = 5

#%% analysis parameters

# baseline window
base_window = [-0.8, 0]
# stimulus window
stim_window = [0, 0.2]
# bin size
binSize = 100e-3
# step size
stepSize = 5e-3
# burn time
burnTime = 0

#%% cluster

maxCores = 48 # total number of cores to use
cores_per_job = 4 # number of cores/job; needs to be set ahead of time using OMP_NUM_THREADS

