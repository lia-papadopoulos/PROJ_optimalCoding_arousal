
"""
settings for clusterTimescale
"""

#-------------------- basic imports ----------------------------#
import sys
import numpy as np

sys.path.append('../../')
import global_settings


#-------------------- path for loading/saving data ----------------------------#

sim_params_path = global_settings.path_to_src_code + 'run_simulations/'
func_path = global_settings.path_to_src_code + 'functions/'
func_path0 = global_settings.path_to_src_code + 'run_simulations/'
load_path = global_settings.path_to_sim_output + ''
save_path = global_settings.path_to_sim_output + 'clusterTimescale/'   
fig_path = global_settings.path_to_sim_figures + 'clusterTimescale/'

#%% load from sim params or not
load_from_simParams = True


#%% simulation details always specified
#sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'
sweep_param_name = 'zeroMean_sd_nu_ext_ee'
#sweep_param_name = 'same_eachClustersd_nu_ext_e_pert'
net_type = 'baseEIclu'
#net_type = 'baseHOM'
nNetworks = 5

#%% if loading from simParams file, give simParams_fname
#simParams_fname = 'simParams_051325_clu_spontLong'
simParams_fname = 'simParams_050925_clu'

#%% if not loading from sim params file, need to specify different information

simID = 113020232105
nTrials = 30
nStim = 5
stim_shape = 'diff2exp'
stim_type = ''
stim_rel_amp = 0.05
n_sweepParams = 1
swept_params_dict = {}
swept_params_dict['param_vals1'] = np.arange(0,0.45,0.05)



#%% cluster params

maxCores = 48
cores_per_job = 4
    
#%% analysis params


gain_based = True

# analysis parameters [all times in seconds]
burnTime_begin = 0.2
burnTime_end = 0.1

window_step = 1e-3
window_std = 25e-3
if gain_based == False:
    rate_thresh_array = np.array([8, 10, 15, 20])
else:
    rate_thresh_array = np.array([0, 1])
    

rate_thresh_plot = 0

