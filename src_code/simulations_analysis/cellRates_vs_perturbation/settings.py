
"""
singleCell_tuning_to_perturbation_settings
"""

import sys
import numpy as np

sys.path.append('../../')
import global_settings

#-------------------- path for loading/saving data ----------------------------#


func_path = global_settings.path_to_src_code + 'functions/'
            
func_path2 = global_settings.path_to_src_code + 'run_simulations/'
            
simParams_path = global_settings.path_to_src_code + 'run_simulations/'
            
load_path = global_settings.path_to_sim_output + ''

save_path = global_settings.path_to_sim_output + 'singleCell_tuning_to_perturbation/'
    
fig_path = global_settings.path_to_sim_figures + 'singleCell_tuning_to_perturbation/'
    
#-------------------- parameters ----------------------------------------------#
   
    
#%% simulations params

load_from_simParams = True

#%% simulation details always specified
#sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'
sweep_param_name = 'zeroMean_sd_nu_ext_ee'
#sweep_param_name = 'same_eachClustersd_nu_ext_e_pert'
net_type = 'baseEIclu'
#net_type = 'baseHOM'
nNets = 5
   
#%% if loading from simParams file, give simParams_fname
#simParams_fname = 'simParams_051325_hom'
#simParams_fname = 'simParams_051325_clu'
#simParams_fname = 'simParams_012425_clu'
simParams_fname = 'simParams_050925_clu'

#%% if not loading from sim params file, need to specify different information

simID = 113020232105
nTrials = 30
stim_shape = 'diff2exp'
stim_type = ''
stim_rel_amp = 0.05
n_sweepParams = 1
swept_params_dict = {}
swept_params_dict['param_vals1'] = np.arange(0,0.45,0.05)



#%% analysis parameters

# burn time
startTime_base = 0.2

# sig level
sig_level = 0.05
