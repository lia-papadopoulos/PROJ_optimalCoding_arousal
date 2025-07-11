
"""
settings file for spont_cvISI_vsPerturbation
"""

import sys

sys.path.append('../../')
import global_settings

sim_params_path = global_settings.path_to_src_code + 'run_simulations/'

func_path = global_settings.path_to_src_code + 'functions/'

func_path0 = global_settings.path_to_src_code + 'run_simulations/'

load_path = global_settings.path_to_sim_output + ''

load_path_plotting = global_settings.path_to_sim_output + 'spont_cvISI_vsPerturbation/'

save_path = global_settings.path_to_sim_output + 'spont_cvISI_vsPerturbation/'
    
fig_path = global_settings.path_to_sim_figures + 'spont_cvISI_vsPerturbation/'
    
    
#%% simulations params
   
simParams_fname = 'simParams_051325_clu_spont'
sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'
net_type = 'baseEIclu'
nNetworks = 2

# window length/steps for computing spike counts
windL = 2500e-3



#%% cluster

maxCores = 48
cores_per_job = 4 # needs to be set ahead of time using OMP_NUM_THREADS

#%% plotting

rate_thresh = 1.
