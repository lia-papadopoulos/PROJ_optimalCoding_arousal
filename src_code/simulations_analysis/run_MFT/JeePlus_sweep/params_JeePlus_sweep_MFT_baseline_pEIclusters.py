
'''
specify all parameters required to run MFT
'''

# basic imports
import numpy as np
import sys

# global settings
sys.path.append('../../../')
import global_settings

# sim params path
sim_params_path = global_settings.path_to_src_code + 'run_simulations/'

# path for loading data
load_path = global_settings.path_to_sim_output + ''

# path for saving data
save_path = global_settings.path_to_sim_output + 'MFT_sweep_JeePlus_baseline/'

# function paths
func_path0 = global_settings.path_to_src_code + 'run_simulations/'
func_path1 = global_settings.path_to_src_code + 'functions/'
func_path2 = global_settings.path_to_src_code + '/MFT/funcs_MFT/clusteredNets_fullTheory/' 

# plotting
loadMFT_path = global_settings.path_to_sim_output + 'MFT_sweep_JeePlus_baseline/'
fig_path = global_settings.path_to_sim_figures + 'MFT_sweep_JeePlus_baseline/'

# simulation params
simParams_fname = 'simParams_041725_clu_varyJEEplus'
net_type = 'baseEIclu'
sweep_param_name = 'JplusEE_sweep'

# parameter index for which to run mft
indParam = 0

# mft parameters
mft_params_dict = {}
mft_params_dict['solve_reduced'] = True
mft_params_dict['nSteps_MFT_DynEqs'] = 2000
mft_params_dict['dt_MFT_DynEqs'] = 1e-4
mft_params_dict['nu_vec'] = np.nan
mft_params_dict['tau_e_MFT_DynEqs'] = 1e-3
mft_params_dict['tau_i_MFT_DynEqs'] = 1e-3
mft_params_dict['stopThresh_MFT_DynEqs'] = 1e-6
mft_params_dict['plot_MFT_DynEqs'] = False
mft_params_dict['min_JplusEE'] = 12
mft_params_dict['max_JplusEE'] = 25
mft_params_dict['delta_JplusEE'] = 0.025
mft_params_dict['nu_high_E'] = 80
mft_params_dict['nu_high_I'] = 60
mft_params_dict['nu_low_E'] = 1
mft_params_dict['nu_low_I'] = 1
mft_params_dict['stability_tau_e'] = 20e-3
mft_params_dict['stability_tau_i'] = 20e-3 
mft_params_dict['n_active_clusters_sweep'] = np.arange(1,12,1)

mft_reduced = True
