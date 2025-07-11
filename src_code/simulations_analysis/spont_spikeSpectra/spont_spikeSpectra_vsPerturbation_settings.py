
"""
settings file for spont_spikeSpectra_vsPerturbation
"""

# imports for setting params
import numpy as np
import sys

sys.path.append('../../')
import global_settings

# path to functions
sim_params_path = global_settings.path_to_src_code + 'run_simulations/'
func_path0 = global_settings.path_to_src_code + 'run_simulations/'
func_path = global_settings.path_to_src_code + 'functions/'
load_path = global_settings.path_to_sim_output + ''
save_path = global_settings.path_to_sim_output + 'spont_spikeSpectra_vsPerturbation/'
    
# for plotting
load_path_plotting = global_settings.path_to_sim_output + 'spont_spikeSpectra_vsPerturbation/'
save_path_plotting = global_settings.path_to_sim_figures + 'spont_spikeSpectra_vsPerturbation/'
    

    
#%% simulations params
   
simParams_fname = 'simParams_051325_clu_spont'
sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'
net_type = 'baseEIclu'
nNetworks = 2

# window length/steps for computing spike counts
windL = 2500e-3

# time step
dt = 1e-3

# frequency resolution
df_array = np.array([0.8, 1.6, 4])


#%% cluster


maxCores = 48
cores_per_job = 4 # needs to be set ahead of time using OMP_NUM_THREADS


#%% plotting

df_plot = 1.6
rate_thresh = 1.
estimation_plot = 'mt'
dcSubtract_type_plot = 0
lowFreq_band = np.array([1,4])