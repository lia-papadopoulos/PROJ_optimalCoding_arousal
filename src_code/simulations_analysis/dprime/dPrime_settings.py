
#%% imports

import numpy as np
import sys

sys.path.append('../../')
import global_settings

#%% paths

sim_params_path = global_settings.path_to_src_code + 'run_simulations/'
func_path = global_settings.path_to_src_code + 'functions/'
func_path0 = global_settings.path_to_src_code + 'run_simulations/'
load_path = global_settings.path_to_sim_output + ''
save_path = global_settings.path_to_sim_output + 'singleCell_dPrime/'
load_path_plotting = global_settings.path_to_sim_output + 'singleCell_dPrime/'
save_path_plotting = global_settings.path_to_sim_figures + 'singleCell_dPrime/'

#%% simulation parameters

simParams_fname = 'simParams_051325_clu'
#simParams_fname = 'simParams_051325_hom'
#simParams_fname = 'simParams_050925_clu'

sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'
#sweep_param_name = 'zeroMean_sd_nu_ext_ee'

net_type = 'baseEIclu'
#net_type = 'baseHOM'
nNetworks = 10
#nNetworks = 5

#%% analysis
# window length/steps for computing spike counts
windL = 100e-3
windStep = 20e-3

#%% cluster
maxCores = 48 # total number of cores to use
cores_per_job = 4 # number of cores/job; needs to be set ahead of time using OMP_NUM_THREADS

#%% plotting
rate_thresh = 0.
param_name_plot = 'arousal [%]'
base_window = np.array([-0.8, 0.])
stimCells_only = False

