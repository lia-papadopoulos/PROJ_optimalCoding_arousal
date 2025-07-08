
"""
settings file for singleCell_dPrime
"""

# imports for setting params
import numpy as np

#%% paths

sim_params_path = '/home/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/scripts/master_sims/'
func_path0 = '/home/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/scripts/master_sims/'
func_path = '/home/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/scripts/functions/'
load_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/test_stim_expSyn/') 
save_path = '/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/test_stim_expSyn/singleCell_dPrime/'
load_path_plotting = '/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/test_stim_expSyn/singleCell_dPrime/'
save_path_plotting = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/Figures/test_stim_expSyn/singleCell_dPrime/')

save_path_multivariate = '/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/test_stim_expSyn/multivariate_dPrime/'
load_path_plotting_multvariate = '/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/test_stim_expSyn/multivariate_dPrime/'
save_path_plotting_multvariate = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/Figures/test_stim_expSyn/multivariate_dPrime/')
    
#%% simulations params
#simParams_fname = 'simParams_040425_hom'
#simParams_fname = 'simParams_051325_clu'
#simParams_fname = 'simParams_051325_hom'
simParams_fname = 'simParams_050925_clu'

#sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'
sweep_param_name = 'zeroMean_sd_nu_ext_ee'

net_type = 'baseEIclu'
#net_type = 'baseHOM'
nNetworks = 5

#%% analysis
# window length/steps for computing spike counts
windL = 100e-3
windStep = 20e-3
tol = 1e-4

#%% cluster
maxCores = 48
cores_per_job = 4 # needs to be set ahead of time using OMP_NUM_THREADS

#%% plotting
rate_thresh = 0.
param_name_plot = 'arousal [%]'
base_window = np.array([-0.8, 0.])
stimCells_only = False

