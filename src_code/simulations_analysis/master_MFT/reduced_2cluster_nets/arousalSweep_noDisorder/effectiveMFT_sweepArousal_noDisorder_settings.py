
"""
settings file 
"""

# imports for setting params
import numpy as np
import sys

sys.path.append('../../../../')
import global_settings

# for task spooler
maxCores = 48
cores_per_job = 4 # needs to be set ahead of time using OMP_NUM_THREADS

# sweep param name
sweep_param_name = 'Jee_reduction_nu_ext_ee_nu_ext_ie'

# number of swept parameters
n_sweepParams = 3

# load parameters from settings
load_arousalParams_from_settings = False

# swept parameter values
swept_params_dict = {}
swept_params_dict['param_vals1'] = np.linspace(0., 0.25, 13)
swept_params_dict['param_vals2']= np.linspace(0,1.75,13)
swept_params_dict['param_vals3'] = np.linspace(0,1.75,13)


# number of arousal values to sample from loaded parameters
nArousal_samples = 50

# factor by which to multiply loaded arousal parameters st they are in
# a relevant range for the 2-cluster network
arousal_multFactor = 0.35

# filenames
arousalParams_fname = '051300002025_clu_baseEIclu_sweep_Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread_stimType_diff2exp_stim_rel_amp0.050_simParams_mft_noDisorder.mat'
fName_begin = '051300002025_clu'

# paths to functions
func_path1 = global_settings.path_to_src_code + 'functions/'
func_path2 = global_settings.path_to_src_code + 'MFT/funcs_MFT/basicEI_networks/'
func_path3 = global_settings.path_to_src_code + 'MFT/funcs_MFT/basicEI_networks/effectiveMFT/'

# path to simParams
simParams_path = global_settings.path_to_src_code + 'simulations_analysis/master_MFT/reduced_2cluster_nets/arousalSweep_noDisorder/'

# path to arousal parameters
arousalParams_path = global_settings.path_to_sim_output + 'simParams_mft/'

# outpaths
fig_outpath = ((global_settings.path_to_2clusterMFT_figures + 'effectiveMFT_sweep_%s_2Ecluster/') % (fName_begin))
data_outpath = ((global_settings.path_to_2clusterMFT_output + 'effectiveMFT_sweep_%s_2Ecluster/') % (fName_begin))




