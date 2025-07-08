
"""
settings file 
"""

# imports for setting params
import numpy as np

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


# paths to functions
func_path1 = '/home/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/scripts/functions/'
func_path2 = '/home/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/scripts/MFT/funcs_MFT/basicEI_networks/'
func_path3 = '/home/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/scripts/MFT/funcs_MFT/basicEI_networks/effectiveMFT/'

# path to simParams
simParams_path = '/home/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/scripts/master_analysis/master_MFT/reduced_2cluster_nets/arousalSweep_noDisorder/'

# path to arousal parameters
arousalParams_path = '/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/test_stim_expSyn/simParams_mft/'
arousalParams_fname = '051300002025_clu_baseEIclu_sweep_Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread_stimType_diff2exp_stim_rel_amp0.050_simParams_mft_noDisorder.mat'

# filename
fName_begin = '051300002025_clu'

# paths
fig_outpath = (('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/Figures/MFT_2Ecluster/effectiveMFT_sweep_%s_2Ecluster/') % (fName_begin))
data_outpath = (('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/MFT_2Ecluster/effectiveMFT_sweep_%s_2Ecluster/') % (fName_begin))
