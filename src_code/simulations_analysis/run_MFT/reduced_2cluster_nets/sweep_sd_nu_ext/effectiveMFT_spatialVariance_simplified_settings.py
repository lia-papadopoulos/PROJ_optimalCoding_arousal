
"""
settings file 
"""

# imports for setting params
import numpy as np

# for task spooler
maxCores = 48
cores_per_job = 4 # needs to be set ahead of time using OMP_NUM_THREADS

# name of parameter swept over
sweep_param_name = "sd_nu_ext_e_pert"

# param vals (must match array in launch jobs)
param_vals = np.arange(0.0, 0.3, 0.025)


# paths to functions
func_path1 = '/home/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/scripts/functions/'
func_path2 = '/home/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/scripts/MFT/funcs_MFT/basicEI_networks/'
func_path3 = '/home/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/scripts/MFT/funcs_MFT/basicEI_networks/effectiveMFT/'

# path to simParams
simParams_path = '/home/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/scripts/master_analysis/master_MFT/reduced_2cluster_nets/sweep_sd_nu_ext/'


# filename
fName_begin = '02052024_manuscriptLP'

# paths
fig_outpath = (('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/Figures/MFT_2Ecluster/effectiveMFT_sweep_%s_2Ecluster/') % (sweep_param_name))
data_outpath = (('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/MFT_2Ecluster/effectiveMFT_sweep_%s_2Ecluster/') % (sweep_param_name))
