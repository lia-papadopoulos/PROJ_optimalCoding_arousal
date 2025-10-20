
import sys
import numpy as np

sys.path.append('../../')
import global_settings


#%% paths

sim_params_path = global_settings.path_to_src_code + 'run_simulations/'
func_path1 = global_settings.path_to_src_code + 'functions/'
func_path2 = global_settings.path_to_src_code + 'run_simulations/'
load_path = global_settings.path_to_sim_output + ''
save_path = global_settings.path_to_sim_output + 'clusterRates_numActiveClusters_sweepJeePlus/'   
    
#%% simulation params
simParams_fname = 'simParams_041725_clu_varyJEEplus'
sweep_param_name = 'JplusEE_sweep'
net_type = 'baseEIclu'
nNetworks = 5


#%% analysis parameters [all times in seconds]
burnTime_begin = 0.2
burnTime_end = 0.1
window_step = 1e-3
window_std = 25e-3
rate_thresh_array = np.array([0,1])


#%% cluster
maxCores = 48
cores_per_job = 4 # needs to be set ahead of time using OMP_NUM_THREADS


