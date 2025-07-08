
import numpy as np


#%% paths

sim_params_path = '/home/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/scripts/master_sims/'

func_path1 = '/home/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/scripts/functions/'

func_path2 = '/home/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/scripts/master_sims/'

load_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/'\
             'PROJ_VariabilityGainMod/data_files/test_stim_expSyn/')

save_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/'\
             'PROJ_VariabilityGainMod/data_files/test_stim_expSyn/clusterRates_numActiveClusters_sweepJeePlus/')
    
fig_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/' \
            'data_files/Figures/test_stim_expSyn/clusterRates_numActiveClusters_sweepJeePlus/')
    

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


