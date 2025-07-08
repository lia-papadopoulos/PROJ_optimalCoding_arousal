
import numpy as np


#%% paths

sim_params_path = '/home/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/scripts/master_sims/'

func_path1 = '/home/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/scripts/functions/'

func_path2 = '/home/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/scripts/master_sims/'

load_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/'\
             'PROJ_VariabilityGainMod/data_files/test_stim_expSyn/')

save_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/'\
             'PROJ_VariabilityGainMod/data_files/test_stim_expSyn/clusterRates_numActiveClusters/')
    
fig_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/' \
            'data_files/Figures/test_stim_expSyn/clusterRates_numActiveClusters/')
    

#%% simulations params

load_from_simParams = True

#%% simulation details always specified
#sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'
sweep_param_name = 'zeroMean_sd_nu_ext_ee'
#sweep_param_name = 'same_eachClustersd_nu_ext_e_pert'
net_type = 'baseEIclu'
#net_type = 'baseHOM'
nNetworks = 5
   
#%% if loading from simParams file, give simParams_fname
#simParams_fname = 'simParams_051325_clu'
#simParams_fname = 'simParams_012425_clu'
simParams_fname = 'simParams_050925_clu'

#%% if not loading from sim params file, need to specify different information

simID = 113020232105
nTrials = 30
nStim = 5
stim_shape = 'diff2exp'
stim_type = ''
stim_rel_amp = 0.05
n_sweepParams = 1
swept_params_dict = {}
swept_params_dict['param_vals1'] = np.arange(0,0.45,0.05)

#%% analysis parameters [all times in seconds]
burnTime_begin = 0.2
burnTime_end = 0.1
window_step = 1e-3
window_std = 25e-3
rate_thresh_array = np.array([0,1])


#%% cluster
maxCores = 48
cores_per_job = 4 # needs to be set ahead of time using OMP_NUM_THREADS


