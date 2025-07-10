
"""
settings file for FF
"""

#%% paths

import sys
import numpy as np

sys.path.append('../../')
import global_settings


sim_params_path = global_settings.path_to_src_code + 'run_simulations/'
func_path = global_settings.path_to_src_code + 'functions/'
func_path0 = global_settings.path_to_src_code + 'run_simulations/'
func_path2 = global_settings.path_to_src_code + 'simulations_analysis/fano_factor'
load_path = global_settings.path_to_sim_output + ''
decode_path = global_settings.path_to_sim_output + 'decoding_analysis/'
save_path = global_settings.path_to_sim_output + 'fanofactor/'   
fig_path = global_settings.path_to_sim_figures + 'fano_factor_pre_evoked/raw_timecourse/'

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
nStim = 5
nTrials = 30
stim_shape = 'diff2exp'
stim_type = ''
stim_rel_amp = 0.05
n_sweepParams = 1
swept_params_dict = {}
swept_params_dict['param_vals1'] = np.arange(0,0.45,0.05)



#%% analysis params
windL = 100e-3
windStep = 20e-3
baseWind_burn = 200e-3
rate_thresh = 1.   
burnTime = 0.2
evoked_window_length = 0.2


#%% decoding params

decode_windL = 100e-3
decode_ensembleSize = 304
decode_rateThresh = 0.
decode_classifier = 'LinearSVC'

#%% cluster
maxCores = 48
cores_per_job = 4 # needs to be set ahead of time using OMP_NUM_THREADS


#%% plotting

t_eval_FFevoked = 't_min_allStim' # t_bestDecoding, t_min_eachStim, t_min_allStim
