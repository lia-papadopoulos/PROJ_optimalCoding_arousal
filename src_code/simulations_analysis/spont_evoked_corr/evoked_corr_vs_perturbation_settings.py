

import sys

sys.path.append('../../')
import global_settings

# path to functions
sim_params_path = global_settings.path_to_src_code + 'run_simulations/'
func_path0 = global_settings.path_to_src_code + 'run_simulations/'
func_path1 = global_settings.path_to_src_code + 'functions/'
func_path2 = global_settings.path_to_src_code + 'data_analysis/'
decode_path = global_settings.path_to_sim_output + 'decoding_analysis/'
load_path = global_settings.path_to_sim_output + ''
save_path = global_settings.path_to_sim_output + 'evoked_corr/'
    
# for clustering
corr_path = global_settings.path_to_sim_output + 'evoked_corr/'
psth_path =  global_settings.path_to_sim_output + 'psth/'
sim_path = global_settings.path_to_sim_output + ''
cluster_outpath = global_settings.path_to_sim_output + 'evoked_corr/hClustering/'
    
    
#%% simulations params
   
simParams_fname = 'simParams_051325_clu'
#simParams_fname = 'simParams_051325_hom'
sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'
net_type = 'baseEIclu'
#net_type = 'baseHOM'
nNetworks = 10


# window length/steps for computing spike counts
windL = 100e-3
windStep = windL
baseWind_burn = 0.2

# number of cells to subsample
nCells_sample = 144


# nShuffles
nShuffles = 100
nSamples = 10


# decoding info
decode_windL = 100e-3
decode_ensembleSize = 160
decode_rateThresh = 0.
decode_classifier = 'LinearSVC'

# decode window or after stim window
use_decode_window = False


#%% cluster job submission

# for task spooler
maxCores = 48
cores_per_job = 4 # needs to be set ahead of time using OMP_NUM_THREADS

