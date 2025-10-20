
#%% imports

import sys
import numpy as np

sys.path.append('../../')
import global_settings


#%% paths

sim_params_path = global_settings.path_to_src_code + 'run_simulations/'
func_path = global_settings.path_to_src_code + 'functions/'
func_path0 = global_settings.path_to_src_code + 'run_simulations/'
load_path = global_settings.path_to_sim_output + ''
save_path = global_settings.path_to_sim_output + 'decoding_analysis/'
load_path_plotting = global_settings.path_to_sim_output + 'decoding_analysis/'   
save_path_plotting = global_settings.path_to_sim_figures + 'decoding/'

#%% simulation details that always need to be specified
sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'
#sweep_param_name = 'zeroMean_sd_nu_ext_ee'
#sweep_param_name = 'same_eachClustersd_nu_ext_e_pert'
#net_type = 'baseEIclu'
net_type = 'baseHOM'
nNetworks = 10
   
#%% whether to load simulation parameters from an existing simParams file
load_from_simParams = True

#%% if loading from simParams file, specify simParams filename
#simParams_fname = 'simParams_050925_clu'
#simParams_fname = 'simParams_051325_clu'
simParams_fname = 'simParams_051325_hom'
#simParams_fname = 'simParams_012425_clu'

#%% if not loading from simParams file, need to specify different information
simID = 113020232105
nTrials = 30
stim_shape = 'diff2exp'
stim_type = ''
stim_rel_amp = 0.05
n_sweepParams = 1
swept_params_dict = {}
swept_params_dict['param_vals1'] = np.arange(0,0.45,0.05)


#%% decoding analysis parameters

# test analysis?
testing = False

# indNet begin
indNet_start = 0

# burn time
burnTime = 0

# window length/steps for computing spike counts
windStep = 20e-3

# classifier
#classifier = 'LDA'
classifier = 'LinearSVC'

# lda solver
lda_solver = 'svd'

# number of folds for cross-validation
nFolds = 5

# number of repetitions for repeated k-fold cross-validation
nReps = 10

# shuffle distribution
compute_shuffleDist = False

# number of shuffles per fold for null distribution
nShuffles = 500

# significance level
sig_level = 0.05

# shuffle percentile
shuffle_percentile = 95

# burn time
burnTime = 0.2

# for saving results
saveName_short = True

# window length
windL_vals = np.array([100e-3])

#ensembleSize_vals = np.array([1, 2, 4, 8, 16, 32])*19
ensembleSize_vals = np.array([160])

# rate threshold
rate_thresh = 0.

# number of cell subsamples
nSamples = 25

# whether to draw only stimulated cells
drawStimNeurons = False

# whether to draw the same number of cells/cluster
draw_equalPerCluster = True

# random seed
seed = 'random'

# chance level based on number of stimuli (5)
chance_level = 0.2

#%% cluster
maxCores = 48 # max number of cores to use
cores_per_job = 4 # number of cores/job; needs to be set ahead of time using OMP_NUM_THREADS

#%% plotting stuff
windL_plot = 100e-3
ensembleSize_plot = 1*19
plot_param_label = 'arousal [%]'
arousalInd_plot = 0
