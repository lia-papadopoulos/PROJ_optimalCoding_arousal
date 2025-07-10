#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
decoding settings
"""

import numpy as np


#%% paths

sim_params_path = '/home/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/scripts/master_sims/'

func_path0 = '/home/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/scripts/master_sims/'


func_path = '/home/liap/PostdocWork_Oregon/My_Projects/' \
            'PROJ_VariabilityGainMod/scripts/functions/'

load_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/'\
             'PROJ_VariabilityGainMod/data_files/test_stim_expSyn/')
    
load_path_plotting = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/'\
                      'PROJ_VariabilityGainMod/data_files/test_stim_expSyn/decoding_analysis/')


save_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/'\
             'PROJ_VariabilityGainMod/data_files/test_stim_expSyn/decoding_analysis/')

save_path_plotting = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/'\
                     'PROJ_VariabilityGainMod/data_files/Figures/test_stim_expSyn/decoding/')
    


    
#%% simulations params

load_from_simParams = True

#%% simulation details always specified
sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'
#sweep_param_name = 'zeroMean_sd_nu_ext_ee'
#sweep_param_name = 'same_eachClustersd_nu_ext_e_pert'
#net_type = 'baseEIclu'
net_type = 'baseHOM'
nNetworks = 10
   
#%% if loading from simParams file, give simParams_fname
#simParams_fname = 'simParams_050925_clu'
#simParams_fname = 'simParams_051325_clu'
simParams_fname = 'simParams_051325_hom'
#simParams_fname = 'simParams_012425_clu'

#%% if not loading from sim params file, need to specify different information

simID = 113020232105
nTrials = 30
stim_shape = 'diff2exp'
stim_type = ''
stim_rel_amp = 0.05
n_sweepParams = 1
swept_params_dict = {}
swept_params_dict['param_vals1'] = np.arange(0,0.45,0.05)


#%% decoding


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

burnTime = 0.2

saveName_short = True

# window length
windL_vals = np.array([100e-3])

#ensembleSize_vals = np.array([1, 2, 4, 8, 16, 32])*19
ensembleSize_vals = np.array([160])

rate_thresh = 0.

nSamples = 25

drawStimNeurons = False

draw_equalPerCluster = True

seed = 'random'

chance_level = 0.2

#%%

# cluster
maxCores = 48
cores_per_job = 4 # needs to be set ahead of time using OMP_NUM_THREADS

#%%

# plotting
windL_plot = 100e-3
#ensembleSize_plot = np.array([19*1, 19*2, 19*4, 19*8, 19*16, 19*32])

ensembleSize_plot = 1*19

plot_param_label = 'arousal [%]'

arousalInd_plot = -1
