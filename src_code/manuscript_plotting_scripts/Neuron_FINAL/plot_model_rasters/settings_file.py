

import numpy as np
import sys

sys.path.append('../../../')
import global_settings


# network to run
model = 'hom'

# path to simulation parameters
sim_params_path =  global_settings.path_to_src_code + 'run_simulations/'


# path to simulation and network generation functions
functions_path0 = global_settings.path_to_src_code + 'run_simulations/'
functions_path1 = global_settings.path_to_src_code + 'functions/'

# path to figures
fig_path = global_settings.path_to_manuscript_figs_final + 'model_rasters_baselineArousal/'

# simulation settings
arousalLevel = 0.
TF = 4.
stimOn = False
nPlot_perClusterE = 4
nPlot_perClusterI = 1

# seeds
externalInput_seed = np.random.choice(10000)
stimClusters_seed = np.random.choice(10000)
stimNeurons_seed = np.random.choice(1000)
networkSeed = np.random.choice(10000)


# sim params name
if model == 'cluster':
    sim_params_name = 'simParams_051325_clu'
    figID = 'Fig3A'
elif model == 'hom':
    sim_params_name = 'simParams_051325_hom'
    figID = 'Fig3B'