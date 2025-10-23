
#%% imports
import sys
import numpy as np

#%% global settings
sys.path.append('../../')
import global_settings

#%% cluster
maxCores = 40 # total number of cores to use for analysis
cores_per_job = 4 # number of cores/job; needs to be set ahead of time using OMP_NUM_THREADS

#%% sessions to run
sessions_to_run = [\
                   'LA3_session3', \
                   'LA8_session1', 'LA8_session2', \
                   'LA9_session1', \
                   'LA9_session3', 'LA9_session4', 'LA9_session5', \
                   'LA11_session1', 'LA11_session2', 'LA11_session3', 'LA11_session4', \
                   'LA12_session1', 'LA12_session2', 'LA12_session3', 'LA12_session4', \
                     
                  ]
    
#%% paths
data_path = global_settings.path_to_processed_data
outpath = global_settings.path_to_data_analysis_output + 'spont_evoked_correlations_pupil/'
func_path1 = global_settings.path_to_src_code + 'data_analysis/'      
func_path2 = global_settings.path_to_src_code + 'functions/'   

#%% analysis parameters
          
# z-scoring
zscore_withinPupil = True

# baseline subtraction
base_subtract = False

# data params
global_pupilNorm = False
highDownsample = False
cellSelection = ''

# trial window
window_length = 100e-3
trial_window_evoked = [-window_length, window_length]

# pupil size method
pupilSize_method = 'avgSize_beforeStim'

# number of trials needed
nTrials_thresh = 50

# number of subsamples
n_subsamples = 100

# pupil bins
pupilBlock_size = 0.1
pupilBlock_step = 0.1
pupilSplit_method = 'percentile'

#### for hierarchical clustering ####
linkage_method = 'average'
fcluster_criterion = 'distance'
cells_toKeep = 'allSigCells' 
rate_thresh = -np.inf
wind_length_clustering = 100e-3
sig_level = 0.05
runShuffle_corr = True
load_path_clustering = global_settings.path_to_data_analysis_output + 'spont_evoked_correlations_pupil/'
psth_path = global_settings.path_to_data_analysis_output + 'psth_allTrials/'
outpath_clustering = global_settings.path_to_data_analysis_output + 'spont_evoked_correlations_pupil/evoked_hClustering_pupilPercentile_combinedBlocks/'


    