
import sys
import numpy as np

sys.path.append('../../')
import global_settings


# cluster
maxCores = 40
cores_per_job = 4 # needs to be set ahead of time using OMP_NUM_THREADS

# all sessions to run
sessions_to_run = [\
                   'LA3_session3', \
                   'LA8_session1', 'LA8_session2', \
                   'LA9_session1', \
                   'LA9_session3', 'LA9_session4', 'LA9_session5', \
                   'LA11_session1', 'LA11_session2', 'LA11_session3', 'LA11_session4', \
                   'LA12_session1', 'LA12_session2', 'LA12_session3', 'LA12_session4', \
                     
                  ]
    
    
# paths
data_path = global_settings.path_to_processed_data
outpath = global_settings.path_to_data_analysis_output + 'spont_evoked_correlations_pupil/'
func_path1 = global_settings.path_to_src_code + 'data_analysis/'      
func_path2 = global_settings.path_to_src_code + 'functions/'   

                
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

# pupil lag in seconds
pupilLag = 0.

# number of trials needed
nTrials_thresh = 50

# number of subsamples
n_subsamples = 100

# size of pupil percentile bins
pupilBlock_size = 0.1
pupilBlock_step = 0.1
pupilSplit_method = 'percentile'


#### for hclustering ####
linkage_method = 'average'
fcluster_criterion='distance'
cells_toKeep = 'allSigCells' 
#cells_toKeep = 'all'
rate_thresh = -np.inf
wind_length_clustering = 100e-3
sig_level = 0.05
runShuffle_corr = True
load_path_clustering = global_settings.path_to_data_analysis_output + 'spont_evoked_correlations_pupil/'
psth_path = global_settings.path_to_data_analysis_output + 'psth_allTrials/'
outpath_clustering = global_settings.path_to_data_analysis_output + 'spont_evoked_correlations_pupil/evoked_hClustering_pupilPercentile_combinedBlocks/'


#### for plotting specifically ####
path_to_cluster_info = global_settings.path_to_data_analysis_output + 'spont_evoked_correlations_pupil/figures/evoked_hClustering_pupilPercentile_combinedBlocks/'
fig_path_responseSim =  global_settings.path_to_data_analysis_output + 'spont_evoked_correlations_pupil/figures/evoked_hClustering_pupilPercentile_combinedBlocks/'
sigThresh_responseSim = 0.05
nPerms_responseSim = 1000
withinClu_avgType = 'v1'
include_selfCorr = False
betweenCorr = 'all' # all or good
responseSim_type = 'pearson'
clusterSize_cutoff = 2
clusterQuality_method = 'contrast'
nPerms = 5000
plot_results = False
sig_level_null = 0.05
fcluster_criterion = 'maxclust'
nullType = 'shuffle' # shuffle or permute
fig_path_clustering = global_settings.path_to_data_analysis_output + 'spont_evoked_correlations_pupil/figures/evoked_hClustering_pupilPercentile_combinedBlocks/'

    