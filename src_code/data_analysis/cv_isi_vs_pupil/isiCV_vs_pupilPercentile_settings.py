
#%% imports
import sys

sys.path.append('../../')
import global_settings


#%% sessions to run
all_sessions_to_run = ['LA3_session3', \
                       'LA8_session1', 'LA8_session2', \
                       'LA9_session1', \
                       'LA9_session3', 'LA9_session4', 'LA9_session5', \
                       'LA11_session1', 'LA11_session2', 'LA11_session3', 'LA11_session4', \
                       'LA12_session1', 'LA12_session2', 'LA12_session3', 'LA12_session4'
                      ]
    

#%% for cluster
maxCores = 40 # total number of cores to use for analysis
cores_per_job = 4 # number of cores/job; needs to be set ahead of time using OMP_NUM_THREADS

#%% paths
func_path1 = global_settings.path_to_src_code + 'data_analysis/'
data_path = global_settings.path_to_processed_data
outpath = global_settings.path_to_data_analysis_output + 'isiCV_pupil/'

#%% analysis parameters

# for pupil bins
bins_from_evokedTrials = True
window_length_percentileComputation = 100e-3
pupilBlock_size = 0.1
pupilBlock_step = 0.1
pupilSplit_method = 'percentile'
pupilSize_method = 'avgSize_beforeStim'

# for cvisi 
window_length = 2500e-3
window_step = 1e-3
inter_window_interval = 0e-3
nTrials_thresh = 2
n_subsamples = 100

# other
stim_duration = 25e-3

# data set loading  parameters
cellSelection = ''
global_pupilNorm = False
highDownsample = False

# rest only
restOnly = False
trialMatch = False
runThresh = 2.
runSpeed_method = 'avgSize_beforeStim'
runBlock_size = 1.
runBlock_step = 1.
runSplit_method = 'percentile'