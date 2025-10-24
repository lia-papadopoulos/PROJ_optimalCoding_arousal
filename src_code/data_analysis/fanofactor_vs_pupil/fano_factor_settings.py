
#%% imports
import sys
import numpy as np

#%% global settings
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
    
    
#%% paths
data_path = global_settings.path_to_processed_data
outpath = global_settings.path_to_data_analysis_output + 'fanofactor_pupil/'
func_path1 = global_settings.path_to_src_code + 'data_analysis/'
func_path2 = global_settings.path_to_src_code + 'functions/'

#%% analysis parameters

# windows
window_length = 100e-3
window_step = 1e-3
inter_window_interval = 0e-3
trial_window_evoked = [-window_length, 400e-3]

# if running multiple window sizes
window_length_sweep = np.array([50e-3, 100e-3, 200e-3])
window_length_percentileComputation = 100e-3

# stimulus duration
stim_duration = 25e-3

# pupil
pupilSize_method = 'avgSize_beforeStim'
pupilBlock_size = 0.1
pupilBlock_step = 0.1
pupilSplit_method = 'percentile'
pupilLag = 0.

# number of trials needed
nTrials_thresh = 25
# number of subsamples
n_subsamples = 100

# rest only
restOnly = False
trialMatch = False
runThresh = 2.
runSpeed_method = 'avgSize_beforeStim'
runBlock_size = 1.
runBlock_step = 1.
runSplit_method = 'percentile'

# data selection
global_pupilNorm = False
highDownsample = False
cellSelection = ''
    
#%% cluster
maxCores = 40 # total number of cores to use for analysis
cores_per_job = 4 # needs to be set ahead of time using OMP_NUM_THREADS