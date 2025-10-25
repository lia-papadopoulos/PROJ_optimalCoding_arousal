
#%% imports

# basic
import sys

# global seetings
sys.path.append('../../')
import global_settings

#%% paths
data_path = global_settings.path_to_processed_data
outpath = global_settings.path_to_data_analysis_output + 'singleCell_dPrime_pupil/'
func_path1 = global_settings.path_to_src_code + 'data_analysis/'      
func_path2 = global_settings.path_to_src_code + 'functions/'   

#%% whether you're loading data from .nwb or .h5
data_filetype = 'nwb' # nwb or h5

#%% sessions to analyze
sessions_to_run = [\
                   'LA3_session3', \
                   'LA8_session1', 'LA8_session2', \
                   'LA9_session1', \
                   'LA9_session3', 'LA9_session4', 'LA9_session5', \
                   'LA11_session1', 'LA11_session2', 'LA11_session3', 'LA11_session4', \
                   'LA12_session1', 'LA12_session2', 'LA12_session3', 'LA12_session4', \
                     
                  ]
    
#%% cluster
maxCores = 40 # total number of cores to for analysis
cores_per_job = 4 # number of cores/job; needs to be set ahead of time using OMP_NUM_THREADS

#%% analysis parameters

# stimulus duration
stim_duration = 25e-3

# window for trials
trial_window = [-100e-3, 450e-3]

# window length
window_length = 100e-3

# window_step
window_step = 10e-3

# pupil bins
pupilBlock_size = 0.1
pupilBlock_step = 0.1
pupilSplit_method = 'percentile'

# pupil size method
pupilSize_method = 'avgSize_beforeStim'

# n subsamples
n_subsamples = 100

# number of trials needed
nTrials_thresh = 20

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
cellSelection = ''
highDownSample = False



