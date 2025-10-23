#%% imports
import sys

#%% global settings
sys.path.append('../../')
import global_settings

#%% sessions to run
sessions_to_run = [\
                   'LA3_session3', \
                   'LA8_session1', 'LA8_session2', \
                   'LA9_session1',\
                   'LA9_session3', 'LA9_session4', 'LA9_session5', \
                   'LA11_session1', 'LA11_session2', 'LA11_session3', 'LA11_session4', \
                   'LA12_session1', 'LA12_session2', 'LA12_session3', 'LA12_session4'
                    ]   

#%% paths
data_path = global_settings.path_to_processed_data
decode_outpath_pupil = global_settings.path_to_data_analysis_output + 'decoding_pupil/'
decode_outpath_allTrials = global_settings.path_to_data_analysis_output + 'decoding_allTrials/'
func_path1 = global_settings.path_to_src_code + 'data_analysis/'
func_path2 = global_settings.path_to_src_code + 'functions/'

#%% data selection
global_pupilNorm = False
cellSelection = ''
highDownSample = False

#%% analysis parameters

# trial window (defined relative to stimulus onset)
trial_window = [-100e-3, 600e-3]

# pupil
pupilSplit_method = 'percentile'
pupilSize_method = 'avgSize_beforeStim'
pupilBlock_size = 0.1
pupilBlock_step = 0.1
rest_only = False
# trial match between run and rest
# set to false if you just want to consider resting decoding vs pupil without worrying about running
trialMatch = False

# number of trials
nTrials_thresh = 50

# when splitting into resting and running
restBlock = 0
runBlock = 1
run_thresh = 2.0
runBlock_size = 1.
runBlock_step = 1.
runSpeed_method = 'avgSize_beforeStim'
runSplit_method = 'percentile'

# decoding
n_kFolds = 5
n_kFold_reps = 10
n_decodeReps = 10
windSize = 100e-3
windStep = 10e-3
decoderType = 'LinearSVC'
crossvalType = 'repeated_stratified_kFold'
runShuffle = False
nShuffles = 100
shuffle_percentile = 95
# additional parameters for LDA
lda_solver = 'svd'
# additional parameters for SVM

#%% cluster
maxCores = 48       # max number of cores to use
cores_per_job = 4   # number of cores/job; needs to be set ahead of time using OMP_NUM_THREADS