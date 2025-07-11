
"""
parameters for decoding
"""

import sys
import numpy as np

sys.path.append('../../')
import global_settings


#-------------------- sessions to run -----------------------------------------#
sessions_to_run = [\
                   'LA3_session3', \
                   'LA8_session1', 'LA8_session2', \
                   'LA9_session1',\
                   'LA9_session3', 'LA9_session4', 'LA9_session5', \
                   'LA11_session1', 'LA11_session2', 'LA11_session3', 'LA11_session4', \
                   'LA12_session1', 'LA12_session2', 'LA12_session3', 'LA12_session4'
                    ]
    

# paths
data_path = global_settings.path_to_processed_data
decode_outpath_pupil = global_settings.path_to_data_analysis_output + 'decoding_pupil/'
decode_outpath_allTrials = global_settings.path_to_data_analysis_output + 'decoding_allTrials/'

# function paths
func_path1 = global_settings.path_to_src_code + 'data_analysis/'
func_path2 = global_settings.path_to_src_code + 'functions/'


# global pupil normalization
global_pupilNorm = False
rateDrift_cellSelection = False

# decoding general
n_kFolds = 5
n_kFold_reps = 10
n_decodeReps = 10
windSize = 100e-3
windStep = 10e-3
#decoderType = 'LDA'
decoderType = 'LinearSVC'
crossvalType = 'repeated_stratified_kFold'
runShuffle = False
nShuffles = 100
shuffle_percentile = 95


# additional parameters for LDA
lda_solver = 'svd'


# additional parameters for SVM


# trial window (defined relative to stimulus onset)
trial_window = [-100e-3, 600e-3]

# when splitting into resting and running
restBlock = 0
runBlock = 1
run_thresh = 2.0
runBlock_size = 1.
runBlock_step = 1.
runSpeed_method = 'avgSize_beforeStim'
runSplit_method = 'percentile'

# when splitting based on pupil size
pupilBin_lower = np.array([0, 0.34, 0.66])
pupilBin_upper = np.array([0.34, 0.66, 1])
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


# computing
maxCores = 40
cores_per_job = 4 # must be set ahead using max omp num threads