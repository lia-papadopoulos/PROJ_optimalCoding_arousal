
#%% imports
import sys
import numpy as np

#%% global settings
sys.path.append('../../')
import global_settings

#%% cluster
maxCores = 40 # total number of cores to use for analysis
cores_per_job = 4 # number of cores/job needs to be set ahead of time using OMP_NUM_THREADS

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
outpath = global_settings.path_to_data_analysis_output + 'spont_spikeSpectra_pupil/'
func_path1 = global_settings.path_to_src_code + 'data_analysis/'      
func_path2 = global_settings.path_to_src_code + 'functions/'   

#%% analysis parameters
          
# main  
window_length_percentileComputation = 100e-3
split_based_on_evokedData = True
window_length = 2500e-3
dt = 1e-3
inter_window_interval = 0e-3
df_array = np.array([0.8, 1.6, 4])
avg_type = 2
stim_duration = 25e-3
nTrials_thresh = 2
n_subsamples = 50
pupilBlock_size = 0.1
pupilBlock_step = 0.1
pupilSplit_method = 'percentile'
pupilSize_method = 'avgSize_beforeStim'
# rest only?
restOnly = False
trialMatch = False
runThresh = 2.
runSpeed_method = 'avgSize_beforeStim'
runBlock_size = 1.
runBlock_step = 1.
runSplit_method = 'percentile'




