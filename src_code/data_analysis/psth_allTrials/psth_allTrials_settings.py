
#%% basic imports
import sys

#%% settings
sys.path.append('../../')
import global_settings

#%% computing cluster
maxCores = 40 # total number of cores to use for analysis
cores_per_job = 4 # number of cores/job; needs to be set ahead of time using OMP_NUM_THREADS

#%% sessions to analyze
sessions_to_run = [\
                   'LA3_session3', \
                   'LA8_session1', 'LA8_session2', \
                   'LA9_session1', \
                   'LA9_session3', \
                   'LA9_session4', 'LA9_session5', \
                   'LA11_session1', 'LA11_session2', 'LA11_session3', 'LA11_session4', \
                   'LA12_session1', 'LA12_session2', 'LA12_session3', 'LA12_session4', \
                     
                  ]
    
#%% paths
data_path = global_settings.path_to_processed_data
outpath = global_settings.path_to_data_analysis_output + 'psth_allTrials/'
func_path1 = global_settings.path_to_src_code + 'data_analysis/'      
func_path2 = global_settings.path_to_src_code + 'functions/'   
         
#%% analysis parameters
       
# window for trials
trial_window = [-100e-3, 450e-3]

# baseline window
baseline_window = [-100e-3, 0]

# stimulus window
stimulus_window = [0, 150e-3]

# window length
window_length = 100e-3

# window_step
window_step = 1e-3

# data params
global_pupilNorm = False
highDownsample = False
cellSelection = ''
