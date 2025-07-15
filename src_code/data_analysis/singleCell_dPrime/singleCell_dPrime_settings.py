
import sys

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
outpath = global_settings.path_to_data_analysis_output + 'singleCell_dPrime_pupil/'
func_path1 = global_settings.path_to_src_code + 'data_analysis/'      
func_path2 = global_settings.path_to_src_code + 'functions/'   

                
# window for trials
trial_window = [-100e-3, 450e-3]

# stimulus duration
stim_duration = 25e-3

# window length
window_length = 100e-3

# window_step
window_step = 10e-3

# size of pupil percentile bins
pupilBlock_size = 0.1
pupilBlock_step = 0.1
pupilSplit_method = 'percentile'

# n subsamples
n_subsamples = 100

# number of trials needed
nTrials_thresh = 20

# pupil size method
pupilSize_method = 'avgSize_beforeStim'

# rest only
restOnly = False
trialMatch = False
runThresh = 1.25
runSpeed_method = 'avgSize_beforeStim'
runBlock_size = 1.
runBlock_step = 1.
runSplit_method = 'percentile'

# rateDrift_cellSelection
rateDrift_cellSelection = False

# pupil normalization
global_pupilNorm = False

# downsampled version of data 
highDownsample = True


### for plotting ###

# pupil binning
binSize = 0.1
minPupil = 0
maxPupil = 1
nPupil_bins = 10

# parameters for good pupil range
minPupil_cutoff = 0.25
maxPupil_cutoff = 0.75

# rate thresh
rate_thresh = 0.

# for statistical testing
stat_test = 'wilcoxon'

# normalization type
norm_type = 'zscore'
    
# cell selection
rateDrift_cellSelection = False
    
# path to analyzed data
path_to_analyzed_data = global_settings.path_to_data_analysis_output + 'singleCell_dPrime_pupil/'

# figure path
fig_path = global_settings.path_to_data_analysis_output + 'singleCell_dPrime_pupil/Figures/'



