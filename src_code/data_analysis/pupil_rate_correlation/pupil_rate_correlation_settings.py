
"""
settings file
"""

import sys

sys.path.append('../../')
import global_settings


# cluster usage
maxCores = 56
cores_per_job = 4 # needs to be set ahead of time using OMP_NUM_THREADS

# sessions to run
all_sessions_to_run = ['LA3_session3', \
                       'LA8_session1', 'LA8_session2', \
                       'LA9_session1', \
                       'LA9_session3', 'LA9_session4', 'LA9_session5', \
                       'LA11_session1', 'LA11_session2', 'LA11_session3', 'LA11_session4', \
                       'LA12_session1', 'LA12_session2', 'LA12_session3', 'LA12_session4'
                      ]

    
# paths 
data_path = global_settings.path_to_processed_data
analyzed_data_path = global_settings.path_to_data_analysis_output + '/rate_pupil_run_correlations/'
fig_outpath = global_settings.path_to_data_analysis_output + '/rate_pupil_run_correlations/Figures/'
func_path1 = global_settings.path_to_src_code + 'data_analysis/'
func_path2 = global_settings.path_to_src_code + 'functions/'

# parameters
window_length = 100e-3
inter_window_interval = 0e-3
stim_duration = 25e-3
percentileBin_size = 0.1
rateDrift_cellSelection = False

# plotting
plot_low_cutoff = 0.25
plot_high_cutoff = 0.75
plot_sig_level = 0.05
plot_rate_thresh = 0.



