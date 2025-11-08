"""
parameters for running spike template amplitude based cell selection
"""

# PARAMETERS
parameters = {}

parameters['n_windows_drift'] = 4
parameters['window_length_baseEvoked'] = 100e-3
parameters['stim_duration'] = 25e-3
parameters['pupilSize_bin'] = 0.2
parameters['rateDrift_thresh_inc'] = 0.9
parameters['rateDrift_thresh_dec'] = 0.9
parameters['pval_soundResp'] = 0.01
parameters['psth_path'] = '/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/analysis_SuData/psth_allTrials/'


