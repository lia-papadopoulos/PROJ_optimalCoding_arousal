
"""
master script for creating intermediate pre-processed dataset
"""
 
#%%

# imports
import sys

# import preprocessing functions
from fcn_no_cellSelection import fcn_no_cellSelection
from fcn_behavioral_preprocessing import fcn_run_behavioral_preprocessing
from fcn_behavioral_preprocessing import fcn_run_behavioral_preprocessing_globalPupilNorm
from fcn_get_stimulus_data import fcn_get_stimulus_data
from fcn_save_processed_data import fcn_save_processed_data

# import parameters
import behavioral_preprocessing_info


# paths
sys.path.append('../../')
import global_settings

# path to data from Su
original_data_path = global_settings.path_to_original_data
# where to save data
outpath = global_settings.path_to_processed_data

# session information
all_sessions_to_run = [
                       'LA3_session3', \
                       'LA8_session1', 'LA8_session2', \
                       'LA9_session1', \
                       'LA9_session3', 'LA9_session4', 'LA9_session5', \
                       'LA11_session1', 'LA11_session2', 'LA11_session3', 'LA11_session4', \
                       'LA12_session1', 'LA12_session2', 'LA12_session3', 'LA12_session4'
                      ]

    
# global pupil normalization (i.e., not per session)
globalPupilNorm = False
pupilNorm_factor = 0.7378132991114432

# highly downsampled
highDownsample = False
    
# end of filename
fname_end = '_no_cellSelection' + '_globalPupilNorm'*globalPupilNorm + '_downSampled'*highDownsample


#%% loop over sessions and process data

for indSession, session_name in enumerate(all_sessions_to_run):
    
    
    ##### cell selection #####################################################
    
    
    # run cell selection
    cellSelection_params, cellSelection_results = \
        fcn_no_cellSelection(session_name, original_data_path)    


    ##### behavioral preprocessing ############################################

    # behavioral preprocessing parameters
    behavioralPreprocessing_inputParams =  behavioral_preprocessing_info.params_dict[session_name]
    
    # run behavioral preprocessing
    if globalPupilNorm == False:
        behavioralPreprocessing_params, behavioralData = \
        fcn_run_behavioral_preprocessing(session_name, behavioralPreprocessing_inputParams, original_data_path) 
    else:
        behavioralPreprocessing_params, behavioralData = \
        fcn_run_behavioral_preprocessing_globalPupilNorm(session_name, behavioralPreprocessing_inputParams, original_data_path, pupilNorm_factor)         
        
    
    ##### extract stimulus data ##############################################
    
    stimulusInfo = fcn_get_stimulus_data(session_name, original_data_path)
    
    
    ##### save the data ######################################################
    fcn_save_processed_data(session_name, outpath, fname_end, \
                            cellSelection_params, cellSelection_results, \
                            behavioralPreprocessing_params, behavioralData, \
                            stimulusInfo)


    print('processing done. data saved.')
    print(session_name)