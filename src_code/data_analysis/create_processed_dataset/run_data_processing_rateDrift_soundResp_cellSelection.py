
"""
master script for creating intermediate pre-processed dataset
using rate drift for cell selection
using sound responsiveness for cell selection
"""
 
#%%

# imports
import sys

# import preprocessing functions
from fcn_rateDrift_soundResp_cellSelection import fcn_rateDrift_soundResp_cellSelection
from fcn_behavioral_preprocessing import fcn_run_behavioral_preprocessing
from fcn_behavioral_preprocessing import fcn_run_behavioral_preprocessing_globalPupilNorm
from fcn_get_stimulus_data import fcn_get_stimulus_data
from fcn_save_processed_data import fcn_save_processed_data

# import parameters
import rateDrift_soundResp_params
import behavioral_preprocessing_info


# paths
sys.path.append('../../')
import global_settings

# original data from Su
processed_data_path = global_settings.path_to_original_data
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
    

# end of filename
if globalPupilNorm == False:
    fname_end = '_rateDrift_soundResp_cellSelection'
else:
    fname_end = '_rateDrift_soundResp_cellSelection_globalPupilNorm'


##%% loop over sessions and process data

for indSession, session_name in enumerate(all_sessions_to_run):
    
    
    ##### cell selection #####################################################
    
    # cell selection parameters and data path
    cellSelection_inputParams = rateDrift_soundResp_params.parameters
    
    # run rate drift cell selection
    cellSelection_params, cellSelection_results = \
        fcn_rateDrift_soundResp_cellSelection(session_name, cellSelection_inputParams, processed_data_path) 
    


    ##### behavioral preprocessing ############################################

    # behavioral preprocessing parameters
    behavioralPreprocessing_inputParams =  behavioral_preprocessing_info.params_dict[session_name]
    
    # run behavioral preprocessing
    if globalPupilNorm == False:
        behavioralPreprocessing_params, behavioralData = \
        fcn_run_behavioral_preprocessing(session_name, behavioralPreprocessing_inputParams, processed_data_path) 
    else:
        behavioralPreprocessing_params, behavioralData = \
        fcn_run_behavioral_preprocessing_globalPupilNorm(session_name, behavioralPreprocessing_inputParams, processed_data_path, pupilNorm_factor)         
        
    
    ##### extract stimulus data ##############################################
    
    stimulusInfo = fcn_get_stimulus_data(session_name, processed_data_path)
    
    
    ##### save the data ######################################################
    fcn_save_processed_data(session_name, outpath, fname_end, \
                            cellSelection_params, cellSelection_results, \
                            behavioralPreprocessing_params, behavioralData, \
                            stimulusInfo)


    print('processing done. data saved.')
    print(session_name)
