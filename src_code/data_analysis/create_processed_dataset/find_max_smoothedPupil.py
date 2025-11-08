
"""
find max smoothed pupil
"""
 
#%%

import numpy as np
import sys

# import preprocessing functions
from fcn_behavioral_preprocessing import fcn_find_max_smoothedPupilTrace

# import parameters
import behavioral_preprocessing_info


#%% sessions and path to data

all_sessions_to_run = [
                       'LA3_session3', \
                       'LA8_session1', 'LA8_session2', \
                       'LA9_session1', \
                       'LA9_session3', 'LA9_session4', 'LA9_session5', \
                       'LA11_session1', 'LA11_session2', 'LA11_session3', 'LA11_session4', \
                       'LA12_session1', 'LA12_session2', 'LA12_session3', 'LA12_session4'
                      ]

    
# path to data that Su processed
sys.path.append('../../')
import global_settings
processed_data_path = global_settings.path_to_original_data

#%% initialize

# max pupil
max_pupil = np.zeros(len(all_sessions_to_run))


#%% loop over sessions and process data

for indSession, session_name in enumerate(all_sessions_to_run):
    
    ##### behavioral preprocessing ############################################

    # behavioral preprocessing parameters
    behavioralPreprocessing_inputParams =  behavioral_preprocessing_info.params_dict[session_name]
    
    # run behavioral preprocessing
    max_smoothed_pupilTrace = fcn_find_max_smoothedPupilTrace(session_name, behavioralPreprocessing_inputParams, processed_data_path)

    max_pupil[indSession] = max_smoothed_pupilTrace

    print(session_name)

print(max_pupil)

print(np.max(max_pupil))