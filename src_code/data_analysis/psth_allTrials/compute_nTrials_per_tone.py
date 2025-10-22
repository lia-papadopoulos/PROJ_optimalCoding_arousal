
'''
compute minimum and maximum number of trials/tone before conditioning on arousal state
'''


#%%
# basic imports
import sys        
import numpy as np

# import settings file
import psth_allTrials_settings as settings

# paths to functions
sys.path.append(settings.func_path1)        
sys.path.append(settings.func_path2)
         
# main functions
from fcn_processedh5data_to_dict import fcn_processedh5data_to_dict
from fcn_SuData import fcn_makeTrials
from fcn_SuData import fcn_trialInfo_eachFrequency


#%% settings

# paths
data_path = settings.data_path
trial_window = settings.trial_window
global_pupilNorm = settings.global_pupilNorm
highDownsample = settings.highDownsample
cellSelection = settings.cellSelection

#%% USER INPUTS

# all sessions to run
sessions_to_run = [\
                   'LA3_session3', \
                   'LA8_session1', 'LA8_session2', \
                   'LA9_session1', \
                   'LA9_session3', \
                   'LA9_session4', 'LA9_session5', \
                   'LA11_session1', 'LA11_session2', 'LA11_session3', 'LA11_session4', \
                   'LA12_session1', 'LA12_session2', 'LA12_session3', 'LA12_session4', \
                     
                  ]


#%% SESSION INFO

data_name = '' + cellSelection + '_globalPupilNorm'*global_pupilNorm + '_downSampled'*highDownsample

#%% array of all nTrials
nTrials_all = np.array([])

#%% loop over sessions
for session_name in sessions_to_run:

    session_info = fcn_processedh5data_to_dict(session_name, data_path, fname_end = data_name)

    session_info['trial_window'] = trial_window

    print('session updated')

    session_info = fcn_makeTrials(session_info)

    session_info = fcn_trialInfo_eachFrequency(session_info)
    
    nTrials = np.rint(session_info['nTrials_eachFrequency']).astype(int)

    nTrials_all = np.append(nTrials_all, nTrials)

    print(session_name)


#%% min and max number of trials/tone across all sessions

min_nTrials = np.min(nTrials_all)
max_nTrials = np.max(nTrials_all)

#%% print the results

print(min_nTrials)
print(max_nTrials)
