

# basic imports
import sys        
import numpy as np

# global settings
sys.path.append('../../')
import global_settings

# path to functions
sys.path.append(global_settings.path_to_src_code + 'data_analysis/')        

# functions
from fcn_processedh5data_to_dict import fcn_processedh5data_to_dict


#%% sessions to run and path to data

sessions_to_run = [\
                   'LA3_session3', \
                   'LA8_session1', 'LA8_session2', \
                   'LA9_session1', 'LA9_session3', 'LA9_session4', 'LA9_session5', \
                   'LA11_session1', 'LA11_session2', 'LA11_session3', 'LA11_session4',
                   'LA12_session1', \
                   'LA12_session2', 'LA12_session3', 'LA12_session4'
                   ]

data_path = global_settings.path_to_processed_data


#%% get number of cells in each session

n_goodUnits = np.zeros((len(sessions_to_run)))

for count, session_name in enumerate(sessions_to_run):

    session_info = fcn_processedh5data_to_dict(session_name, data_path)

    n_goodUnits[count] = session_info['nCells']


#%% print results
    
print('# good units in each session')
print(n_goodUnits)

# average
avg_n_goodUnits = np.mean(n_goodUnits)
print('avg # good units = %0.3f' % avg_n_goodUnits)
