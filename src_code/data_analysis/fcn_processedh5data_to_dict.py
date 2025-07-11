

import h5py
import numpy as np

def fcn_processedh5data_to_dict(session_name, data_path, fname_end = ''):

          
    ########## load data ############################################################ 
    
    filename = (('%s%s%s_processed_data.h5') % (data_path, session_name, fname_end))
    data = h5py.File(filename, 'r')
    
    ########## data dict ############################################################ 
    data_dict = {}
    data_dict['session_name'] = session_name
    
    ########## behavioral data ############################################################ 
    
    data_dict['time_stamp'] = data['behavioral_data']['time'][:]
    data_dict['norm_pupilTrace'] = data['behavioral_data']['pupil_trace'][:]
    data_dict['walk_trace'] = data['behavioral_data']['run_trace'][:]

    ########## cell spike times ############################################################ 
    
    data_dict['cell_spk_times'] = data['cell_spikeTimes'][:]
    
    ########## stimulus data ############################################################
    
    data_dict['stim_on_time'] = data['stim_data']['stim_on_time'][:]
    data_dict['stim_freq'] = data['stim_data']['stim_Hz'][:]
    data_dict['spont_blocks'] = data['stim_data']['spont_blocks'][:]
    data_dict['stim_duration'] = data['stim_data'].attrs['stim_duration']
    
    ########## other data ############################################################
    data_dict['nTrials'] = np.size(data_dict['stim_freq'])
    data_dict['nCells'] = np.size(data_dict['cell_spk_times'])
    
    ########## close file ############################################################ 
    
    data.close()


    return data_dict