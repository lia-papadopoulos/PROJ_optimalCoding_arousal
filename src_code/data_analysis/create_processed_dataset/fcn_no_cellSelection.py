
'''
keep all units from su
'''

#%%

import numpy as np
import h5py


def fcn_no_cellSelection(session_name, processed_data_path):

    
    
    # load in the preprocessed data
    f = h5py.File(processed_data_path + session_name + '.mat','r')
    
    # extract data
    nCells = f['spk_Good_Aud'].shape[1]
        
    # cell spike times
    cell_spk_times = np.zeros(nCells,dtype='object')
    for i in range(0,nCells):
        spkTimes_ref = f['spk_Good_Aud'][0,i]
        cell_spk_times[i] = f[spkTimes_ref][0]
    
    # close file
    f.close()
        
    good_units = np.arange(0, nCells)
    
            
    #%% output results as dictionary
    
    output_params = {}
    results = {}
    
    output_params['session_name'] = session_name
    output_params['preprocessed_data_path_Su'] = processed_data_path

    results['good_units'] = good_units
    results['cell_spk_times'] = cell_spk_times
    
    return output_params, results
