'''
function for saving processed data as hd5f file
'''

import h5py
import numpy as np

def fcn_save_processed_data(session_name, outpath, \
                            cellSelection_params, cellSelection_results, \
                            behavioralPreprocessing_params, behavioralData, \
                            stimulusInfo):

    
    # create hdf5 file
    filename = outpath + session_name + '_processed_data.h5'
    hf = h5py.File(filename, 'w')

    # group for cell selection parameters
    grp1 = hf.create_group('cellSelection_params')
    for key, values in cellSelection_params.items():
        grp1.attrs[key] = values

    # group for behavioral preprocessing parameters
    grp2 = hf.create_group('behavioralPreprocessing_params')
    for key, values in behavioralPreprocessing_params.items():
        grp2.attrs[key] = values
        

    # cell spike times
    dt = h5py.vlen_dtype(np.dtype('float64'))
    hf.create_dataset('cell_spikeTimes', data=cellSelection_results['cell_spk_times'], dtype=dt)
    
    # good units
    hf.create_dataset('good_units', data=cellSelection_results['good_units'])

    # behavioral data    
    grpBehavioral = hf.create_group('behavioral_data')
    for key, values in behavioralData.items():
        if type(values) == str:
            grpBehavioral.attrs[key] = values
        else:
            grpBehavioral.create_dataset(key, data=values)

    # stimulus data
    grpStim = hf.create_group('stim_data')
    for key, values in stimulusInfo.items():
        if type(values) == str:
            grpStim.attrs[key] = values
        else:
            grpStim.create_dataset(key, data=values)   
        

    hf.close()


