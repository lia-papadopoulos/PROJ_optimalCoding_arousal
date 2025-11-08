
"""
Created on Tue Apr 22 17:52:24 2025
@author: liapapadopoulos
"""

import sys
import h5py
import numpy as np
from scipy.io import loadmat


sys.path.append('../../')
import global_settings

sys.path.append(global_settings.path_to_src_code + 'data_analysis/')       


def fcn_soundResp_cellSelection(session_name, params, data_path):
    
 
    # load in the preprocessed data
    f = h5py.File(data_path + session_name + '.mat','r')
    
    # extract data
    nCells = f['spk_Good_Aud'].shape[1]
    
    # cell spike times
    cell_spk_times = np.zeros(nCells,dtype='object')
    for i in range(0,nCells):
        spkTimes_ref = f['spk_Good_Aud'][0,i]
        cell_spk_times[i] = f[spkTimes_ref][0]
        
    # close file
    f.close()

    # unpack parameters
    pval_soundResp = params['pval_soundResp']
    psth_path = params['psth_path']
    nSig_tones_thresh = params['nSig_tones_thresh']
    
            
    # psth tone responsiveness
    psth_data = loadmat(( ('%spsth_allTrials_%s_windLength0.100s_no_cellSelection.mat') % (psth_path, session_name) ), simplify_cells=True)
    pval = psth_data['psth_pval_corrected']
    min_pval = np.min(pval, axis=1)
    min_pval_thresh = min_pval < pval_soundResp
    nSig_tones = np.sum(min_pval_thresh, 1)
    
    non_responsiveCells = np.nonzero(nSig_tones < nSig_tones_thresh)[0]
    
    # all bad units
    bad_units = non_responsiveCells
    
    # good units
    good_units = np.setdiff1d(np.arange(0,nCells), bad_units)
    
    # just keep good units
    cell_spk_times_good = cell_spk_times[good_units].copy()
    
    # return
    output_params = {}
    results = {}
    
    output_params['session_name'] = session_name
    output_params['preprocessed_data_path_Su'] = data_path
    output_params['pval_soundResp'] = pval_soundResp
    output_params['psth_path'] = psth_path
    

    results['bad_units'] = bad_units
    results['non_responsiveCells'] = non_responsiveCells
    results['good_units'] = good_units
    results['cell_spk_times'] = cell_spk_times_good
    
    
    return output_params, results