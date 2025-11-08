
"""
Created on Tue Apr 22 17:52:24 2025
@author: liapapadopoulos
"""

import sys
import h5py
import numpy as np

import behavioral_preprocessing_info
from fcn_behavioral_preprocessing import fcn_run_behavioral_preprocessing


sys.path.append('../../')
import global_settings

sys.path.append(global_settings.path_to_src_code + 'data_analysis/')       

from fcn_SuData import fcn_spikeTimes_trials_cells
from fcn_SuData import fcn_compute_spikeCnts_inTrials
from fcn_SuData import fcn_compute_avgPupilSize_inTrials

def fcn_rateDrift_cellSelection(session_name, params, data_path):
    
    
    ##### behavioral preprocessing ############################################

    # behavioral preprocessing parameters
    behavioralPreprocessing_inputParams =  behavioral_preprocessing_info.params_dict[session_name]
    
    behavioralPreprocessing_params, behavioralData = \
        fcn_run_behavioral_preprocessing(session_name, behavioralPreprocessing_inputParams, data_path)  
        
    
    time_smooth = behavioralData['time']
    pupil_trace = behavioralData['pupil_trace']
    run_trace = behavioralData['run_trace']
    whisk_trace = behavioralData['whisk_trace']
    
    # unpack parameters
    n_windows_drift = params['n_windows_drift']
    window_length_baseEvoked = params['window_length_baseEvoked']
    stim_duration = params['stim_duration']
    pupilSize_bin = params['pupilSize_bin']
    rateDrift_thresh = params['rateDrift_thresh']
    
    # trial window for baseline evoked activity
    trial_window_baseEvoked = [-window_length_baseEvoked, 0]

    # load in the preprocessed data
    f = h5py.File(data_path + session_name + '.mat','r')
    
    # extract data
    stim_on_time = np.sort(f['stim_on_time'][0])
    nCells = f['spk_Good_Aud'].shape[1]
    
    # cell spike times
    cell_spk_times = np.zeros(nCells,dtype='object')
    for i in range(0,nCells):
        spkTimes_ref = f['spk_Good_Aud'][0,i]
        cell_spk_times[i] = f[spkTimes_ref][0]
        
    # close file
    f.close()
        
    # update session info
    session_info = dict()

    session_info['session_name'] = session_name
    
    session_info['time_stamp'] = time_smooth
    session_info['norm_pupilTrace'] = pupil_trace
    session_info['norm_whiskTrace'] = whisk_trace
    session_info['walk_trace'] = run_trace
    
    session_info['stim_on_time'] = stim_on_time
    session_info['stim_duration'] = stim_duration
    session_info['nCells'] = nCells
    session_info['cell_spk_times'] = cell_spk_times
    
    
    trial_start_baseEvoked = []
    trial_end_baseEvoked = []
    
    
    for indTrial in range(0, len(stim_on_time)):
        
        t0 = stim_on_time[indTrial] + trial_window_baseEvoked[0]
        tF = stim_on_time[indTrial] + trial_window_baseEvoked[1]
        trial_start_baseEvoked = np.append(trial_start_baseEvoked, t0)
        trial_end_baseEvoked = np.append(trial_end_baseEvoked, tF)

    
    session_info['nTrials'] = np.size(trial_start_baseEvoked)
    session_info['trial_start'] = trial_start_baseEvoked
    session_info['trial_end'] = trial_end_baseEvoked
    
    session_info = fcn_spikeTimes_trials_cells(session_info)
    session_info = fcn_compute_spikeCnts_inTrials(session_info)
    
    trial_rates_baseEvoked = session_info['spkCounts_trials_cells']/window_length_baseEvoked
    
    avg_pupilSize_baseEvoked = fcn_compute_avgPupilSize_inTrials(session_info, trial_start_baseEvoked, trial_end_baseEvoked)
    
    
    # get median pupil size across all trials
    median_pupilSize = np.nanmedian(avg_pupilSize_baseEvoked)

    # get trials in bin of fixed sized centered around median
    middlePupil_trials_baseEvoked = np.nonzero((avg_pupilSize_baseEvoked > median_pupilSize - pupilSize_bin/2) &\
                                           (avg_pupilSize_baseEvoked < median_pupilSize + pupilSize_bin/2))[0]


    n_windows = n_windows_drift
    n_avg_trials = np.floor(len(middlePupil_trials_baseEvoked)/n_windows).astype(int)
    
    # average rate in each block
    avgRate_midPupil_trials_baseEvoked = np.zeros((nCells, n_windows))
    
    # percent change
    pctChange = np.zeros((nCells))
    
    # drifting cells
    driftingCells = np.array([])  
    
    
    # loop over blocks
    for indBlock in range(0,n_windows):
    
        indStart = indBlock*n_avg_trials 
        indEnd = indStart + n_avg_trials
        indTrials = middlePupil_trials_baseEvoked[indStart:indEnd]
        avgRate_midPupil_trials_baseEvoked[:, indBlock] = np.mean(trial_rates_baseEvoked[indTrials, :], axis=0)
    

    for indCell in range(0, nCells):
        pctChange[indCell] = (np.abs(avgRate_midPupil_trials_baseEvoked[indCell, 0]-avgRate_midPupil_trials_baseEvoked[indCell, -1]))/avgRate_midPupil_trials_baseEvoked[indCell, 0]

    # driftingCells
    driftingCells = np.nonzero(pctChange > rateDrift_thresh)[0]
    
    # good units
    good_units = np.setdiff1d(np.arange(0,nCells), driftingCells)
    
    # just keep good units
    cell_spk_times_good = cell_spk_times[good_units].copy()
    
    # return
    output_params = {}
    results = {}
    
    output_params['session_name'] = session_name
    output_params['preprocessed_data_path_Su'] = data_path
    output_params['n_windows_drift'] = n_windows_drift
    output_params['window_length_baseEvoked'] = window_length_baseEvoked
    output_params['pupilSize_bin'] = pupilSize_bin
    output_params['rateDrift_thresh'] = rateDrift_thresh    


    results['good_units'] = good_units
    results['cell_spk_times'] = cell_spk_times_good
    
    
    return output_params, results