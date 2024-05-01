"""
@author: liapapadopoulos

fcn_preprocess_SuData

Set of functions to pre-process behavioral data from Su's neuropixel recordings

"""

import numpy as np
import h5py


#%% NORMALIZE DATA

'''
Normalize an Nx1 array of a behavioral variable vs time
Noramlization is done by dividing by max value, such that the output corresponds to the % of max

INPUTS
    behavioral_trace:           Nx1 array holding behavioral state values at sampled time points
    
OUTPUTS
    norm_behavioral_trace:      Nx1 array holding normalized behavioral state values at sampled time points

'''

def fcn_normalize_percentMax(behavioral_trace):
    
    norm_behavioral_trace = behavioral_trace/np.nanmax(behavioral_trace)
    
    return norm_behavioral_trace


#%% REMOVE ARTIFACTS FROM BEHAVIORAL TRACES

'''
INPUTS
    time_points:                    array of time points at which behavioral traces are sampled     
    pupil_trace:                    pupil trace vs time 
    run_trace:                      running trace vs time 
    whisk_trace:                    whisking trace vs time 
    remove_windows:                 array of time windows to remove from data
    diff_thresh:                    threshold on difference of consecutive pupil time points for first time point to be considered an artifact
    delta_t_compare:                at each time t, the pupil size at time t is compared to the pupil size at time t + delta_t_compare to determine if
                                    there's an artifact at time t
    artifact_window:                2x1 array; first element is amount of time before first detected artificat to remove from trace (should be <0)
                                    second element is amount of time after first detected artifact to remove from trace (should be >0)
OUPUTS:
    pupil_trace_corrected:          corrected pupil trace with artifacts removed
    run_trace_corrected:            corrected run trace with artifacts removed
    whisk_trace_corrected:          corrected whisk trace with artifacts removed
'''

def fcn_remove_pupilArtifacts(time_points, pupil_trace, run_trace, whisk_trace, \
                              remove_windows, diff_thresh, delta_t_compare, artifact_window):
    
    # sampling rate
    samp_rate = np.min(np.diff(time_points))
    
    
    # convert delta_t_compare to samples
    nInds_delta_t_compare = np.round(delta_t_compare/samp_rate).astype(int)
    
    # convert dt_before and dt_after to samples
    nInds_dt_before = np.round(artifact_window[0]/samp_rate).astype(int)
    nInds_dt_after = np.round(artifact_window[1]/samp_rate).astype(int)
    
    # difference in pupil trace values between two time points
    diff_pupil_trace = np.abs(pupil_trace[nInds_delta_t_compare:]-pupil_trace[:-nInds_delta_t_compare])
    
    # indices of pupil trace artifacts
    tInd_artifacts = np.nonzero(diff_pupil_trace >= diff_thresh)[0]
    
    # corrected pupil trace
    pupil_trace_corrected = pupil_trace.copy()
    run_trace_corrected = run_trace.copy()
    whisk_trace_corrected = whisk_trace.copy()
    
        
    # loop over remove_windows and nan-out those periods of time 
    nWinds = np.shape(remove_windows)[0]
    for iWind in range(0, nWinds):
        tStart = remove_windows[iWind,0]
        tEnd = remove_windows[iWind,1]
        iStart = np.argmin(np.abs(time_points-tStart))
        iEnd = np.argmin(np.abs(time_points-tEnd))
        pupil_trace_corrected[iStart:iEnd+1] = np.nan
        run_trace_corrected[iStart:iEnd+1] = np.nan
        whisk_trace_corrected[iStart:iEnd+1] = np.nan
            
        
    # loop over artifacts and nan-out the window around those artifacts
    
    # variable indicating if artifacts still remain
    artifacts_remaining = True
    
    # loop while artifacts remain
    while artifacts_remaining:
        
        # get the first artifact index
        tInd = tInd_artifacts[0]
        
        # convert to indices        
        tInd_begin = tInd + nInds_dt_before #(already includes - sign)
        tInd_end =  tInd + nInds_dt_after
        
        # remove artifacts from pupil trace
        pupil_trace_corrected[tInd_begin:tInd_end+1] = np.nan
        
        # remove artifacts from running and whisk traces
        run_trace_corrected[tInd_begin:tInd_end+1] = np.nan
        whisk_trace_corrected[tInd_begin:tInd_end+1] = np.nan
        
        # skip over any other artifacts within this window
        ind_remaining_artifacts = np.nonzero( tInd_artifacts >= tInd_end )[0]
        tInd_artifacts = tInd_artifacts[ind_remaining_artifacts]
        
        # update artifacts remaining
        if np.size(tInd_artifacts) == 0:
            artifacts_remaining = False    
    
        
    # return corrected pupil trace
    return pupil_trace_corrected, run_trace_corrected, whisk_trace_corrected





#%% SMOOTH + DOWNSAMPLE BEHAVIORAL DATA

'''
INPUTS
    time_points:                    array of time points at which behavioral data is sampled    
    pupil_trace:                    pupil trace vs time 
    run_trace:                      running trace vs time 
    whisk_trace:                    whisking trace vs time 
    window_length:                  length of window over which to average behavioral data [seconds]
    window_step:                    amount of time to slide window at each step [seconds]
        
OUPUTS:
    time_smooth:                    Nx1 array of time stamps corresponding to the center of the N windows [seconds]
    pupil_trace_smooth:             Nx1 array of pupil values; each value corresponds to window average
    run_trace_corrected:            Nx1 array of run speed values; each value corresponds to window average
    whisk_trace_corrected:          Nx1 array of whisk energy values; each value corresponds to window average
'''

def fcn_smooth_downsample_behavioralData(time_points, \
                                         pupil_trace, \
                                         run_trace, \
                                         whisk_trace, \
                                         window_length, window_step):
    
    # original sampling rate
    samp_rate = np.min(np.diff(time_points))    
    
    # half window length in samples
    wHalfLength_samples = np.round((window_length/2/samp_rate)).astype(int)
    
    # window step in samples
    wStep_samples = np.round((window_step/samp_rate)).astype(int)
    
    # number of windows [last window might span past end of data; in that case just average over existing data]
    nWindows = np.ceil(np.size(time_points)/wStep_samples).astype(int)
    
    # initialize arrays for smoothed traces
    time_smooth = np.zeros(nWindows)
    pupil_trace_smooth = np.zeros(nWindows)
    run_trace_smooth = np.zeros(nWindows)
    whisk_trace_smooth = np.zeros(nWindows)

    
    # start loop over behavioral traces
    for windInd in range(0, nWindows+1):
        
        indStart = windInd*wStep_samples
        indEnd = indStart + 2*wHalfLength_samples
        indCenter = indStart + wHalfLength_samples
        
        
        if indCenter > np.size(time_points):
            
            break
        
        if indEnd > np.size(time_points):
            
            indEnd = np.size(time_points)
            
           
        time_smooth[windInd] = time_points[indCenter]
        pupil_trace_smooth[windInd] = np.mean(pupil_trace[indStart:indEnd+1])
        run_trace_smooth[windInd] = np.mean(run_trace[indStart:indEnd+1])
        whisk_trace_smooth[windInd] = np.mean(whisk_trace[indStart:indEnd+1])
        
        
    
    # get data up to but not including the last checked window
    time_smooth = time_smooth[:windInd]
    pupil_trace_smooth = pupil_trace_smooth[:windInd]
    run_trace_smooth = run_trace_smooth[:windInd]
    whisk_trace_smooth = whisk_trace_smooth[:windInd]
                
    
    return time_smooth, pupil_trace_smooth, run_trace_smooth, whisk_trace_smooth
        

#%% RUN PRE-PROCESSING STEPS

'''
master function that runs pre-processing steps

INPUTS
    session_name:                   name of session to process    
    params:                         parameters dictionary for a particular session
    data_path:                      for for loading data


'''

def fcn_run_behavioral_preprocessing(session_name, params, data_path):
   
    
    # open the file   
    f = h5py.File(data_path + session_name + '.mat','r')

    # extract data
    time_stamps = f['time_stamp'][0]
    run_trace_raw = f['walk'][0]
    pupil_trace_raw = f['pupil'][0]
    whisk_trace_raw = f['whisk'][0]

    # running units
    run_trace_units = ''.join(chr(i[0]) for i in f[f['beh_units'][1,0]][:])
    pupil_trace_units = 'max_normalized'
    whisk_trace_units = 'max_normalized'
    

    # close file
    f.close()
    
    # get parameters
    remove_windows = params['remove_windows']
    artifact_thresh_pupil = params['artifact_thresh_pupil'] 
    delta_t_compare = params['delta_t_compare']
    artifact_window = params['artifact_window']
    smoothing_window_length = params['smoothing_window_length']
    smoothing_window_step = params['smoothing_window_step']
                           
    
    # normalize pupil
    pupil_trace_raw_norm = fcn_normalize_percentMax(pupil_trace_raw)
        
    # normalize whisk
    whisk_trace_raw_norm = fcn_normalize_percentMax(whisk_trace_raw)
            
    # remove artifacts based on pupil trace
    pupil_trace_corrected, run_trace_corrected, whisk_trace_corrected = \
    fcn_remove_pupilArtifacts(time_stamps, \
                              pupil_trace_raw_norm, run_trace_raw, whisk_trace_raw_norm,\
                              remove_windows, artifact_thresh_pupil, delta_t_compare, artifact_window)
             
    
    # smooth and downsample the data
    time_smooth, pupil_trace_smooth, run_trace_smooth, whisk_trace_smooth = \
    fcn_smooth_downsample_behavioralData(time_stamps, \
                                         pupil_trace_corrected, run_trace_corrected, whisk_trace_corrected, \
                                         smoothing_window_length, smoothing_window_step)
            
    # renormalize pupil
    pupil_trace_smooth_norm = fcn_normalize_percentMax(pupil_trace_smooth)  
        
    # renormalize whisk
    whisk_trace_smooth_norm = fcn_normalize_percentMax(whisk_trace_smooth)    
    
    
    # output parameters and data as dictionaries
    output_params = params.copy()
    results = {}
    results['time'] = time_smooth
    results['pupil_trace'] = pupil_trace_smooth_norm
    results['run_trace'] = run_trace_smooth
    results['whisk_trace'] = whisk_trace_smooth_norm
    results['run_trace_units'] = run_trace_units
    results['pupil_trace_units'] = pupil_trace_units
    results['whisk_trace_units'] = whisk_trace_units
    
    
    # return
    return output_params, results




