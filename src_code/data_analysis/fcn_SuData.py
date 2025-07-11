
"""
fcn_SuData
"""

import numpy as np
import sys
from nitime.utils import dpss_windows
import scipy.fft

# global settings
sys.path.append('../')
import global_settings


# decoding functions
sys.path.append(global_settings.path_to_src_code + 'functions/')  
from fcn_decoding import fcn_stratified_kFold_crossVal
from fcn_decoding import fcn_repeated_stratified_kFold_crossVal


#%% TRIAL START AND END TIMES

def fcn_makeTrials(session):
        
    # number of trials
    nTrials = session['nTrials']
    
    # trial window
    trial_window = session['trial_window'].copy()
    
    # stimulus onset
    stim_on_time = session['stim_on_time'].copy()
    
    # intertrial interval
    ITIs = np.diff(stim_on_time)
    
    # check trial window against smallest iti
    if (np.sum(np.abs(trial_window)) > np.min(ITIs)):
       
        sys.exit('trial window is longer than smallest ITI')

    #--------------------------- trial start/end ------------------------- #
    trial_start = np.zeros(nTrials)
    trial_end = np.zeros(nTrials)
    
    # loop over trials
    for trialInd in range(0,nTrials,1):
        
        trial_start[trialInd] = stim_on_time[trialInd] + trial_window[0]
        trial_end[trialInd] = stim_on_time[trialInd] + trial_window[1] 
        
        
    # update session
    session['trial_start'] = trial_start
    session['trial_end'] = trial_end

    # return
    return session


#%% TRIAL START AND END TIMES -- SPONTANEOUS BLOCKS

def fcn_makeTrials_spont(session, window_length, inter_window_interval):
    
    # unpack session
    n_spontBlocks = session['n_spontBlocks']
    spontBlock_start = session['spontBlock_start'].copy()
    spontBlock_end = session['spontBlock_end'].copy()
    
    # start and end of trials
    trial_start = []
    trial_end = []

    # loop over spontaneous blocks
    for ind_spontBlock in range(0, n_spontBlocks):
        
        blockStart = spontBlock_start[ind_spontBlock]
        blockEnd = spontBlock_end[ind_spontBlock]
        
            
        t0 = np.arange(blockStart, blockEnd - window_length, window_length + inter_window_interval)
                
        tF = t0 + window_length
        
        trial_start = np.append(trial_start, t0)
        trial_end = np.append(trial_end, tF)  
        
        
    session['nTrials'] = np.size(trial_start)
    session['trial_start'] = trial_start
    session['trial_end'] = trial_end
    

    return session


#%% FIND ALL TRIALS THAT DON'T HAVE BEHAVIORAL STATE VARIABLES DUE TO ARTIFACT REMOVAL

'''
inputs
    trial_start:            start time of each trial
    trial_end:              end time of each trial
    time_stamp:             time stamp of each behavioral state measurement
    pupil_trace_corrected:  pupil trace 
                            [if the corrected pupil trace, will contain nan 
                             values at certain times where an artifact was
                             detected during the preprocessing steps]
outputs:
    bad_trials:             trials during which a nan value in the pupil trace
                            detected
'''

def fcn_find_badTrials(trial_start, trial_end, time_stamp, pupil_trace):
        
    nTrials = np.size(trial_start)
    
    bad_trials=np.array([])
    
    # loop over trials
    for trialInd in range(0,nTrials):
        
        tInd_start = np.argmin(np.abs(time_stamp-trial_start[trialInd]))
        tInd_end = np.argmin(np.abs(time_stamp-trial_end[trialInd]))
        pupil_in_trial = pupil_trace[tInd_start:tInd_end+1]
        
        if np.any(np.isnan(pupil_in_trial)):
            
            bad_trials = np.append(trialInd, bad_trials)
            
        bad_trials = bad_trials.astype(int)
            
    return bad_trials


#%% TRIALS CORRESPONDING TO EACH STIMULUS FREQUENCY

def fcn_trialInfo_eachFrequency(session):
    
    trial_start = session['trial_start']
    trial_end = session['trial_end']
    time_stamp = session['time_stamp']
    pupil_trace = session['norm_pupilTrace']
    
    stim_freq = session['stim_freq'].copy()
    
    unique_freq = np.unique(stim_freq)
    
    nFreqs = np.size(unique_freq)
    
    trials_freq = np.zeros(nFreqs,dtype='object')
    
    nTrials_freq = np.zeros(nFreqs)
        
    # get bad trials that don't have behavioral data
    bad_trials = fcn_find_badTrials(trial_start, trial_end, time_stamp, pupil_trace)
    
    
    for ind_freq in range(0,nFreqs):
        
        freq = unique_freq[ind_freq]
        
        all_trials_thisFrequency = np.nonzero(stim_freq == freq)[0]
        
        bad_trials_thisFrequency = np.intersect1d(bad_trials, all_trials_thisFrequency)
        
        all_trials_thisFrequency = np.setdiff1d(all_trials_thisFrequency, bad_trials_thisFrequency)
        
        trials_freq[ind_freq] = all_trials_thisFrequency
        
        nTrials_freq[ind_freq] = np.size(all_trials_thisFrequency)
    
    session['unique_frequencies'] = unique_freq
    session['n_frequencies'] = nFreqs
    session['trialInds_eachFrequency'] = trials_freq
    session['nTrials_eachFrequency'] = nTrials_freq
    
    # return
    return session


#%% COMPUTE AVERAGE PUPIL SIZE IN EACH TRIAL

def fcn_compute_avgPupilSize_inTrials(session, t_beginAvg, t_endAvg):
    
    # unpack session
    tPts = session['time_stamp'].copy()
    pupilSize = session['norm_pupilTrace'].copy()     

    # number of trials
    nTrials = np.size(t_beginAvg)    

    # average pupil size in each trial

    avg_pupilSize = np.zeros(nTrials)
      
    # loop over trials
    for trialInd in range(0,nTrials,1):
         
        # indices of pupil data for this trial
        trial_pupilInds = np.nonzero( (tPts >= t_beginAvg[trialInd]) & \
                                      (tPts <= t_endAvg[trialInd]) )[0]
            
        # pupil size in this trial
        trial_pupilSize = pupilSize[trial_pupilInds]
        
        # average size in trial
        # will be nan if any nan values in trial
        avg_pupilSize[trialInd] = np.mean(trial_pupilSize)
        
    # return
    return avg_pupilSize


#%% COMPUTE AVERAGE WHISK ENERGY IN EACH TRIAL

def fcn_compute_avgWhiskEnergy_inTrials(session, t_beginAvg, t_endAvg):
    
    # unpack session
    
    nTrials = session['nTrials']
    tPts = session['time_stamp'].copy()
    whiskEnergy = session['norm_whiskTrace'].copy()     

    
    # average whisk energy in each trial

    avg_whiskEnergy = np.zeros(nTrials)
      
    # loop over trials
    for trialInd in range(0,nTrials,1):
         
        # indices of whisk data for this trial
        trial_whiskInds = np.nonzero( (tPts >= t_beginAvg[trialInd]) & \
                                      (tPts <= t_endAvg[trialInd]) )[0]
            
        # whisk energy in this trial
        trial_whiskEnergy = whiskEnergy[trial_whiskInds]
        
        # average energy in trial
        # will be nan if any nan values in trial
        avg_whiskEnergy[trialInd] = np.mean(trial_whiskEnergy)
        
    # return
    return avg_whiskEnergy



#%% COMPUTE AVERAGE RUNNING SPEED IN EACH TRIAL


def fcn_compute_avgRunSpeed_inTrials(session, t_beginAvg, t_endAvg):
    
    # unpack session
    
    nTrials = session['nTrials']
    tPts = session['time_stamp'].copy()
    runSpeed = session['walk_trace'].copy()
        
    
    # average run speed in each trial
    avg_runSpeed = np.zeros(nTrials)
          
    # loop over trials
    for trialInd in range(0,nTrials,1):
                 
        # indices of run data for this trial
        trial_runTrace_inds = np.nonzero( (tPts >= t_beginAvg[trialInd]) & \
                                          (tPts <= t_endAvg[trialInd]) )[0]
                        
        # run speed in this trial
        trial_runSpeed = runSpeed[trial_runTrace_inds]
               
        # average speed in trial
        # will be nan if any nan values in trial
        #avg_runSpeed[trialInd] = np.mean(trial_runSpeed)
        avg_runSpeed[trialInd] = np.mean(np.abs(trial_runSpeed))
                        
        
    # return
    session['avg_runSpeed'] = avg_runSpeed
    
    return session

#%%

def fcn_compute_minRunSpeed_inTrials(session, t_beginAvg, t_endAvg):
    
    # unpack session
    
    nTrials = session['nTrials']
    tPts = session['time_stamp'].copy()
    runSpeed = session['walk_trace'].copy()
        
    
    # average run speed in each trial
    min_runSpeed = np.zeros(nTrials)
          
    # loop over trials
    for trialInd in range(0,nTrials,1):
                 
        # indices of run data for this trial
        trial_runTrace_inds = np.nonzero( (tPts >= t_beginAvg[trialInd]) & \
                                          (tPts <= t_endAvg[trialInd]) )[0]
                        
        # run speed in this trial
        trial_runSpeed = runSpeed[trial_runTrace_inds]
               
        # average speed in trial
        # will be nan if any nan values in trial
        min_runSpeed[trialInd] = np.min(trial_runSpeed)
                        
        
    # return
    session['min_runSpeed'] = min_runSpeed
    
    return session



#%% DETERMINE IF TRIALS ARE RESTING OR RUNNING BASED ON AVG SPEED IN TRIAL

def fcn_determine_runningTrials(session, run_thresh):
    

    # average run speed in every trial 
    speed_measure = session['avg_runSpeed'].copy()
    
    # minimum run speed in every trial
    #speed_measure = session['min_runSpeed'].copy()
    
    # number of trials
    nTrials = len(speed_measure)
    
    # marks whether trial is running or not
    running = np.zeros(nTrials)*np.nan
    
    # get running trials
    runTrials = np.nonzero(speed_measure >= run_thresh)[0]
    running[runTrials] = True
    
    # get resting trials
    restTrials = np.nonzero(speed_measure < run_thresh)[0]
    running[restTrials] = False

    # everything else will be a nan

    
    # return
    session['running'] = running
    return session





#%% COMPUTE PUPIL STATISTIC USED TO CHARACTERIZE EACH TRIAL

def fcn_compute_pupilMeasure_eachTrial(session):
    
    # unpack session
    pupilSize_method = session['pupilSize_method']
    stim_onset = session['stim_on_time'].copy() 
    stim_duration = session['stim_duration']
    trial_start = session['trial_start'].copy()
    trial_end = session['trial_end'].copy()
    trial_window = session['trial_window'].copy()
    
    # check if we want to implement a pupil lag
    if 'pupil_lag' in session:
        
        pupil_lag = session['pupil_lag']
    
    else:
        
        pupil_lag = 0.
    
    # different options for pupil size method
    
    # avg across whole trial
    if pupilSize_method == 'avgSize_acrossTrial':
    
        t_beginAvg = trial_start.copy() + pupil_lag
        t_endAvg = trial_end.copy() + pupil_lag
        avg_pupilSize = fcn_compute_avgPupilSize_inTrials(session, t_beginAvg, t_endAvg) 
        
    # average size before stim onset        
    elif pupilSize_method == 'avgSize_beforeStim':
        
        t_beginAvg = stim_onset + trial_window[0] + pupil_lag
        t_endAvg = stim_onset + pupil_lag
        avg_pupilSize = fcn_compute_avgPupilSize_inTrials(session, t_beginAvg, t_endAvg)   
        
    # average size before and across stimulus duration
    elif pupilSize_method == 'avgSize_beforeStim_acrossDuration':
        
        t_beginAvg = stim_onset + trial_window[0] + pupil_lag
        t_endAvg = stim_onset + stim_duration + pupil_lag
        avg_pupilSize = fcn_compute_avgPupilSize_inTrials(session, t_beginAvg, t_endAvg)
        
    
    # unknown method        
    else:
        
        sys.exit('unidentified pupil size method provided')
        
    # save to session    
    session['trial_pupilMeasure'] = avg_pupilSize
    
    # return
    return session


#%% GET TRIALS IN A GIVEN PUPIL SIZE RANGE

'''
get all the trials with avg pupil sizes within some range

notes
    avg pupil size could be computed in different ways
    trials that have a nan value for their pupil size measure will not be included
###
'''
def fcn_getTrials_in_pupilRange(session, cutoff_low, cutoff_high):
    
    # unpack session
    trial_pupilMeasure = session['trial_pupilMeasure'].copy()
    
    # find trials that fall within specified range
    trials_in_pupilRange = np.nonzero( (trial_pupilMeasure >= cutoff_low) & \
                                       (trial_pupilMeasure < cutoff_high) )[0]
       
    return trials_in_pupilRange 



#%% GET % OF MAX PUPIL SIZE CORRESPONDING TO CERTAIN PERCENTILE BLOCKS

def fcn_pupilPercentile_to_pupilSize(session):

    # unpack session
    trial_pupilMeasure = session['trial_pupilMeasure'].copy()
    pupilBlock_size = session['pupilBlock_size']
    pupilBlock_step = session['pupilBlock_step']
    
    # number of blocks
    nBlocks = int(1/(pupilBlock_size))
    
    # initialize
    pupilSize_percentileBlocks = np.zeros((2, nBlocks))

    # loop over blocks and get range of pupil sizes in each one
    for blockInd in range(0, nBlocks):
        
        # low and high pct
        pct_low = int(blockInd*pupilBlock_step*100)
        pct_high = int(blockInd*pupilBlock_step*100 + pupilBlock_size*100)

        # cutoffs
        cutoff_low = np.nanpercentile(trial_pupilMeasure, pct_low)
        cutoff_high = np.nanpercentile(trial_pupilMeasure, pct_high)
        
        # pupil sizes for start/end of block
        pupilSize_percentileBlocks[0, blockInd] = cutoff_low
        pupilSize_percentileBlocks[1, blockInd] = cutoff_high
        
        
    return pupilSize_percentileBlocks
        

#%% GET TRIALS IN EACH PUPIL BLOCK

# given a pupil block size and pupil block step, loop over the corresponding 
# set of pupil blocks and get all trials in each one

def fcn_get_trials_in_pupilBlocks(session):

    # unpack session
    trial_pupilMeasure = session['trial_pupilMeasure'].copy()
    pupilBlock_size = session['pupilBlock_size']
    pupilBlock_step = session['pupilBlock_step']
    pupilSplit_method = session['pupilSplit_method']
    
    # number of blocks
    nBlocks = int(np.ceil((1 - pupilBlock_size)/(pupilBlock_step)) + 1)
    
    # trials in each block
    pupil_block_trials = np.zeros(nBlocks, dtype='object')

    # loop over blocks and get range of pupil sizes in each one
    for blockInd in range(0, nBlocks):
        
        # low and high pct
        pct_low = int(blockInd*pupilBlock_step*100)
        pct_high = int(blockInd*pupilBlock_step*100 + pupilBlock_size*100)

        # split pupil size distribution based on percentile ranges
        if pupilSplit_method == 'percentile':
        
            # cutoffs
            cutoff_low = np.nanpercentile(trial_pupilMeasure, pct_low)
            cutoff_high = np.nanpercentile(trial_pupilMeasure, pct_high)
                    
        # split pupil size distribution based on percentage of max value [normalized to 1]           
        elif pupilSplit_method == 'pct_of_max':
            
            cutoff_low = pct_low/100
            cutoff_high = pct_high/100
            
        # unknown splitting method    
        else:
            
            sys.exit('unspecified pupil splitting method')

        # get trials in each block
        trials_in_pupilRange = fcn_getTrials_in_pupilRange(session, cutoff_low, cutoff_high)
        
        # store the trials
        pupil_block_trials[blockInd] = trials_in_pupilRange
        
    # make sure we got to 100th percentile
    if pct_high != 100:       
        sys.exit('pupil block size and step chosen such that 100th%ile was not reached.')
        
    # save to session
    session['pupil_block_trials'] = pupil_block_trials
    
    # return
    return session

#%% NUMBER OF TRIALS OF EACH STIMULUS IN EACH PUPIL BLOCK

def fcn_trials_perFrequency_perPupilBlock(session):
    
    # number of pupil blocks
    nBlocks = len(session['pupil_block_trials'])
    
    # number of stim frequencys
    nFreqs = session['n_frequencies']
        
    # nTrials per pupil class
    trials_perFreq_perPupilBlock = np.zeros((nFreqs, nBlocks), dtype='object')
    nTrials_perFreq_perPupilBlock = np.zeros((nFreqs, nBlocks))
    
    # initialize min number of trials across all pupil classes and stimuli  
    min_nTrials = np.inf
        
    # loop over stimulus frequency
    for ind_freq in range(0, nFreqs):

        # loop over pupil blocks
        for ind_block in range(0,nBlocks):
            
            # trial indices in this pupil block
            trialInds_pupilBlock = session['pupil_block_trials'][ind_block].copy() 

            # trial indices for this stimulus
            trialInds_stim = session['trialInds_eachFrequency'][ind_freq].copy()
            
            trials_perFreq_perPupilBlock[ind_freq, ind_block] = np.intersect1d(trialInds_pupilBlock,  trialInds_stim)
            
            # intersection with pupil block
            nTrials_perFreq_perPupilBlock[ind_freq, ind_block] = \
                len(trials_perFreq_perPupilBlock[ind_freq, ind_block])
                    
            # update minimum number of trials
            min_nTrials = np.min([min_nTrials, nTrials_perFreq_perPupilBlock[ind_freq, ind_block]])
    
    # save to session
    session['trials_perFreq_perPupilBlock'] = trials_perFreq_perPupilBlock
    session['nTrials_perFreq_perPupilBlock'] = nTrials_perFreq_perPupilBlock
    session['max_nTrials'] = min_nTrials # maximum # of trials we can subsample from stimulus conditions
    
    # return
    return session


#%% GET TRIALS IN EACH RUNNING BLOCK

# given a run block size and run block step, loop over the corresponding 
# set of run blocks and get all trials in each one


def fcn_get_trials_in_runBlocks(session, trials):

    # unpack session
    trial_runSpeed = session['avg_runSpeed'][trials]
    runBlock_size = session['runBlock_size']
    runBlock_step = session['runBlock_step']
    runSplit_method = session['runSplit_method']
    
    # number of blocks
    nBlocks = int(np.ceil((1 - runBlock_size)/(runBlock_step)) + 1)
    
    # trials in each block
    run_block_trials = np.zeros(nBlocks, dtype='object')

    # loop over blocks and get range of run speeds in each one
    for blockInd in range(0, nBlocks):
        
        # low and high pct
        pct_low = int(blockInd*runBlock_step*100)
        pct_high = int(blockInd*runBlock_step*100 + runBlock_size*100)

        # split pupil size distribution based on percentile ranges
        if runSplit_method == 'percentile':
        
            # cutoffs
            cutoff_low = np.nanpercentile(trial_runSpeed, pct_low)
            cutoff_high = np.nanpercentile(trial_runSpeed, pct_high)
            
        # unknown splitting method    
        else:
            
            sys.exit('unspecified running splitting method')

        # get trials in each block
        trialInds_in_speedRange = np.nonzero( (trial_runSpeed >= cutoff_low) & (trial_runSpeed < cutoff_high) )[0]
        
        # store the trials
        run_block_trials[blockInd] = trials[trialInds_in_speedRange]
        
    # make sure we got to 100th percentile
    if pct_high != 100:       
        sys.exit('run block size and step chosen such that 100th%ile was not reached.')
        
    # save to session
    session['run_block_trials'] = run_block_trials
    
    # return
    return session




#%% NUMBER OF TRIALS OF EACH STIMULUS IN EACH RUN BLOCK

def fcn_trials_perFrequency_perRunBlock(session):
    
    # number of pupil blocks
    nBlocks = len(session['run_block_trials'])
    
    # number of stim frequencys
    nFreqs = session['n_frequencies']
        
    # nTrials per pupil class
    trials_perFreq_perRunBlock = np.zeros((nFreqs, nBlocks), dtype='object')
    nTrials_perFreq_perRunBlock = np.zeros((nFreqs, nBlocks))
    
    # initialize min number of trials across all pupil classes and stimuli  
    min_nTrials = np.inf
        
    # loop over stimulus frequency
    for ind_freq in range(0, nFreqs):

        # loop over pupil blocks
        for ind_block in range(0,nBlocks):
            
            # trial indices in this pupil block
            trialInds_runBlock = session['run_block_trials'][ind_block].copy() 

            # trial indices for this stimulus
            trialInds_stim = session['trialInds_eachFrequency'][ind_freq].copy()
            
            trials_perFreq_perRunBlock[ind_freq, ind_block] = np.intersect1d(trialInds_runBlock,  trialInds_stim)
            
            # intersection with run block
            nTrials_perFreq_perRunBlock[ind_freq, ind_block] = \
                len(trials_perFreq_perRunBlock[ind_freq, ind_block])
                    
            # update minimum number of trials
            min_nTrials = np.min([min_nTrials, nTrials_perFreq_perRunBlock[ind_freq, ind_block]])
    
    # save to session
    session['trials_perFreq_perRunBlock'] = trials_perFreq_perRunBlock
    session['nTrials_perFreq_perRunBlock'] = nTrials_perFreq_perRunBlock
    session['max_nTrials_runBlocks'] = min_nTrials # maximum # of trials we can subsample from stimulus conditions
    
    # return
    return session



#%% TRIALS OF EACH STIMULUS FOR RESTING AND RUNNING

def fcn_trials_perFrequency_perRestRun(session):
    
    # number of stim frequencys
    nFreqs = session['n_frequencies']
        
    # nTrials per frequency during resting and running trials
    trials_perFreq_rest = np.zeros((nFreqs), dtype='object')
    trials_perFreq_run = np.zeros((nFreqs), dtype='object')
    
    nTrials_perFreq_rest = np.zeros((nFreqs))
    nTrials_perFreq_run = np.zeros((nFreqs))
    
        
    # loop over stimulus frequency
    for ind_freq in range(0, nFreqs):

        # trial indices for running
        trialInds_rest = np.nonzero(session['running'] == False)[0]
        
        # trial indices for resting
        trialInds_run = np.nonzero(session['running'] == True)[0]

        # trial indices for this stimulus
        trialInds_stim = session['trialInds_eachFrequency'][ind_freq].copy()
        
        # intersection with resting & running trials
        trials_perFreq_rest[ind_freq] = np.intersect1d(trialInds_stim,  trialInds_rest)
        trials_perFreq_run[ind_freq] = np.intersect1d(trialInds_stim,  trialInds_run)
            
        nTrials_perFreq_rest[ind_freq] = len(trials_perFreq_rest[ind_freq])
        nTrials_perFreq_run[ind_freq] = len(trials_perFreq_run[ind_freq])
         
    # minimum number of trials across all conditions       
    min_nTrials_rest = np.min(nTrials_perFreq_rest)
    min_nTrials_run = np.min(nTrials_perFreq_run)
    max_nTrials = int(np.min([min_nTrials_rest, min_nTrials_run]))            

    
    # save to session
    session['trials_perFreq_rest'] = trials_perFreq_rest
    session['trials_perFreq_run'] = trials_perFreq_run
    session['nTrials_perFreq_rest'] = nTrials_perFreq_rest
    session['nTrials_perFreq_run'] = nTrials_perFreq_run
    session['max_nTrials'] = max_nTrials # maximum # of trials we can subsample from stimulus conditions
    
    # return
    return session

#%% spike times of each cell in every trial


def fcn_spikeTimes_trials_cells_spont(session):
    
    # unpack session
    nTrials = session['nTrials']
    nCells = session['nCells']
    cell_spk_times = session['cell_spk_times'].copy()
    trial_start = session['trial_start'].copy()
    trial_end = session['trial_end'].copy()
    
    # for each trial, compute spike times of each cell relative to stim onset
    spikeTimes_trials_cells = np.zeros((nTrials, nCells), dtype='object')
    
    
    for ind_trial in range(0,nTrials):
        
        for ind_cell in range(0,nCells):
            
            spikeInds_trial_cell = np.nonzero( (cell_spk_times[ind_cell] >= trial_start[ind_trial]) & \
                                               (cell_spk_times[ind_cell] <= trial_end[ind_trial]) )[0]
            
            
            spikeTimes_trials_cells[ind_trial, ind_cell] = cell_spk_times[ind_cell][spikeInds_trial_cell] 
                

            
    # return session
    session['spikeTimes_trials_cells'] = spikeTimes_trials_cells
    
    return session
    

#%% COMPUTE SPIKE TIMES OF EACH CELL IN ALL TRIALS; ALIGN TO STIM ONSET

def fcn_spikeTimes_trials_cells(session):
    
    # unpack session
    nTrials = session['nTrials']
    nCells = session['nCells']
    cell_spk_times = session['cell_spk_times'].copy()
    stim_onset = session['stim_on_time'].copy()
    trial_start = session['trial_start'].copy()
    trial_end = session['trial_end'].copy()
    
    # for each trial, compute spike times of each cell relative to stim onset
    spikeTimes_trials_cells = np.zeros((nTrials, nCells), dtype='object')
    
    
    for ind_trial in range(0,nTrials):
        
        for ind_cell in range(0,nCells):
            
            spikeInds_trial_cell = np.nonzero( (cell_spk_times[ind_cell] >= trial_start[ind_trial]) & \
                                               (cell_spk_times[ind_cell] <= trial_end[ind_trial]) )[0]
                
            
            spikeTimes_trials_cells[ind_trial, ind_cell] = \
                cell_spk_times[ind_cell][spikeInds_trial_cell] - stim_onset[ind_trial]
                    
            
            
    # return session
    session['spikeTimes_trials_cells'] = spikeTimes_trials_cells
    
    return session


#%% COMPUTE SPIKE COUNTS IN EACH TRIAL
#
#   OUPUTS:
#       spkCounts_trials_cells --   array of shape (nTrials, nCells) 
#                                   containing spike counts of each cell in 
#                                   each trial
    
def fcn_compute_spikeCnts_inTrials(session):
    
    # spike times of each cell in every trial
    spikeTimes_trials_cells = session['spikeTimes_trials_cells'].copy()
    
    # number of cells
    nCells = np.shape(spikeTimes_trials_cells)[1]
    
    # number of trials
    nTrials = np.shape(spikeTimes_trials_cells)[0]
    
    # spike counts of each cell in each trial
    spkCounts_trials_cells = np.zeros((nTrials, nCells))
    
    # loop over trials
    for trialInd in range(0,nTrials,1):
        
        # loop over cells
        for cellInd in range(0,nCells,1):
            
            # count spikes
            spkCounts_trials_cells[trialInd, cellInd] = np.size(spikeTimes_trials_cells[trialInd, cellInd])
            
    
    # return
    session['spkCounts_trials_cells'] = spkCounts_trials_cells
    
    return session


#%% COMPUTE SPIKE COUNT RATE IN EACH TRIAL
#
#   OUPUTS:
#       spkRate_trials_cells --     array of shape (nTrials, nCells) 
#                                   containing spike rates of each cell in 
#                                   each trial
    
def fcn_compute_spikeRate_inTrials(session, trial_length):
    
    # spike counts in trials
    spkCounts_trials_cells = session['spkCounts_trials_cells'].copy()
    
    # divide by trial length
    spkRate_trials_cells = spkCounts_trials_cells/trial_length
    
    # return
    session['spkRate_trials_cells'] = spkRate_trials_cells
    
    return session




#%% APPLY GAUSSIAN SMOOTHING TO AN ARBITRARY SIGNAL


def fcn_gaussian_smooth(t_raw, timeSeries_raw, tStart, tEnd, tStep, sigma):
    
    tPts = np.arange(tStart, tEnd, tStep)

    timeSeries_smooth = np.zeros(len(tPts))


    # loop over evaluation time points
    for tInd in range(0,len(tPts)):
        
        # get evaluation time point
        tEval = tPts[tInd]
                        
        # convolve time series with gaussian filter
        gauss_weight = np.exp(-((t_raw-tEval)**2)/(2*sigma**2))
        gauss_weight_norm = gauss_weight/np.sum(gauss_weight)
                
        conv_signal = gauss_weight_norm*timeSeries_raw
   
        # weighted average at this time point is sum of contributions from all data points
        timeSeries_smooth[tInd] = np.sum(conv_signal)
        
        print(tInd)


    return tPts, timeSeries_smooth 
        


#%% COMPUTE SMOOTHED PSTH (gaussian) OF A SINGLE CELL IN A GIVEN SET OF TRIALS

# INPUTS
# --- spikeTimes_aligned:   object array of shape (nTrials,)
#                           the ith element gives spike times of cell on the
#                           ith trial
# --- tStart:               first time point at which to compute psth
# --- tEnd:                 last time point at which to compute psth
# --- tStep:                step size of computation time points for psth
# --- sigma:                width of gaussian kernal
# OUTPUTS
# --- tPts:                 time points at which psth was evaluated
# --- psth:                 array of shape (nTrials, len(tPts)) giving the
#                           value of the psth in each trial at each time point

def fcn_compute_singleTrial_psth_gaussian(spikeTimes_aligned, tStart, tEnd, tStep, sigma):
    
    tPts = np.arange(tStart, tEnd, tStep)
    
    nTrials = np.size(spikeTimes_aligned)
    
    psth = np.zeros((nTrials, len(tPts)))
    
    # loop over trials
    for trialInd in range(0,nTrials):
        
        # spike times of a given cell in this trial
        spkTimes_trial = spikeTimes_aligned[trialInd]
        
        # loop over all spike times and compute rate
        for ts_ind in range(0,np.size(spkTimes_trial)):
            
            # guassian filter
            ts = spkTimes_trial[ts_ind]
            conv_spike = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-((tPts-ts)**2)/(2*sigma**2))

            # add to running sum
            psth[trialInd, :] += conv_spike
    
    # return psth
    return tPts, psth


#%% COMPUTE SPIKE COUNTS IN A WINDOW RELATIVE TO STIMULUS ONSET

# INPUTS
# --- spikeTimes_aligned:   object array of shape (nTrials,)
#                           the ith element gives spike times of cell on the
#                           ith trial, aligned to stimulus onset
# --- tStart:               beginning of spike count window
# --- tEnd:                 end of spike count window


def fcn_compute_spkCounts_inWindow(spikeTimes_aligned, tStart, tEnd):
    

    nTrials = np.size(spikeTimes_aligned)
     
    spkCounts_window = np.zeros((nTrials))
     
    # loop over trials
    for trialInd in range(0,nTrials):
        
        # spike times of a given cell in this trial
        spkTimes_trial = spikeTimes_aligned[trialInd]
        
        
        # get spike times in analysis window
        spikeInds_window = np.nonzero( (spkTimes_trial <= tEnd) & (spkTimes_trial >= tStart) )[0]
        
        
        # spike counts
        spkCounts_window[trialInd] = np.size(spikeInds_window)
        
    # return
    return spkCounts_window



#%% COMPUTE SPIKE TIMES IN A WINDOW RELATIVE TO STIMULUS ONSET

# INPUTS
# --- spikeTimes_aligned:   object array of shape (nTrials,)
#                           the ith element gives spike times of cell on the
#                           ith trial, aligned to stimulus onset
# --- tStart:               beginning of spike count window
# --- tEnd:                 end of spike count window


def fcn_compute_spkTimes_inWindow(spikeTimes_aligned, tStart, tEnd):
    

    nTrials = np.size(spikeTimes_aligned)
     
    spkTimes_window = np.zeros((nTrials), dtype='object')
     
    # loop over trials
    for trialInd in range(0,nTrials):
        
        # spike times of a given cell in this trial
        spkTimes_trial = spikeTimes_aligned[trialInd]
        
        # get spike times in analysis window
        spikeInds_window = np.nonzero( (spkTimes_trial <= tEnd) & (spkTimes_trial >= tStart) )[0]
        
        # spike counts
        spkTimes_window[trialInd] = spkTimes_trial[spikeInds_window].copy()
        
    # return
    return spkTimes_window
     


#%% compute spike counts in a sliding window for all trials and cells in a session

def fcn_compute_windowed_spikeCounts(session, wind_length, wind_step):
    
    # unpach session
    nTrials = session['nTrials']
    trial_window = session['trial_window'].copy()
    spikeTimes_trials_cells = session['spikeTimes_trials_cells'].copy()
    
    # number of cells
    nCells = np.size(spikeTimes_trials_cells, 1)
    
    # number of time windows
    nWindows = int((trial_window[1] - trial_window[0] - wind_length)/wind_step)
   
    # save array with number of spikes of each cell in each window of each trial
    nSpikes = np.zeros((nTrials, nCells, nWindows))
    
    # window times = time of window end relative to trial start time
    t_windows = np.zeros(nWindows)

    # loop over trials 
    for trialInd in range(0,nTrials,1):
                        
        # loop over cell ids and count spikes in each window
        for cellInd in range(0,nCells,1):
            
            tSpikes_cell = spikeTimes_trials_cells[trialInd, cellInd]
            
            # loop over windows
            for windInd in range(0,nWindows,1):
                
                # begining and end of windows
                windBegin = trial_window[0] + windInd*wind_step 
                windEnd = windBegin + wind_length
                
                # times of spikes
                tInd = np.where((tSpikes_cell >= windBegin) & (tSpikes_cell < windEnd))[0]
                
                # number of spikes                       
                nSpikes[trialInd, cellInd, windInd] = np.size(tInd)
                
                # time point = end of window
                t_windows[windInd] = windEnd
                
    # return
    return t_windows, nSpikes


#%% compute time-varying spike count for a trial

def fcn_compute_timeVarying_spkCount(spike_train, tStart, tEnd, binSize, dt):
    
    # number of time windows
    window_length = tEnd-tStart
    
    bin_lower = np.arange(0, window_length - binSize + dt, dt)
    bin_upper = np.arange(binSize, window_length + dt, dt) 
    
    nBins = len(bin_lower)
    
    # number of spikes in each window
    nSpikes = np.zeros((nBins), dtype='int')
    
    # loop over windows
    for binInd in range(0,nBins,1):

        # times of spikes
        tInd = np.nonzero((spike_train >= bin_lower[binInd]) & (spike_train <= bin_upper[binInd]))[0]
        
        # number of spikes                       
        nSpikes[binInd] = np.size(tInd)

    
    return nSpikes, bin_upper


#%% COMPUTE INTERSPIKE INTERVAL FOR EACH CELL

def fcn_compute_interspike_interval(spikeTimes_trials):
    
    # number of cells
    nTrials = np.size(spikeTimes_trials) 
    
    # initialize
    isi = np.array([])
    
    # loop over cells and trials
    for indTrial in range(0, nTrials):
        
        isi_thisTrial = np.diff(spikeTimes_trials[indTrial])
        
        isi = np.append(isi, isi_thisTrial)
    

    return isi


#%% power spectrum for spike train data

# data = nTrials x nTpts (binned spike counts)

def mt_specpb_lp(data, Fs=1000, NW=4, trial_ave=True, type = 0):
    
    nTrials = np.size(data, 0)
    nTpts = np.size(data,1)
    dt = 1/Fs
    
    tapers, _ = dpss_windows(nTpts, NW, 2*NW-1)             # Compute the tapers,
    tapers *= np.sqrt(Fs)                                   # ... and scale them.
    
    nTapers = np.size(tapers, 0)

    dataT = np.ones((nTrials, nTapers, nTpts ))*np.nan
    
    for indTrial in range(0, nTrials):
        
        trial_data = data[indTrial, :].copy()
        
        for indTaper in range(0, nTapers):
            
            dataT[indTrial, indTaper, :] = trial_data*tapers[indTaper,:]
            
            
    T = scipy.fft.rfft(tapers)                                 # Compute the fft of the tapers.
    J = scipy.fft.rfft(dataT)                                  # Compute the fft of the tapered data.

    for indTrial in range(0, nTrials):
        
        if type == 0:
            J[indTrial, :, :] = J[indTrial, :, :] - T*np.mean(data[indTrial, :])

        else:
            J[indTrial, :, :] = J[indTrial, :, :] - T*np.mean(data)
    
    
    J *= np.conj(J)                                        # Compute the spectrum
    S = np.real(np.mean(J,1))

    # normalize by rate in this trial
    Snorm = np.ones(np.shape(S))*np.nan
    
    for indTrial in range(0, nTrials):
        if type == 0:
            Snorm[indTrial, :] = S[indTrial, :]/(np.sum(data[indTrial,:])/(dt*nTpts))
        else:
            Snorm[indTrial, :] = S[indTrial, :]/(np.sum(data)/(dt*nTpts*nTrials))

    f = scipy.fft.rfftfreq(nTpts, 1 / Fs)
    
    if trial_ave: 
        S = np.mean(S,0)                        # Average across trials.
        Snorm = np.mean(Snorm, 0)
        
    return f, S, Snorm



# data = nTrials x nTpts (binned spike counts)

def raw_specpb_lp(data, Fs=1000, trial_ave=True, type = 0):
    
    nTrials = np.size(data, 0)
    nTpts = np.size(data,1)
    dt = 1/Fs
    
    window_func = np.ones(nTpts)
    window_func = window_func*1/np.sqrt(dt*nTpts)
    
    dataT = np.ones((nTrials, nTpts ))*np.nan
    
    for indTrial in range(0, nTrials):
        
        trial_data = data[indTrial, :].copy()
        
        dataT[indTrial, :] = trial_data*window_func
    
    J = scipy.fft.rfft(dataT)
    R = scipy.fft.rfft(window_func)

    for indTrial in range(0, nTrials):
        
        if type == 0:
            J[indTrial, :] = J[indTrial, :] - R*np.mean(data[indTrial, :])
        else:
            J[indTrial, :] = J[indTrial, :] - R*np.mean(data)
    
    J *= np.conj(J)                                        # Compute the spectrum
    S = np.real(J)

    # normalize by rate in this trial
    Snorm = np.ones(np.shape(S))*np.nan
    
    for indTrial in range(0, nTrials):
        if type == 0:
            Snorm[indTrial, :] = S[indTrial, :]/(np.sum(data[indTrial,:])/(dt*nTpts))
        else:
            Snorm[indTrial, :] = S[indTrial, :]/(np.sum(data)/(dt*nTpts*nTrials))

    f = scipy.fft.rfftfreq(nTpts, 1 / Fs)
    
    if trial_ave: 
        S = np.mean(S,0)                        # Average across trials.
        Snorm = np.mean(Snorm, 0)
        
    return f, S, Snorm



#%% COMPUTE SPIKE COUNT CORRELATIONS

#   INPUTS:
#       spkCounts_trials_cells --   output of fcn_compute_spikeCnts_inTrials
#                                   (nTrials, nCells)
#                                   [spike counts of each cell in every trial]
#
#   OUPUTS:
#       r_sc --                     array of shape (nCells, nCells) where each
#                                   element is the spike count correlation  
#                                   between the corresponding neurons

def fcn_compute_spikeCount_corr(spkCounts_trials_cells):
        
    # spike count correlation
    r_sc = np.corrcoef(spkCounts_trials_cells, rowvar=False)         
    
    # return
    return r_sc



#%% COMPUTE PARTIAL SPIKE COUNT CORRELATIONS

#   INPUTS:
#       spkCounts_trials_cells --   output of fcn_compute_spikeCnts_inTrials
#                                   (nTrials, nCells)
#                                   [spike counts of each cell in every trial]
#       third_variable --           third variable that we want to control for
#                                   (nTrials)
#
#   OUPUTS:
#       par_r_sc --                     array of shape (nCells, nCells) where each
#                                   element is the spike count correlation  
#                                   between the corresponding neurons

def fcn_compute_spikeCount_parcorr(spkCounts_trials_cells, third_variable):
        
    # number of cells
    nCells = np.shape(spkCounts_trials_cells)[1]
    
    # partial corr
    par_r_sc = np.ones((nCells,nCells))*np.nan
    
    for i in range(0,nCells):
        for j in range(0, nCells):
            

            r_sc = np.corrcoef(spkCounts_trials_cells[:,i], spkCounts_trials_cells[:,j])[0,1]
            rxz = np.corrcoef(spkCounts_trials_cells[:,i], third_variable)[0,1]
            ryz = np.corrcoef(spkCounts_trials_cells[:,j], third_variable)[0,1]
            
            par_r_sc[i,j] = (r_sc - rxz*ryz)/( ((1-rxz**2) * (1-ryz**2))**(1/2) )
    
    
    # return
    return par_r_sc


#%% COMPUTE SPIKE COUNT COVARIANCE
#
#   INPUTS:
#       spkCounts_trials_cells --   output of fcn_compute_spikeCnts_inTrials
#                                   (nTrials, nCells)
#                                   [spike counts of each cell in every trial]
#
#   OUPUTS:
#       cov_sc --                   array of shape (nCells, nCells) where each
#                                   element is the spike count correlation  
#                                   between the corresponding neurons
    
def fcn_compute_spikeCount_cov(spkCounts_trials_cells):
    
    # spike count correlation
    cov_sc = np.cov(spkCounts_trials_cells, rowvar=False)
            
    # return
    return cov_sc


#%% COMPUTE DIMENSIONALITY
#
#   INPUTS:
#       cov_sc --                   output of fcn_compute_spikeCount_cov
#                                   [pairwise spike counts covariance matrix]
#
#   OUPUTS:
#       dimensionality --           a single number corresponding to the
#                                   dimensionality of the neural data

def fcn_compute_dimensionality(cov_sc):

    tr_cov = np.trace(cov_sc)
    tr_cov_sq = np.trace(np.matmul(cov_sc,cov_sc))
    dimensionality = (tr_cov**2)/tr_cov_sq 
    
    # return
    return dimensionality


#%% COMPUTE FANO FACTOR

'''
inputs
    skCounts:           (nTrials,) or (nTrials, nVar1, nVar2,...)
                        trials should always be first dimension
outputs
    fano_factor:        scalar or (nVar1, nVar2, ...)
'''

def fcn_fanoFactor(spkCounts):
    
    fano_factor = np.var(spkCounts,0)/np.mean(spkCounts,0)
    
    return fano_factor


#%% COMPUTE TRIAL SHUFFLED DATA

def fcn_trial_shuffled_spikeCounts(spkCounts_trials_cells, rand_seed = None):
    
    if rand_seed == None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(rand_seed)
    
    nTrials = np.size(spkCounts_trials_cells, 0)
    nCells = np.size(spkCounts_trials_cells, 1)
    
    spkCounts_trials_cells_shuffle = np.ones((nTrials, nCells))*np.nan
    
    for indCell in range(0, nCells):
        
        trialInds_shuffle = rng.permutation(nTrials)
        
        spkCounts_trials_cells_shuffle[:, indCell] = spkCounts_trials_cells[trialInds_shuffle, indCell]
    
    return spkCounts_trials_cells_shuffle, trialInds_shuffle



#%% MULTICLASS DECODING

def fcn_decoding_multiclass(session, trialInds_allClasses, \
                            wind_length, wind_step, \
                            classifier, compute_shuffleDist,\
                            nShuffles, shuffle_percentile, \
                            crossVal_type, lda_solver, nFolds=5, nReps=1):
    
    
    # number of classes
    nClasses = len(trialInds_allClasses)
    
    # compute spike times of each cell in each window of each trial   
    t_windows, nSpikes = fcn_compute_windowed_spikeCounts(session, wind_length, wind_step)
    
    # number of time windows        
    nWindows = np.size(t_windows)
    
    # initialize decoding quantities     
    accuracy = np.zeros(nWindows)
    mean_accuracy_shuffle = np.zeros(nWindows)
    sd_accuracy_shuffle = np.zeros(nWindows)
    lowPercentile_accuracy_shuffle = np.zeros(nWindows)
    highPercentile_accuracy_shuffle = np.zeros(nWindows)
    p_accuracy =  np.zeros(nWindows)
    confusion_mat = np.zeros((nClasses, nClasses, nWindows))
    
    # loop across time windows
    for indWind in range(0, nWindows):

        # organize data for decoding
        
        # 0th class
        X = nSpikes[trialInds_allClasses[0],:,indWind].copy()
        classLabels = np.zeros(len(trialInds_allClasses[0]))
        
        # 1st to nth class
        for classInd in range(1, nClasses):
            X = np.vstack((X, nSpikes[trialInds_allClasses[classInd],:,indWind]))
            classLabels = np.hstack((classLabels, classInd*np.ones(len(trialInds_allClasses[classInd]))))
            
        # run decoding
        if crossVal_type == 'stratified_kFold':
            accuracy[indWind], \
            mean_accuracy_shuffle[indWind], sd_accuracy_shuffle[indWind], \
            lowPercentile_accuracy_shuffle[indWind], \
            highPercentile_accuracy_shuffle[indWind], \
            p_accuracy[indWind], confusion_mat[:,:,indWind] = \
            fcn_stratified_kFold_crossVal(X, classLabels, classifier, nFolds, \
                                         lda_solver, compute_shuffleDist, nShuffles, \
                                         shuffle_percentile)
                
                
        elif crossVal_type == 'repeated_stratified_kFold':
            accuracy[indWind], \
            mean_accuracy_shuffle[indWind], sd_accuracy_shuffle[indWind], \
            lowPercentile_accuracy_shuffle[indWind], \
            highPercentile_accuracy_shuffle[indWind], \
            p_accuracy[indWind], confusion_mat[:,:,indWind] = \
                fcn_repeated_stratified_kFold_crossVal(X, classLabels, classifier, \
                                                      nFolds, nReps, \
                                                      compute_shuffleDist, nShuffles, \
                                                      shuffle_percentile, lda_solver)
        
        # incompatible cross validation type
        else:
            sys.exit('unrecognized cross validation type entered.')
            
        
        print(indWind)
    
    
    # return
    return t_windows, accuracy, \
           mean_accuracy_shuffle, sd_accuracy_shuffle, \
           lowPercentile_accuracy_shuffle, highPercentile_accuracy_shuffle, \
           p_accuracy, confusion_mat
      

#%%
def fcn_decoding_multiclass_baseSubtract(session, trialInds_allClasses, \
                            wind_length, wind_step, \
                            classifier, compute_shuffleDist,\
                            nShuffles, shuffle_percentile, \
                            crossVal_type, lda_solver, nFolds=5, nReps=1):
    
    
    # number of classes
    nClasses = len(trialInds_allClasses)
    
    # compute spike times of each cell in each window of each trial   
    t_windows, nSpikes = fcn_compute_windowed_spikeCounts(session, wind_length, wind_step)

    
    for indWindow in range(0, len(t_windows)):
        nSpikes[:,:,indWindow] = nSpikes[:,:,indWindow] - nSpikes[:,:,0]

    

    # number of time windows        
    nWindows = np.size(t_windows)
    
    # initialize decoding quantities     
    accuracy = np.zeros(nWindows)
    mean_accuracy_shuffle = np.zeros(nWindows)
    sd_accuracy_shuffle = np.zeros(nWindows)
    lowPercentile_accuracy_shuffle = np.zeros(nWindows)
    highPercentile_accuracy_shuffle = np.zeros(nWindows)
    p_accuracy =  np.zeros(nWindows)
    confusion_mat = np.zeros((nClasses, nClasses, nWindows))
    
    # loop across time windows
    for indWind in range(1, nWindows):

        # organize data for decoding
        
        # 0th class
        X = nSpikes[trialInds_allClasses[0],:,indWind].copy()
        classLabels = np.zeros(len(trialInds_allClasses[0]))
        
        # 1st to nth class
        for classInd in range(1, nClasses):
            X = np.vstack((X, nSpikes[trialInds_allClasses[classInd],:,indWind]))
            classLabels = np.hstack((classLabels, classInd*np.ones(len(trialInds_allClasses[classInd]))))

            
        # run decoding
        if crossVal_type == 'stratified_kFold':
            accuracy[indWind], \
            mean_accuracy_shuffle[indWind], sd_accuracy_shuffle[indWind], \
            lowPercentile_accuracy_shuffle[indWind], \
            highPercentile_accuracy_shuffle[indWind], \
            p_accuracy[indWind], confusion_mat[:,:,indWind] = \
            fcn_stratified_kFold_crossVal(X, classLabels, classifier, nFolds, \
                                         lda_solver, compute_shuffleDist, nShuffles, \
                                         shuffle_percentile)
                
                
        elif crossVal_type == 'repeated_stratified_kFold':
            accuracy[indWind], \
            mean_accuracy_shuffle[indWind], sd_accuracy_shuffle[indWind], \
            lowPercentile_accuracy_shuffle[indWind], \
            highPercentile_accuracy_shuffle[indWind], \
            p_accuracy[indWind], confusion_mat[:,:,indWind] = \
                fcn_repeated_stratified_kFold_crossVal(X, classLabels, classifier, \
                                                      nFolds, nReps, \
                                                      compute_shuffleDist, nShuffles, \
                                                      shuffle_percentile, lda_solver)
        
        # incompatible cross validation type
        else:
            sys.exit('unrecognized cross validation type entered.')
            
        
        print(indWind)
    
    
    # return
    return t_windows, accuracy, \
           mean_accuracy_shuffle, sd_accuracy_shuffle, \
           lowPercentile_accuracy_shuffle, highPercentile_accuracy_shuffle, \
           p_accuracy, confusion_mat
      


#%% function to randomize trials across neurons for a given stimulus label

'''
inputs
    spikeCounts:     (nTrials, nCells) array of spike counts for a given stimulus condition
    
outputs
    spikeCounts_trialRandomized
'''

def fcn_randomize_spikeCounts(spikeCounts):
    
    # number of cells
    nCells = np.shape(spikeCounts)[1]
    
    # number of trials
    nTrials = np.shape(spikeCounts)[0]
    
    # initialize randomized spike counts
    spikeCounts_trialRandomized = np.zeros((nTrials, nCells))
    
    # loop over cells and randomize trial order
    for cellInd in range(0, nCells):
        
        permutedInds = np.random.permutation(nTrials)
        spikeCounts_trialRandomized[:,cellInd] = spikeCounts[permutedInds, cellInd]
    
    
    return spikeCounts_trialRandomized
    
    
    
#%% MULTICLASS DECODING: SHUFFLE TRIALS ACROSS NEURONS TO DESTROY NOISE CORRELATIONS

def fcn_decoding_multiclass_randomizeTrials(session, trialInds_allClasses, \
                                            wind_length, wind_step, \
                                            classifier, compute_shuffleDist,\
                                            nShuffles, shuffle_percentile, \
                                            crossVal_type, nFolds=5, nReps=1):
    
    sys.exit('this function does not currently work.')

    # number of classes
    nClasses = len(trialInds_allClasses)
    
    # compute spike times of each cell in each window of each trial   
    t_windows, nSpikes = fcn_compute_windowed_spikeCounts(session, wind_length, wind_step)
        
    # number of time windows        
    nWindows = np.size(t_windows)
    
    # number of cells
    nCells = np.shape(nSpikes)[1]
    
    # total number of decoding trials
    n_decodingTrials = 0
    for classInd in range(0, nClasses):      
        n_decodingTrials += np.size(trialInds_allClasses[classInd])       
    
    # randomized spikes
    nSpikes_randomized = np.zeros((n_decodingTrials, nCells, nWindows))
    
    # class lables
    classLabels = np.zeros(n_decodingTrials)
    
    # trial inds 
    rand_trialInds_allClasses = np.zeros(nClasses, dtype='object')
    
    
    
    # loop over classes and randomize trials
    
    startInd = 0

    for classInd in range(0, nClasses):
                
        trials_thisStim = trialInds_allClasses[classInd].copy()
        n_trials_thisStim = np.size(trials_thisStim)
        
        endInd = startInd + n_trials_thisStim
        
        rand_trialInds_allClasses[classInd] = np.arange(startInd, endInd, 1)
        classLabels[startInd:endInd] = classInd
        
        # loop over cells
        for cellInd in range(0, nCells):
            
            # permute trial order for this cell
            trialInds_rand = np.random.permutation(trials_thisStim)
            
            # get spikes in each window
            for windInd in range(0, nWindows):
            
                nSpikes_randomized[startInd:endInd, cellInd, windInd] = nSpikes[trialInds_rand, cellInd, windInd]
                
        # update starting index
        startInd = endInd


    # initialize decoding quantities     
    accuracy = np.zeros(nWindows)
    mean_accuracy_shuffle = np.zeros(nWindows)
    sd_accuracy_shuffle = np.zeros(nWindows)
    lowPercentile_accuracy_shuffle = np.zeros(nWindows)
    highPercentile_accuracy_shuffle = np.zeros(nWindows)
    p_accuracy =  np.zeros(nWindows)
    confusion_mat = np.zeros((nClasses, nClasses, nWindows))
    
    
    # loop across time windows
    for indWind in range(0, nWindows):

        # spike counts in this window
        X = nSpikes_randomized[:,:,indWind].copy()
            
        # run decoding
        if crossVal_type == 'stratified_kFold':
            accuracy[indWind], \
            mean_accuracy_shuffle[indWind], sd_accuracy_shuffle[indWind], \
            lowPercentile_accuracy_shuffle[indWind], \
            highPercentile_accuracy_shuffle[indWind], \
            p_accuracy[indWind], confusion_mat[:,:,indWind] = \
            fcn_stratified_kFold_crossVal(X, classLabels, classifier, nFolds, \
                                         compute_shuffleDist, nShuffles, \
                                         shuffle_percentile)
                
                
        elif crossVal_type == 'repeated_stratified_kFold':
            accuracy[indWind], \
            mean_accuracy_shuffle[indWind], sd_accuracy_shuffle[indWind], \
            lowPercentile_accuracy_shuffle[indWind], \
            highPercentile_accuracy_shuffle[indWind], \
            p_accuracy[indWind], confusion_mat[:,:,indWind] = \
                fcn_repeated_stratified_kFold_crossVal(X, classLabels, classifier, \
                                                      nFolds, nReps, \
                                                      compute_shuffleDist, nShuffles, \
                                                      shuffle_percentile)
        
        # incompatible cross validation type
        else:
            sys.exit('unrecognized cross validation type entered.')
            
        
        print(indWind)    
        
    # return
    return t_windows, accuracy, \
           mean_accuracy_shuffle, sd_accuracy_shuffle, \
           lowPercentile_accuracy_shuffle, highPercentile_accuracy_shuffle, \
           p_accuracy, confusion_mat
           







           