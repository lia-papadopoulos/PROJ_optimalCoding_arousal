'''
psth vs pupil size
'''

#%% IMPORTS

# basic imports
import sys        
import numpy as np
import numpy.matlib
import argparse
from scipy.io import savemat

# import settings file
import psth_vs_pupil_settings as settings

# paths to functions
sys.path.append(settings.func_path1)        
sys.path.append(settings.func_path2)

# functions
from fcn_processedh5data_to_dict import fcn_processedh5data_to_dict
from fcn_statistics import fcn_MannWhitney_twoSided
from fcn_SuData import fcn_makeTrials
from fcn_SuData import fcn_spikeTimes_trials_cells
from fcn_SuData import fcn_compute_windowed_spikeCounts
from fcn_SuData import fcn_trialInfo_eachFrequency
from fcn_SuData import fcn_compute_pupilMeasure_eachTrial
from fcn_SuData import fcn_get_trials_in_pupilBlocks
from fcn_SuData import fcn_trials_perFrequency_perPupilBlock
from fcn_SuData import fcn_pupilPercentile_to_pupilSize
from fcn_SuData import fcn_compute_minRunSpeed_inTrials
from fcn_SuData import fcn_determine_runningTrials
from fcn_SuData import fcn_trials_perFrequency_perRunBlock


#%% PARAMETERS

data_path = settings.data_path
outpath = settings.outpath
trial_window = settings.trial_window
baseline_window = settings.baseline_window
stimulus_window = settings.stimulus_window
window_length = settings.window_length
window_step = settings.window_step
stim_duration = settings.stim_duration
pupilBlock_size = settings.pupilBlock_size
pupilBlock_step = settings.pupilBlock_step
pupilSplit_method = settings.pupilSplit_method
nTrials_thresh = settings.nTrials_thresh
pupilSize_method = settings.pupilSize_method
restOnly = settings.restOnly
trialMatch = settings.trialMatch
runThresh = settings.runThresh
runSpeed_method = settings.runSpeed_method

#%% CHECKS

if ( (restOnly == True) and (trialMatch == True) ):
    sys.exit('cant do rest only w/ trial matching yet; need to add multiple subsamples')


#%% USER INPUTS

# argparser
parser = argparse.ArgumentParser() 

# session name
parser.add_argument('-session_name', '--session_name', type=str, default = '')
    
# arguments of parser
args = parser.parse_args()

# argparse inputs
session_name = args.session_name



#%% SESSION INFO

session_info = fcn_processedh5data_to_dict(session_name, data_path)


#%% UPDATE SESSION INFO

session_info['trial_window'] = trial_window

session_info['pupilBlock_size'] = pupilBlock_size
session_info['pupilBlock_step'] = pupilBlock_step
session_info['pupilSplit_method'] = pupilSplit_method
session_info['pupilSize_method'] = pupilSize_method

session_info['runSpeed_method'] = runSpeed_method

print('session updated')


#%% make trials
session_info = fcn_makeTrials(session_info)


#%% trials of each frequency
session_info = fcn_trialInfo_eachFrequency(session_info)


#%% compute spike times of each cell in every trial [aligned to stimulus onset]
session_info = fcn_spikeTimes_trials_cells(session_info)


#%% compute psth of each cell in every trial
t_windows, nSpikes = fcn_compute_windowed_spikeCounts(session_info, window_length, window_step)


#%% compute time-varying rate of each cell in every trial
psth_allTrials = nSpikes/window_length


#%% compute pupil measure in each trial
session_info = fcn_compute_pupilMeasure_eachTrial(session_info)

#%% running info 

# compute run speed in each trial
session_info = fcn_compute_minRunSpeed_inTrials(session_info, session_info['trial_start'], session_info['trial_end'])

# classify trials as running or resting
session_info = fcn_determine_runningTrials(session_info, runThresh)

# rest and run trials
run_trials = np.nonzero(session_info['running'])[0]
rest_trials = np.nonzero(session_info['running']==0)[0]

# pupil size of running trials
pupilSize_runTrials = session_info['trial_pupilMeasure'].copy()
pupilSize_runTrials[rest_trials] = np.nan
    
# get trials in each run block
run_block_trials = np.zeros(1, dtype='object')
run_block_trials[0] = run_trials.copy()
session_info['run_block_trials'] = run_block_trials.copy()

# determine number of trials of each stimulus type in each run block
session_info = fcn_trials_perFrequency_perRunBlock(session_info)


#%% if we are only doing resting trials, then set running trials to nan

if restOnly == True:
    
    # nan out running trials from pupil data
    session_info['trial_pupilMeasure'][run_trials] = np.nan


#%% get trials in each pupil block
session_info = fcn_get_trials_in_pupilBlocks(session_info)


#%% determine number of trials of each stimulus type in each pupil block
session_info = fcn_trials_perFrequency_perPupilBlock(session_info)


#%% get pupil sizes corresponding to beginning and end of each pupil block
pupilSize_percentileBlocks = fcn_pupilPercentile_to_pupilSize(session_info)
pupilBin_centers = np.mean(pupilSize_percentileBlocks, 0)


#%% number of pupil block
n_pupilBlocks = np.size(pupilSize_percentileBlocks[0,:])


#%% number of trials to subsample of each frequency in each pupil block
nTrials_subsample = int(session_info['max_nTrials'])
if nTrials_subsample <= nTrials_thresh:
    sys.exit('not enough trials')

#%% quantities for psth analyses

# number of time points
n_tPts = np.size(t_windows)

# number of cells
nCells = session_info['nCells']

# unique frequencies
nFreq = session_info['n_frequencies']
uniqueFreq = session_info['unique_frequencies']

# indices in baseline window
base_tInds = np.nonzero( (t_windows <= baseline_window[1]) & (t_windows > baseline_window[0]) )[0]
        
# indices in stimulus window
stim_tInds = np.nonzero( (t_windows <= stimulus_window[1]) & (t_windows >= stimulus_window[0]) )[0]


#%% quantities to compute

# single trial measures
singleTrial_psth = np.zeros((nTrials_subsample, nCells, n_tPts, nFreq, n_pupilBlocks))
singleTrial_gain = np.zeros((nTrials_subsample, nCells, n_tPts, nFreq, n_pupilBlocks))
singleTrial_gain_alt = np.zeros((nTrials_subsample, nCells, n_tPts, nFreq, n_pupilBlocks))

# trial averages
trialAvg_psth = np.zeros((nCells, n_tPts, nFreq, n_pupilBlocks))
trialAvg_gain = np.zeros((nCells, n_tPts, nFreq, n_pupilBlocks))
trialAvg_gain_alt = np.zeros((nCells, n_tPts, nFreq, n_pupilBlocks))

# trial sds
trialSd_psth = np.zeros((nCells, n_tPts, nFreq, n_pupilBlocks))
trialSd_gain = np.zeros((nCells, n_tPts, nFreq, n_pupilBlocks))
trialSd_gain_alt = np.zeros((nCells, n_tPts, nFreq, n_pupilBlocks))

# significance of responses
psth_pval = np.ones((nCells, n_tPts, nFreq, n_pupilBlocks))*np.inf
psth_pval_baseline = np.ones((nCells, n_tPts, nFreq, n_pupilBlocks))*np.inf


# loop over cells
for cellInd in range(0,nCells):
    
    # loop over pupil blocks
    for pupilInd in range(0, n_pupilBlocks):

        # loop over frequencies
        for freqInd in range(0,nFreq):
        
              
            # all trials for this frequency and pupil block
            all_trials = session_info['trials_perFreq_perPupilBlock'][freqInd, pupilInd].copy()
            
            # sample trials
            sample_trials = np.random.choice(all_trials, size = nTrials_subsample, replace = False)
    
            # single trial psth
            singleTrial_psth[:, cellInd, :, freqInd, pupilInd] = psth_allTrials[sample_trials, cellInd, :]
        
            # mean baseline of trial average rate
            trialAvg_baselineRate = np.mean(singleTrial_psth[:, cellInd, base_tInds, freqInd, pupilInd])
        
            # single trial response gain (substract baseline avg of trial avg rate)
            singleTrial_gain[:, cellInd, :, freqInd, pupilInd] = singleTrial_psth[:, cellInd, :, freqInd, pupilInd] - trialAvg_baselineRate
                                
        
            # single trial response gain alt
            for i in range(0, nTrials_subsample):
            
                # baseline rate of this trial
                baseAvg_rate = np.mean(singleTrial_psth[i, cellInd, base_tInds, freqInd, pupilInd])

                # single trial gain
                singleTrial_gain_alt[i, cellInd, :, freqInd, pupilInd] =  singleTrial_psth[i, cellInd, :, freqInd, pupilInd] - baseAvg_rate
                
  
            
            # statistical significance of stimulus response: compare stim response to baseline

            # baseline psth for all trials
            base_psth = singleTrial_psth[:, cellInd, base_tInds, freqInd, pupilInd].flatten()        

            # loop over stimulus time points
            for _, indT in enumerate(stim_tInds):
                       
                # get psth at this time point
                stim_psth = singleTrial_psth[:, cellInd, indT, freqInd, pupilInd]
                       
                # if stim and base psth's are the same, continue to next time point
                if ( np.all(stim_psth == 0) and np.all(base_psth == 0) ):
                
                    continue
            
                # run statistical test at this time point
                _, pval = fcn_MannWhitney_twoSided(base_psth, stim_psth)

                # store pval
                psth_pval[cellInd, indT, freqInd, pupilInd] = pval
                
                
                
            # statistical significance of baseline response (sanity check)
            # compare N data points at each baseline time point to (M-1)*(N) other baseline time points
            # loop over baseline time points
            for count, indT in enumerate(base_tInds):      
                
                # get psth at this time point
                base_psth_t = singleTrial_psth[:, cellInd, indT, freqInd, pupilInd]
                
                # make sure not all baseline time points are zero
                if np.all(base_psth == 0):
                    
                    continue
                
                # run statistical test
                compare_tPts = np.setdiff1d(base_tInds, indT)
                _, pval = fcn_MannWhitney_twoSided(base_psth_t, singleTrial_psth[:, cellInd, compare_tPts, freqInd, pupilInd].flatten())
                
                # store pval
                psth_pval_baseline[cellInd, indT, freqInd, pupilInd] = pval

                
                
            
            # trial avg psth
            trialAvg_psth[cellInd, :, freqInd, pupilInd] = np.mean(singleTrial_psth[:, cellInd, :, freqInd, pupilInd], axis=0)
            
            # trial average response gain
            trialAvg_gain[cellInd, :, freqInd, pupilInd] = np.mean(singleTrial_gain[:, cellInd, :, freqInd, pupilInd], axis=0)
        
            # trial average response gain alt
            trialAvg_gain_alt[cellInd, :, freqInd, pupilInd] = np.mean( singleTrial_gain_alt[:, cellInd, :, freqInd, pupilInd], axis=0 )
            
            # trial standard deviation gain
            trialSd_psth[cellInd, :, freqInd, pupilInd] = np.std(singleTrial_psth[:, cellInd, :, freqInd, pupilInd], axis=0)         
            
            # trial standard deviation gain
            trialSd_gain[cellInd, :, freqInd, pupilInd] = np.std(singleTrial_gain[:, cellInd, :, freqInd, pupilInd], axis=0)
            
            # trial standard deviation gain alt
            trialSd_gain_alt[cellInd, :, freqInd, pupilInd] = np.std(singleTrial_gain_alt[:, cellInd, :, freqInd, pupilInd], axis=0)            
            
                
#%% correct p values for multiple comparisons

psth_pval_corrected = psth_pval*np.size(stim_tInds)
psth_pval_baseline_corrected = psth_pval_baseline*np.size(base_tInds)


#%% SAVE DATA


params = {'session_path':         data_path, \
          'session_name':         session_name, \
          'stim_duration':        stim_duration, \
          'baseline_window':      baseline_window, \
          'stimulus_window':      stimulus_window, \
          'trial_window':         trial_window, \
          'window_length':        window_length, \
          'window_step':          window_step, \
          'pupilBlock_size':      pupilBlock_size, \
          'pupilBlock_step':      pupilBlock_step, \
          'pupilSplit_method':    pupilSplit_method, \
          'pupilSize_method':     pupilSize_method, \
          'restOnly':             restOnly, \
          'runThresh':            runThresh, \
          'runSpeed_method':      runSpeed_method, \
          'trialMatch':           trialMatch, \
          'nTrials_thresh':       nTrials_thresh}

    
results = {'params':                                params, \
           't_window':                              t_windows, \
           'nTrials_subsample':                     nTrials_subsample, \
           'uniqueFreq':                            uniqueFreq, \
           'base_tInds':                            base_tInds, \
           'stim_tInds':                            stim_tInds, \
           'trialAvg_psth':                         trialAvg_psth, \
           'trialAvg_gain':                         trialAvg_gain, \
           'trialAvg_gain_alt':                     trialAvg_gain_alt, \
           'trialSd_psth':                          trialSd_psth, \
           'trialSd_gain':                          trialSd_gain, \
           'trialSd_gain_alt':                      trialSd_gain_alt, \
           'psth_pval_corrected':                   psth_pval_corrected, \
           'psth_pval_baseline_corrected':          psth_pval_baseline_corrected, \
           'pupilBin_centers':                      pupilBin_centers, \
           'pupilSize_percentileBlocks':            pupilSize_percentileBlocks}
           
        
if ( (restOnly == True) and (trialMatch == False) ):
    fName_end = '_restOnly'
        
else:
    fName_end = ''
    
save_filename = ( (outpath + 'psth_pupil_%s%s_windLength%0.3fs.mat') % (session_name, fName_end, window_length) )      
savemat(save_filename, results) 

