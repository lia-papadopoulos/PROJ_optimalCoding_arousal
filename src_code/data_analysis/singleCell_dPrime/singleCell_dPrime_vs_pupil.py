

#%% IMPORTS

# basic imports
import sys        
import numpy as np
import numpy.matlib
import argparse
from scipy.io import savemat

# import settings file
import singleCell_dPrime_settings as settings

# paths to functions
sys.path.append(settings.func_path1)        
sys.path.append(settings.func_path2)

# main functions
from fcn_processedh5data_to_dict import fcn_processedh5data_to_dict
from fcn_SuData import fcn_makeTrials
from fcn_SuData import fcn_spikeTimes_trials_cells
from fcn_SuData import fcn_compute_windowed_spikeCounts
from fcn_SuData import fcn_trialInfo_eachFrequency
from fcn_SuData import fcn_compute_pupilMeasure_eachTrial
from fcn_SuData import fcn_get_trials_in_pupilBlocks
from fcn_SuData import fcn_trials_perFrequency_perPupilBlock
from fcn_SuData import fcn_pupilPercentile_to_pupilSize
from fcn_SuData import fcn_compute_avgRunSpeed_inTrials
from fcn_SuData import fcn_determine_runningTrials


#%% PARAMETERS

data_path = settings.data_path
outpath = settings.outpath
trial_window = settings.trial_window
stim_duration = settings.stim_duration
window_length = settings.window_length
window_step = settings.window_step
pupilBlock_size = settings.pupilBlock_size
pupilBlock_step = settings.pupilBlock_step
pupilSplit_method = settings.pupilSplit_method
n_subsamples = settings.n_subsamples
nTrials_thresh = settings.nTrials_thresh
pupilSize_method = settings.pupilSize_method
restOnly = settings.restOnly
trialMatch = settings.trialMatch
runThresh = settings.runThresh
runSpeed_method = settings.runSpeed_method
runBlock_size = settings.runBlock_size
runBlock_step = settings.runBlock_step
runSplit_method = settings.runSplit_method
rateDrift_cellSelection = settings.rateDrift_cellSelection
global_pupilNorm = settings.global_pupilNorm


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


#%% GET DATA

data_name = '' + '_rateDrift_cellSelection'*rateDrift_cellSelection + '_globalPupilNorm'*global_pupilNorm


session_info = fcn_processedh5data_to_dict(session_name, data_path, fname_end = data_name)


#%% UPDATE SESSION INFO


session_info['trial_window'] = trial_window

session_info['pupilBlock_size'] = pupilBlock_size
session_info['pupilBlock_step'] = pupilBlock_step
session_info['pupilSplit_method'] = pupilSplit_method
session_info['pupilSize_method'] = pupilSize_method

session_info['runSpeed_method'] = runSpeed_method
session_info['runSplit_method'] = runSplit_method
session_info['runBlock_size'] = runBlock_size
session_info['runBlock_step'] = runBlock_step

print('session updated')


#%% make trials
session_info = fcn_makeTrials(session_info)

#%% trials of each frequency
session_info = fcn_trialInfo_eachFrequency(session_info)

#%% compute spike times of each cell in every trial [aligned to stimulus onset]
session_info = fcn_spikeTimes_trials_cells(session_info)

#%% compute psth of each cell in every trial
t_windows, nSpikes = fcn_compute_windowed_spikeCounts(session_info, window_length, window_step)

#%% compute pupil measure in each trial
session_info = fcn_compute_pupilMeasure_eachTrial(session_info)

#%% running info 

# compute run speed in each trial
session_info = fcn_compute_avgRunSpeed_inTrials(session_info, session_info['trial_start'], session_info['trial_end'])

# classify trials as running or resting
session_info = fcn_determine_runningTrials(session_info, runThresh)

# rest and run trials
run_trials = np.nonzero(session_info['running'])[0]
rest_trials = np.nonzero(session_info['running']==0)[0]

# pupil size of running trials
pupilSize_runTrials = session_info['trial_pupilMeasure'].copy()
pupilSize_runTrials[rest_trials] = np.nan
    

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
    
    
#%% quantities for dprime analyses

# number of time points
n_tPts = np.size(t_windows)

# unique frequencies
nFreq = session_info['n_frequencies']
uniqueFreq = session_info['unique_frequencies']



#%% initialize quantities that we'll compute

nCells = session_info['nCells']

dprime = np.zeros((n_tPts, nCells, nFreq, nFreq, n_pupilBlocks, n_subsamples))

base_rate = np.zeros((nCells, nFreq, n_pupilBlocks, n_subsamples))

avg_pupilSize_dprimeTrials_pupilBlocks = np.zeros((n_pupilBlocks))


#%% compute d prime 

        
# loop over pupil blocks
for pupilInd in range(0, n_pupilBlocks):

    
    allTrials_sampled = np.array([])
    

    # loop over frequencies
    for freqInd_A in range(0,nFreq):
        
        # all trials for this frequency and pupil block
        all_trials_A = session_info['trials_perFreq_perPupilBlock'][freqInd_A, pupilInd].copy()
        
        for freqInd_B in range(0, nFreq):
               
            # all trials for this frequency and pupil block
            all_trials_B = session_info['trials_perFreq_perPupilBlock'][freqInd_B, pupilInd].copy()
        
            # for each subsample
            for indSample in range(0, n_subsamples):
            
                # sample trials
                sample_trials_A = np.random.choice(all_trials_A, size = nTrials_subsample, replace = False)
                sample_trials_B = np.random.choice(all_trials_B, size = nTrials_subsample, replace = False)
            
                allTrials_sampled = np.rint(np.append(allTrials_sampled, np.append(sample_trials_A, sample_trials_B))).astype(int)
            
                # loop over cells
                for cellInd in range(0,nCells):
                    
                    # baseline spike count
                    base_rate[cellInd, freqInd_A, pupilInd, indSample] = np.mean(nSpikes[sample_trials_A, cellInd, 0].copy(), axis=0)/window_length
            
                    # loop over time
                    for timeInd in range(0, n_tPts):
    
                        # spike counts
                        spkCountsA = nSpikes[sample_trials_A, cellInd, timeInd].copy()
                        spkCountsB = nSpikes[sample_trials_B, cellInd, timeInd].copy()

                        # average across trials
                        mu_A = np.mean(spkCountsA)
                        mu_B = np.mean(spkCountsB)
                        
                        # variance across trials
                        var_A = np.var(spkCountsA)
                        var_B = np.var(spkCountsB)
                        
                        # dprime_AB
                        dprime[timeInd, cellInd, freqInd_A, freqInd_B, pupilInd, indSample] = (np.abs(mu_A - mu_B))/( np.sqrt( (1/2)*(var_A + var_B) ) )


    avg_pupilSize_dprimeTrials_pupilBlocks[pupilInd] = np.mean(session_info['trial_pupilMeasure'][allTrials_sampled])

# average dprime across samples
sampleAvg_dprime = np.nanmean(dprime, axis=5)

# average dprime across frequency pairs
freqAvg_sampleAvg_dprime = np.nanmean(sampleAvg_dprime, axis=(2,3)) # time, cells, pupil

# average baseline spike count across samples
sampleAvg_base_rate = np.mean(base_rate, axis=3)

# average baseline spike count across frequencies
freqAvg_sampleAvg_base_rate = np.mean(sampleAvg_base_rate, axis=1)


#%% SAVE THE RESULTS


params = {'session_path':         data_path, \
          'session_name':         session_name, \
          'stim_duration':        stim_duration, \
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
          'runBlock_size':        runBlock_size, \
          'runBlock_step':        runBlock_step, \
          'trialMatch':           trialMatch, \
          'nTrials_thresh':       nTrials_thresh, \
          'rateDrift_cellSelection': rateDrift_cellSelection, \
          'n_subsamples':         n_subsamples}

    
results = {'params':                                params, \
           'nTrials_subsample':                     nTrials_subsample, \
           'uniqueFreq':                            uniqueFreq, \
           'freqAvg_sampleAvg_dprime':              freqAvg_sampleAvg_dprime, \
           'freqAvg_sampleAvg_base_rate':           freqAvg_sampleAvg_base_rate, \
           'pupilBin_centers':                      pupilBin_centers, \
           'pupilSize_percentileBlocks':            pupilSize_percentileBlocks, \
           'avg_pupilSize_dprimeTrials_pupilBlocks': avg_pupilSize_dprimeTrials_pupilBlocks}
           
        
if ( (restOnly == True) and (trialMatch == False) ):
    fName_end = 'restOnly'
        
else:
    fName_end = ''
    
        

save_filename = ( (outpath + 'singleCell_dPrime_pupil_%s_windLength%0.3fs_%s%s.mat') % (session_name, window_length, fName_end, data_name) )      
savemat(save_filename, results) 
                        
                    
                    
                    
                    
                    
           
                    
                    