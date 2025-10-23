#%% basic imports
import sys     
import numpy as np
import numpy.matlib
import argparse
from scipy.io import savemat

#%% settings
import fano_factor_settings as settings

#%% paths to functions
sys.path.append(settings.func_path1)        
sys.path.append(settings.func_path2)
from fcn_processedh5data_to_dict import fcn_processedh5data_to_dict
from fcn_SuData import fcn_makeTrials
from fcn_SuData import fcn_spikeTimes_trials_cells
from fcn_SuData import fcn_trialInfo_eachFrequency
from fcn_SuData import fcn_compute_pupilMeasure_eachTrial
from fcn_SuData import fcn_get_trials_in_pupilBlocks
from fcn_SuData import fcn_pupilPercentile_to_pupilSize
from fcn_SuData import fcn_getTrials_in_pupilRange
from fcn_SuData import fcn_compute_avgRunSpeed_inTrials
from fcn_SuData import fcn_determine_runningTrials
from fcn_SuData import fcn_trials_perFrequency_perRunBlock
from fcn_SuData import fcn_makeTrials_spont
from fcn_SuData import fcn_spikeTimes_trials_cells_spont
from fcn_SuData import fcn_compute_spikeCnts_inTrials
from fcn_SuData import fcn_compute_avgPupilSize_inTrials
from fcn_SuData import fcn_compute_windowed_spikeCounts
from fcn_SuData import fcn_trials_perFrequency_perPupilBlock
from fcn_SuData import fcn_fanoFactor

#%% unpack parameters

data_path = settings.data_path
outpath = settings.outpath
window_length = settings.window_length
window_step = settings.window_step
inter_window_interval = settings.inter_window_interval
trial_window_evoked = settings.trial_window_evoked
stim_duration = settings.stim_duration
pupilBlock_size = settings.pupilBlock_size
pupilBlock_step = settings.pupilBlock_step
pupilSplit_method = settings.pupilSplit_method
pupilLag = settings.pupilLag
nTrials_thresh = settings.nTrials_thresh
n_subsamples = settings.n_subsamples
pupilSize_method = settings.pupilSize_method
restOnly = settings.restOnly
trialMatch = settings.trialMatch
runThresh = settings.runThresh
runSpeed_method = settings.runSpeed_method
runBlock_size = settings.runBlock_size
runBlock_step = settings.runBlock_step
runSplit_method = settings.runSplit_method
global_pupilNorm = settings.global_pupilNorm
highDownsample = settings.highDownsample
cellSelection = settings.cellSelection

#%% checks

if restOnly == True:
    sys.exit('need to make sure this code works for resting data only')

if ( (restOnly == True) and (trialMatch == True) ):
    sys.exit('cant do rest only w/ trial matching yet; need to add multiple subsamples')
    
    

#%% user input

# argparser
parser = argparse.ArgumentParser() 

# session name
parser.add_argument('-session_name', '--session_name', type=str, default = '')
    
# arguments of parser
args = parser.parse_args()

# argparse inputs
session_name = args.session_name


#%% get data

data_name = '' + cellSelection + '_globalPupilNorm'*global_pupilNorm + '_downSampled'*highDownsample

session_info = fcn_processedh5data_to_dict(session_name, data_path, fname_end = data_name)

nCells = session_info['nCells']


#%% update session info

session_info['trial_window'] = trial_window_evoked

session_info['pupilSize_method'] = pupilSize_method
session_info['pupilSplit_method'] = pupilSplit_method
session_info['pupilBlock_size'] = pupilBlock_size
session_info['pupilBlock_step'] = pupilBlock_step
session_info['pupil_lag'] = pupilLag

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
    
# get trials in each run block
run_block_trials = np.zeros(1, dtype='object')
run_block_trials[0] = run_trials
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

#%% number of pupil blocks
n_pupilBlocks = len(session_info['pupil_block_trials'])
    
#%% number of trials to subsample of each frequency in each pupil block
nTrials_subsample_evoked = int(session_info['max_nTrials'])

#%% quantities for fano factor analysis

# unique frequencies
nFreq = session_info['n_frequencies']
uniqueFreq = session_info['unique_frequencies']

# evoked trial spike times
all_trials_evoked = session_info['trials_perFreq_perPupilBlock'].copy()
t_window, singleTrial_spikeCounts_evoked = fcn_compute_windowed_spikeCounts(session_info, window_length, window_step)
nWindows = len(t_window)

# average pupil size of all evoked trials
avg_pupilSize_allTrials_evoked = session_info['trial_pupilMeasure'].copy()

#%% pupil size corresponding to each percentile block

# pupil size corresponding ot beginning and end of each pupil percentile bine
pupilSize_percentileBlocks_evoked = fcn_pupilPercentile_to_pupilSize(session_info)

# pupil sizes corresponding to beginning and end of each pupil block
pupilBin_centers_evoked = np.mean(pupilSize_percentileBlocks_evoked, 0)

#%% ------------------ SPONTANEOUS DATA ANALYSIS ----------------------------------

del session_info

session_info = fcn_processedh5data_to_dict(session_name, data_path, fname_end = data_name)


#%% start and end of spontaneous blocks

spont_blocks = session_info['spont_blocks']

session_info['spontBlock_start'] = spont_blocks[0,:].copy()
session_info['spontBlock_end'] = spont_blocks[1,:].copy()
session_info['n_spontBlocks'] = np.size(session_info['spontBlock_start'])

#%% make trials

session_info = fcn_makeTrials_spont(session_info, window_length, inter_window_interval)
trial_start = session_info['trial_start'].copy()
trial_end = session_info['trial_end'].copy()

print('made trials')

#%% pupil splitting information for spontaneous blocks

session_info['pupilBlock_size'] = pupilBlock_size
session_info['pupilBlock_step'] = pupilBlock_step
session_info['pupilSplit_method'] = pupilSplit_method

#%% spike times
session_info = fcn_spikeTimes_trials_cells_spont(session_info)

#%% spike counts of all cell sin all trials
session_info = fcn_compute_spikeCnts_inTrials(session_info)

#%% average pupil size across window

avg_pupilSize = fcn_compute_avgPupilSize_inTrials(session_info, trial_start+pupilLag, trial_end+pupilLag)
session_info['trial_pupilMeasure'] = avg_pupilSize.copy()

#%% running speed

session_info = fcn_compute_avgRunSpeed_inTrials(session_info, trial_start, trial_end)

session_info = fcn_determine_runningTrials(session_info, runThresh)

# running trials
run_trials = np.nonzero(session_info['running'])[0]
rest_trials = np.nonzero(session_info['running']==0)[0]

if restOnly == True:
    
    # nan out running trials from pupil data
    session_info['trial_pupilMeasure'][run_trials] = np.nan
    
#%% bin trials according to evoked pupil percentile bins

pupil_block_trials = np.zeros(n_pupilBlocks, dtype='object')
nTrials_in_pupilBlocks = np.zeros((n_pupilBlocks))


for indPupil in range(0, n_pupilBlocks):

    lowPupil = pupilSize_percentileBlocks_evoked[0, indPupil]
    highPupil = pupilSize_percentileBlocks_evoked[1, indPupil]
    
    pupil_block_trials[indPupil] = fcn_getTrials_in_pupilRange(session_info, lowPupil, highPupil)
    
    nTrials_in_pupilBlocks[indPupil] = np.size( pupil_block_trials[indPupil] )


session_info['pupil_block_trials'] = pupil_block_trials.copy()

#%% number of trials to subsample in each pupil block

nTrials_subsample_spont = np.min(nTrials_in_pupilBlocks)
nTrials_subsample_spont = int(nTrials_subsample_spont)

#%% average pupil size of spont trials in each block
avg_pupilSize_allTrials_spont = session_info['trial_pupilMeasure'].copy()

#%% quantities for fano factor

all_trials_spont = session_info['pupil_block_trials'].copy()
singleTrial_spikeTimes_spont = session_info['spikeTimes_trials_cells'].copy()
singleTrial_spikeCounts_spont = session_info['spkCounts_trials_cells'].copy()

#%% clear session info
del session_info

#%% number of trials to subsample

nTrials_subsample = np.min(np.array([nTrials_subsample_evoked, nTrials_subsample_spont])).astype(int)
    
if nTrials_subsample < nTrials_thresh:
    print(nTrials_subsample)
    sys.exit('not enough trials')


#%% quantities to compute

# fano factor of every cell for each frequency in each pupil block
spont_fano = np.zeros((nCells, nFreq, n_pupilBlocks, n_subsamples))
evoked_fano = np.zeros((nCells, nFreq, n_pupilBlocks, n_subsamples, nWindows))

diff_fanofactor = np.zeros((nCells, nFreq, n_pupilBlocks, nWindows))
diff_fanofactor_pre = np.zeros((nCells, nFreq, n_pupilBlocks, nWindows))

spont_trialAvg_spikeCount = np.zeros((nCells, n_pupilBlocks))
evoked_trialAvg_spikeCount = np.zeros((nCells, nFreq, n_pupilBlocks, nWindows))
pre_trialAvg_spikeCount = np.zeros((nCells, nFreq, n_pupilBlocks))

# average pupil size of low and high pupil trials
avg_pupilSize_spontTrials = np.zeros((n_pupilBlocks, n_subsamples))
avg_pupilSize_evokedTrials = np.zeros((n_pupilBlocks, nFreq, n_subsamples))


for indSample in range(0, n_subsamples):

    for pupilInd in range(0, n_pupilBlocks):
                
        # sample trials
        trial_inds_spont = np.random.choice(all_trials_spont[pupilInd], size = nTrials_subsample, replace = False).astype(int)
        # average pupil size of sampled trials
        avg_pupilSize_spontTrials[pupilInd, indSample] = np.mean(avg_pupilSize_allTrials_spont[trial_inds_spont])

        for freqInd in range(0,nFreq):    
            
            # sample trials
            trial_inds_evoked = np.random.choice(all_trials_evoked[freqInd, pupilInd], size = nTrials_subsample, replace = False).astype(int)
            # average pupil size of sampled trials
            avg_pupilSize_evokedTrials[pupilInd, freqInd, indSample] = np.mean(avg_pupilSize_allTrials_evoked[trial_inds_evoked])
            
            # for each cell
            for cellInd in range(0,nCells):

                # spontaneous spike counts
                all_trials_spont_spike_counts = singleTrial_spikeCounts_spont[all_trials_spont[pupilInd], cellInd].copy()
                spont_spike_counts = singleTrial_spikeCounts_spont[trial_inds_spont, cellInd].copy()
                
                # spontaneous fano
                spont_fano[cellInd, freqInd, pupilInd, indSample] = fcn_fanoFactor(spont_spike_counts)
                
                # evoked spike counts
                all_trials_evoked_spike_counts = singleTrial_spikeCounts_evoked[all_trials_evoked[freqInd, pupilInd], cellInd].copy()                      
                evoked_spike_counts = singleTrial_spikeCounts_evoked[trial_inds_evoked, cellInd, :].copy()
                
                # evoked fano
                evoked_fano[cellInd, freqInd, pupilInd, indSample, :] = fcn_fanoFactor(evoked_spike_counts)
                
                # trial average spike counts
                spont_trialAvg_spikeCount[cellInd, pupilInd] = np.mean(all_trials_spont_spike_counts)
                evoked_trialAvg_spikeCount[cellInd, freqInd, pupilInd,:] = np.mean(all_trials_evoked_spike_counts,0)
                pre_trialAvg_spikeCount[cellInd, freqInd, pupilInd] = np.mean(all_trials_evoked_spike_counts[:, 0])
                

# average over subsamples
spont_fanofactor = np.nanmean(spont_fano, axis = (3, 1))
evoked_fanofactor = np.nanmean(evoked_fano, axis = 3)

# diff fano
for indWindow in range(0, nWindows):
    for indFreq in range(0, nFreq):
        diff_fanofactor[:, indFreq, :, indWindow] = evoked_fanofactor[:, indFreq, :, indWindow] - spont_fanofactor
        diff_fanofactor_pre[:, indFreq, :, indWindow] = evoked_fanofactor[:, indFreq, :, indWindow] - evoked_fanofactor[:, indFreq, :, 0]
                                

avg_pupilSize_spontTrials = np.mean(avg_pupilSize_spontTrials, 1)
avg_pupilSize_evokedTrials = np.mean(avg_pupilSize_evokedTrials, axis=(1,2))


#%% SAVE RESULTS


params = {'session_path':         data_path, \
          'session_name':         session_name, \
          'stim_duration':        stim_duration, \
          'trial_window_evoked':  trial_window_evoked, \
          'window_length':        window_length, \
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
          'cellSelection':        cellSelection, \
          'window_step':          window_step}
    
    
results = {'params':                                params, \
           'nTrials_subsample':                     nTrials_subsample, \
           'uniqueFreq':                            uniqueFreq, \
           'diff_fanofactor':                       diff_fanofactor, \
           'spont_fanofactor':                      spont_fanofactor, \
           'evoked_fanofactor':                     evoked_fanofactor, \
           'spont_trialAvg_spikeCount':             spont_trialAvg_spikeCount, \
           'evoked_trialAvg_spikeCount':            evoked_trialAvg_spikeCount, \
           'pupilBin_centers_evoked':                      pupilBin_centers_evoked, \
           'pupilSize_percentileBlocks_evoked':            pupilSize_percentileBlocks_evoked, \
           'avg_pupilSize_evokedTrials':                    avg_pupilSize_evokedTrials, \
           'avg_pupilSize_spontTrials':                    avg_pupilSize_spontTrials, \
           'diff_fanofactor_pre':                           diff_fanofactor_pre, \
           'pre_trialAvg_spikeCount':                       pre_trialAvg_spikeCount, \
           't_window':                                      t_window}
    

        

           

if ( (restOnly == True) and (trialMatch == False) ):
    fName_end = 'restOnly'
        
else:
    fName_end = ''
    

save_filename = ( (outpath + 'spont_evoked_fanofactor_pupilPercentile_raw_%s_windLength%0.3fs_pupilLag%0.3fs_%s%s.mat') % (session_name, window_length, pupilLag, fName_end, data_name) )      
savemat(save_filename, results) 


