

#%% basic imports
import sys     
import numpy as np
import argparse
from scipy.io import savemat
import numpy.matlib

#%% settings
import evoked_corr_settings as settings

#%% functions
sys.path.append(settings.func_path1)        
sys.path.append(settings.func_path2)

from fcn_statistics import fcn_zscore
from fcn_processedh5data_to_dict import fcn_processedh5data_to_dict
from fcn_SuData import fcn_compute_spikeCount_corr
from fcn_SuData import fcn_compute_spkCounts_inWindow
from fcn_SuData import fcn_trial_shuffled_spikeCounts
from fcn_SuData import fcn_makeTrials
from fcn_SuData import fcn_spikeTimes_trials_cells
from fcn_SuData import fcn_trialInfo_eachFrequency
from fcn_SuData import fcn_compute_pupilMeasure_eachTrial
from fcn_SuData import fcn_trials_perFrequency_perPupilBlock
from fcn_SuData import fcn_get_trials_in_pupilBlocks


#%% unpack settings
zscore_withinPupil = settings.zscore_withinPupil
base_subtract = settings.base_subtract
window_length = settings.window_length
trial_window_evoked = settings.trial_window_evoked
pupilSize_method = settings.pupilSize_method
nTrials_thresh = settings.nTrials_thresh
n_subsamples = settings.n_subsamples
pupilBlock_size = settings.pupilBlock_size
pupilBlock_step = settings.pupilBlock_step
pupilSplit_method = settings.pupilSplit_method
data_path = settings.data_path
outpath = settings.outpath
global_pupilNorm = settings.global_pupilNorm
highDownsample = settings.highDownsample
cellSelection = settings.cellSelection

#%% user input

# argparser
parser = argparse.ArgumentParser() 

# session name
parser.add_argument('-session_name', '--session_name', type=str, default = '')
    
# arguments of parser
args = parser.parse_args()

# argparse inputs
session_name = args.session_name

#%% get data for this session

data_name = '' + cellSelection + '_globalPupilNorm'*global_pupilNorm + '_downSampled'*highDownsample

session_info = fcn_processedh5data_to_dict(session_name, data_path, fname_end = data_name)

#%% number of cells
nCells = session_info['nCells']

#%% update session dictionary
session_info['trial_window'] = trial_window_evoked
session_info['pupilSize_method'] = pupilSize_method
session_info['pupilBlock_size'] = pupilBlock_size
session_info['pupilBlock_step'] = pupilBlock_step
session_info['pupilSplit_method'] = pupilSplit_method


#%% make trials
session_info = fcn_makeTrials(session_info)

#%% trials of each frequency
session_info = fcn_trialInfo_eachFrequency(session_info)

#%% compute spike times of each cell in every trial [aligned to stimulus onset]
session_info = fcn_spikeTimes_trials_cells(session_info)
singleTrial_spikeTimes = session_info['spikeTimes_trials_cells'].copy()

#%% spike counts of all cell sin all trials

singleTrial_spikeCounts_evoked = np.zeros(np.shape(singleTrial_spikeTimes))
singleTrial_spikeCounts_base = np.zeros(np.shape(singleTrial_spikeTimes))

for indCell in range(0, nCells):
    singleTrial_spikeCounts_evoked[:, indCell] = fcn_compute_spkCounts_inWindow(singleTrial_spikeTimes[:,indCell], 0., window_length)
    singleTrial_spikeCounts_base[:, indCell] = fcn_compute_spkCounts_inWindow(singleTrial_spikeTimes[:,indCell], -window_length,0)
    
if base_subtract:
    singleTrial_spikeCounts_evoked = singleTrial_spikeCounts_evoked - singleTrial_spikeCounts_base

#%% compute pupil measure in each trial
session_info = fcn_compute_pupilMeasure_eachTrial(session_info)

# trials in pupil blocks
session_info = fcn_get_trials_in_pupilBlocks(session_info)

#%% determine number of trials of each stimulus type in each pupil block
session_info = fcn_trials_perFrequency_perPupilBlock(session_info)

nPupilBins = np.size(session_info['nTrials_perFreq_perPupilBlock'], 1)

nTrials_subsample = np.array([])

for indPupil in range(0, nPupilBins):
    
    nTrials = np.min(session_info['nTrials_perFreq_perPupilBlock'][:, indPupil])

    nTrials_subsample = np.append(nTrials_subsample, nTrials)

nTrials_subsample = int(np.min(nTrials_subsample))


print(session_info['nTrials_perFreq_perPupilBlock'])

#%% quantities for correlation

# unique frequencies
nFreq = session_info['n_frequencies']
uniqueFreq = session_info['unique_frequencies']

# evoked trial spike times
all_trials_evoked = session_info['trials_perFreq_perPupilBlock'].copy()

# average pupil size
avg_pupilSize_allTrials_evoked = session_info['trial_pupilMeasure'].copy()


print(nTrials_subsample)
if nTrials_subsample < nTrials_thresh:
    sys.exit('not enough trials')


#%% initialize

# pairwise correlations in each pupil bin
corr_evoked_allPupil = np.ones((nCells, nCells, n_subsamples))*np.nan
corr_evoked_allPupil_shuffle = np.ones((nCells, nCells, n_subsamples))*np.nan

corr_evoked_allPupil_eachFreq = np.ones((nCells, nCells, nFreq, n_subsamples))*np.nan
corr_evoked_allPupil_eachFreq_shuffle = np.ones((nCells, nCells, nFreq, n_subsamples))*np.nan

corr_evoked_eachPupil_eachFreq = np.ones((nCells, nCells, nPupilBins, nFreq, n_subsamples))*np.nan
corr_evoked_eachPupil_eachFreq_shuffle = np.ones((nCells, nCells, nPupilBins, nFreq, n_subsamples))*np.nan

corr_base_allPupil = np.ones((nCells, nCells, n_subsamples))*np.nan
corr_base_allPupil_shuffle = np.ones((nCells, nCells, n_subsamples))*np.nan

trialAvg_evoked_spkCount = np.ones((nCells, nPupilBins, nFreq, n_subsamples))*np.nan
trialAvg_base_spkCount = np.ones((nCells, nPupilBins, nFreq, n_subsamples))*np.nan

avg_pupilSize_evokedTrials = np.ones((nPupilBins, nFreq, n_subsamples))*np.nan

#%% compute spike count correlations

for indSample in range(0, n_subsamples):

    allPupil_evoked_spike_counts = np.ones((1,nCells))*np.nan
    
    spkCounts_all = np.array([])
    spkCounts_all.shape = (0, nCells)

    spkCounts_all_shuffle = np.array([])
    spkCounts_all_shuffle.shape = (0, nCells)

    spkCounts_all_base = np.array([])
    spkCounts_all_base.shape = (0, nCells)

    spkCounts_all_base_shuffle = np.array([])
    spkCounts_all_base_shuffle.shape = (0, nCells)
    
    for freqInd in range(0,nFreq):
        
        spkCounts_all_eachFreq = np.array([])
        spkCounts_all_eachFreq.shape = (0, nCells)
        
        spkCounts_all_eachFreq_shuffle = np.array([])
        spkCounts_all_eachFreq_shuffle.shape = (0, nCells)
        
        
        for pupilInd in range(0, nPupilBins):
            
            # sample trials
            trial_inds_evoked = np.random.choice(all_trials_evoked[freqInd, pupilInd], size = nTrials_subsample, replace = False).astype(int)
            
            # evoked spike counts
            evoked_spike_counts = singleTrial_spikeCounts_evoked[trial_inds_evoked, :].copy()

            # base spike counts
            base_spike_counts = singleTrial_spikeCounts_base[trial_inds_evoked, :].copy()
            
            # trial average
            trialAvg_evoked_spkCount[:, pupilInd, freqInd, indSample] = np.mean(evoked_spike_counts, 0)
            trialAvg_base_spkCount[:, pupilInd, freqInd, indSample] = np.mean(base_spike_counts, 0)
            
            # zscoring
            if zscore_withinPupil:
                for indCell in range(0, nCells):
                    evoked_spike_counts[:, indCell] = fcn_zscore(evoked_spike_counts[:, indCell])
                    base_spike_counts[:, indCell] = fcn_zscore(base_spike_counts[:, indCell])
                    
                    
            spkCounts_all = np.vstack((spkCounts_all, evoked_spike_counts))
            spkCounts_base_all = np.vstack((spkCounts_all_base, base_spike_counts))
            spkCounts_all_eachFreq = np.vstack((spkCounts_all_eachFreq, evoked_spike_counts))

            # average pupil size of evoked trials
            avg_pupilSize_evokedTrials[pupilInd, freqInd, indSample] = np.mean(avg_pupilSize_allTrials_evoked[trial_inds_evoked])       
            
            # evoked correlation for each frequency
            corr_evoked_eachPupil_eachFreq[:,:,pupilInd,freqInd,indSample] = fcn_compute_spikeCount_corr(evoked_spike_counts)
            
            # evoked correlation of shuffled data
            spkCounts_eachPupil_eachFreq_shuffle, _ = fcn_trial_shuffled_spikeCounts(evoked_spike_counts)
            corr_evoked_eachPupil_eachFreq_shuffle[:,:,pupilInd,freqInd,indSample] = fcn_compute_spikeCount_corr(spkCounts_eachPupil_eachFreq_shuffle)
            
            
        # all pupils combined
        spkCounts_all_eachFreq_shuffle, _ = fcn_trial_shuffled_spikeCounts(spkCounts_all_eachFreq)
        corr_evoked_allPupil_eachFreq[:, :, freqInd, indSample] = fcn_compute_spikeCount_corr(spkCounts_all_eachFreq)
        corr_evoked_allPupil_eachFreq_shuffle[:,:,freqInd,indSample] = fcn_compute_spikeCount_corr(spkCounts_all_eachFreq_shuffle)

    # shuffle spike counts from all pupil bins and tones
    spkCounts_all_shuffle, _ = fcn_trial_shuffled_spikeCounts(spkCounts_all)
    spkCounts_base_all_shuffle, _ = fcn_trial_shuffled_spikeCounts(spkCounts_base_all)


    # spike count correlations computed from spike counts combined across pupils and tones 
    corr_evoked_allPupil[:, :, indSample] = fcn_compute_spikeCount_corr(spkCounts_all)
    corr_evoked_allPupil_shuffle[:, :, indSample] = fcn_compute_spikeCount_corr(spkCounts_all_shuffle)

    corr_base_allPupil[:, :, indSample] = fcn_compute_spikeCount_corr(spkCounts_base_all)
    corr_base_allPupil_shuffle[:, :, indSample] = fcn_compute_spikeCount_corr(spkCounts_base_all_shuffle)
                


# averaging
corr_evoked_allPupil = np.nanmean(corr_evoked_allPupil, axis=(2))

corr_evoked_eachPupil_freqAvg = np.mean(corr_evoked_eachPupil_eachFreq, axis=(3,4))
corr_evoked_eachPupil_freqAvg_shuffle = np.mean(corr_evoked_eachPupil_eachFreq_shuffle, axis=(3))

corr_base_allPupil = np.nanmean(corr_base_allPupil, axis=(2))

corr_evoked_allPupil_freqAvg = np.nanmean(corr_evoked_allPupil_eachFreq, axis=(2,3))

corr_evoked_pupilAvg_freqAvg = np.nanmean(corr_evoked_eachPupil_eachFreq, axis=(2,3,4))

avg_pupilSize_evokedTrials = np.mean(avg_pupilSize_evokedTrials, axis=(1,2))
trialAvg_evoked_spkCount = np.mean(trialAvg_evoked_spkCount, axis=(1,2,3))
trialAvg_base_spkCount = np.mean(trialAvg_base_spkCount, axis=(1,2,3))

corr_evoked_pupilAvg_freqAvg_shuffle = np.nanmean(corr_evoked_eachPupil_eachFreq_shuffle, axis=(2,3))

#%% SAVE RESULTS


params = {'session_path':         data_path, \
          'session_name':         session_name, \
          'window_length':        window_length, \
          'trial_window_evoked':  trial_window_evoked, \
          'nSubsamples':          n_subsamples, \
          'pupilSize_method':     pupilSize_method, \
          'pupilBlock_size':      pupilBlock_size, \
          'pupilBlock_step':      pupilBlock_step, \
          'pupilSplit_method':    pupilSplit_method, \
          'nTrials_thresh':       nTrials_thresh, \
          'cellSelection':        cellSelection, \
          'global_pupilNorm':     global_pupilNorm, \
          'highDownsample':       highDownsample, \
          'zscore_withinPupil':   zscore_withinPupil}

    
results = {'params':                                params, \
           'nTrials_subsample':                     nTrials_subsample, \
           'avg_pupilSize_evokedTrials':                  avg_pupilSize_evokedTrials, \
           'trialAvg_evoked_spkCount':         trialAvg_evoked_spkCount, \
           'trialAvg_base_spkCount':         trialAvg_base_spkCount, \
           'corr_evoked_eachPupil_eachFreq':    corr_evoked_eachPupil_eachFreq, \
           'corr_evoked_allPupil':                              corr_evoked_allPupil, \
           'corr_evoked_allPupil_shuffle':                      corr_evoked_allPupil_shuffle,\
           'corr_evoked_eachPupil_freqAvg':                     corr_evoked_eachPupil_freqAvg, \
           'corr_evoked_eachPupil_freqAvg_shuffle':             corr_evoked_eachPupil_freqAvg_shuffle, \
           'corr_base_allPupil':                              corr_base_allPupil, \
           'corr_base_allPupil_shuffle':                      corr_base_allPupil_shuffle,\
           'corr_evoked_allPupil_freqAvg':                   corr_evoked_allPupil_freqAvg, \
           'corr_evoked_allPupil_eachFreq_shuffle':         corr_evoked_allPupil_eachFreq_shuffle, \
           'corr_evoked_pupilAvg_freqAvg':                corr_evoked_pupilAvg_freqAvg, \
           'corr_evoked_eachPupil_eachFreq_shuffle':        corr_evoked_eachPupil_eachFreq_shuffle, \
           'corr_evoked_pupilAvg_freqAvg_shuffle':          corr_evoked_pupilAvg_freqAvg_shuffle
}            
        

base_subtract_str = '_baselineSubtract'*base_subtract
save_filename = ( (outpath + 'evoked_correlations_pupilPercentile_combinedBlocks_%s_windLength%0.3fs_%s%s.mat') % (session_name, window_length, base_subtract_str, data_name) )      
savemat(save_filename, results) 
