

# basic imports
import sys        
import numpy as np
import numpy.matlib
import argparse
from scipy.io import savemat

# import settings file
import psth_allTrials_settings as settings

# paths to functions
sys.path.append(settings.func_path1)        
sys.path.append(settings.func_path2)
         
# main functions
from fcn_processedh5data_to_dict import fcn_processedh5data_to_dict
from fcn_statistics import fcn_Wilcoxon
from fcn_statistics import fcn_MannWhitney_twoSided
from fcn_SuData import fcn_makeTrials
from fcn_SuData import fcn_spikeTimes_trials_cells
from fcn_SuData import fcn_compute_windowed_spikeCounts
from fcn_SuData import fcn_trialInfo_eachFrequency


#%% settings

# paths
data_path = settings.data_path
outpath = settings.outpath
trial_window = settings.trial_window
baseline_window = settings.baseline_window
stimulus_window = settings.stimulus_window
window_length = settings.window_length
window_step = settings.window_step
rateDrift_cellSelection = settings.rateDrift_cellSelection
global_pupilNorm = settings.global_pupilNorm
highDownsample = settings.highDownsample

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

data_name = '' + '_rateDrift_cellSelection'*rateDrift_cellSelection + '_globalPupilNorm'*global_pupilNorm + '_downSampled'*highDownsample

session_info = fcn_processedh5data_to_dict(session_name, data_path, fname_end = data_name)

#%% UPDATE SESSION INFO

session_info['trial_window'] = trial_window

print('session updated')

#%% make trials
session_info = fcn_makeTrials(session_info)

#%% trials of each frequency
session_info = fcn_trialInfo_eachFrequency(session_info)

#%% compute spike times of each cell in every trial [aligned to stimulus onset]
session_info = fcn_spikeTimes_trials_cells(session_info)

#%% compute psth of each cell to each frequency
t_windows, nSpikes = fcn_compute_windowed_spikeCounts(session_info, window_length, window_step)

#%% compute time-varying rate of each cell to each frequency
psth_allTrials = nSpikes/window_length

#%% quantities for psth analyses

# number of time points
n_tPts = np.size(t_windows)

# unique frequencies
nFreq = session_info['n_frequencies']
uniqueFreq = session_info['unique_frequencies']

# minimum number of trials of each frequency
min_nTrials_perFreq = int(np.rint(np.min(session_info['nTrials_eachFrequency'])))

# indices in baseline window
base_tInds = np.nonzero( (t_windows <= baseline_window[1]) & (t_windows > baseline_window[0]) )[0]
base_sig_tInd = np.argmin(np.abs(t_windows - 0.))
        
# indices in stimulus window
stim_tInds = np.nonzero( (t_windows <= stimulus_window[1]) & (t_windows >= stimulus_window[0]) )[0]
stim_sig_tInd = np.argmin(np.abs(t_windows - window_length))
        
# number of stimulus windows
n_stimWindows = np.size(stim_tInds)

# number of cells
nCells = session_info['nCells']

print(t_windows[base_sig_tInd], t_windows[stim_sig_tInd])



#%% quantities to compute

trialAvg_psth = np.zeros((nCells, n_tPts, nFreq))
psth_pval = np.ones((nCells, n_tPts, nFreq))*np.inf
psth_pval_baseline = np.ones((nCells, n_tPts, nFreq))*np.inf
pval_preStim_vs_postStim = np.ones((nCells, nFreq))*np.nan

delta_trialAvg_psth = np.zeros((nCells, n_tPts, nFreq))
trialAvg_gain = np.zeros((nCells, n_tPts, nFreq))
trialAvg_gain_alt = np.zeros((nCells, n_tPts, nFreq))


# loop over frequencies
for freqInd in range(0, nFreq):
    
    nTrials = np.rint(session_info['nTrials_eachFrequency'][freqInd]).astype(int)
    
    psth = np.zeros((nTrials, nCells, n_tPts))
    gain = np.zeros((nTrials, nCells, n_tPts))
    gain_alt = np.zeros((nTrials, nCells, n_tPts))

    
    # loop over cells
    for cellInd in range(0,nCells):

    
        # freq
        freq = uniqueFreq[freqInd]
    
        # trial indices
        trialInds = session_info['trialInds_eachFrequency'][freqInd]
            
        # single trial psth
        psth[:, cellInd, :] = psth_allTrials[trialInds, cellInd, :]
        
        # trial avg psth
        trialAvg_psth[cellInd, :, freqInd] = np.mean(psth[:, cellInd, :], axis=0)
        
        
        # mean baseline of trial average rate
        trialAvg_baselineRate = np.mean(trialAvg_psth[cellInd, base_tInds])
        
        
        # delta psth
        delta_trialAvg_psth[cellInd, :, freqInd] = trialAvg_psth[cellInd, :, freqInd] - trialAvg_baselineRate
        
        
        # single trial response gain (substract baseline avg of trial avg rate)
        gain[:, cellInd, :] = psth_allTrials[trialInds, cellInd, :] - trialAvg_baselineRate
                        
        # trial average response gain
        trialAvg_gain[cellInd, :, freqInd] = np.mean(gain[:, cellInd, :], axis=0)
        
        
        # single trial response gain alt
        for count, i in enumerate(trialInds):
            
            # baseline rate of this trial
            baseAvg_rate = np.mean(psth_allTrials[i, cellInd, base_tInds])

            # single trial gain
            gain_alt[count, cellInd, :] =  psth_allTrials[i, cellInd, :] - baseAvg_rate
        
        
        # trial average gain alt
        trialAvg_gain_alt[cellInd, :, freqInd] = np.mean( gain_alt[:, cellInd, :], axis=0 )
        
    
        # statistical significance of stimulus response
        
        
        # compare single trial responses in static baseline window to those in static stimulus window
        preStim_psth = psth[:, cellInd, base_sig_tInd].copy()
        postStim_psth = psth[:, cellInd, stim_sig_tInd].copy()
        
        # run statistical test at this time point
        stat_test_data = fcn_Wilcoxon(preStim_psth, postStim_psth)

        pval_preStim_vs_postStim[cellInd, freqInd] = stat_test_data['pVal_2sided']
        
        
        # baseline psth for all trials
        base_psth = psth[:, cellInd, base_tInds].flatten()        

        # loop over stimulus time points
        for _, indT in enumerate(stim_tInds):
                       
            # get psth at this time point
            stim_psth = psth[:, cellInd, indT]
                       
            # if stim and base psth's are the same, continue to next time point
            if ( np.all(stim_psth == 0) and np.all(base_psth == 0) ):
                
                continue
            
            # run statistical test at this time point
            _, pval = fcn_MannWhitney_twoSided(base_psth, stim_psth)


            # store pval
            psth_pval[cellInd, indT, freqInd] = pval
            
        
        
        # statistical significance of baseline response (sanity check)
        # compare N data points at each baseline time point to (M-1)*(N) other baseline time points
        # loop over baseline time points
        for count, indT in enumerate(base_tInds):      
            
            # get psth at this time point
            base_psth_t = psth[:, cellInd, indT]
            
            # make sure not all baseline time points are zero
            if np.all(base_psth == 0):
                
                continue
            
            # run statistical test
            compare_tPts = np.setdiff1d(base_tInds, indT)
            _pval = fcn_MannWhitney_twoSided(base_psth_t, psth[:, cellInd, compare_tPts].flatten())
            
            # store pval
            psth_pval_baseline[cellInd, indT, freqInd] = pval
            
            
# corrected p values for multiple comparisons
psth_pval_corrected = psth_pval*n_stimWindows
psth_pval_baseline_corrected = psth_pval_baseline*np.size(base_tInds)

        
#%% average gain across cells

cellAvg_trialAvg_psth = np.mean(trialAvg_psth, axis=0)
cellAvg_trialAvg_gain = np.mean(trialAvg_gain, axis=0)
cellAvg_trialAvg_gain_alt = np.mean(trialAvg_gain_alt, axis=0)
cellAvg_delta_trialAvg_psth = np.mean(delta_trialAvg_psth, axis=0)



#%% SAVE DATA


params = {'session_path':         data_path, \
          'session_name':         session_name, \
          'baseline_window':      baseline_window, \
          'stimulus_window':      stimulus_window, \
          'trial_window':         trial_window, \
          'window_length':        window_length, \
          'window_step':          window_step}

    
results = {'params':                                params, \
           't_window':                              t_windows, \
           'nTrials_freq':                          session_info['nTrials_eachFrequency'], \
           'nFreq':                                 nFreq, \
           'uniqueFreq':                            uniqueFreq, \
           'trialAvg_psth':                         trialAvg_psth, \
           'trialAvg_gain':                         trialAvg_gain, \
           'trialAvg_gain_alt':                     trialAvg_gain_alt, \
           'delta_trialAvg_psth':                   delta_trialAvg_psth, \
           'cellAvg_trialAvg_psth':                 cellAvg_trialAvg_psth, \
           'cellAvg_delta_trialAvg_psth':           cellAvg_delta_trialAvg_psth, \
           'cellAvg_trialAvg_gain':                 cellAvg_trialAvg_gain, \
           'cellAvg_trialAvg_gain_alt':             cellAvg_trialAvg_gain_alt, \
           'psth_pval':                             psth_pval, \
           'psth_pval_corrected':                   psth_pval_corrected, \
           'n_stimWindows':                         n_stimWindows, \
           'psth_pval_baseline':                    psth_pval_baseline, \
           'psth_pval_baseline_corrected':          psth_pval_baseline_corrected, \
           'pval_preStim_vs_postStim':              pval_preStim_vs_postStim, \
           'preStim_tInd':                          base_sig_tInd, \
           'postStim_tInd':                         stim_sig_tInd, \
           'n_baseWindows':                         len(base_tInds)}
           

        
save_filename = ( (outpath + 'psth_allTrials_%s_windLength%0.3fs%s.mat') % (session_name, window_length, data_name) )      
savemat(save_filename, results) 





