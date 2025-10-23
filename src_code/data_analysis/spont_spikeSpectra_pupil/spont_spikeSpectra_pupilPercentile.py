

#%% imports

# basic imports
import sys     
import numpy as np
import argparse
from scipy.io import savemat

# settings file
import spont_spikeSpectra_settings as settings

# functions
sys.path.append(settings.func_path1)        
sys.path.append(settings.func_path2)

from fcn_processedh5data_to_dict import fcn_processedh5data_to_dict
from fcn_SuData import fcn_makeTrials_spont
from fcn_SuData import fcn_spikeTimes_trials_cells_spont
from fcn_SuData import fcn_compute_spikeCnts_inTrials
from fcn_SuData import fcn_compute_avgPupilSize_inTrials
from fcn_SuData import mt_specpb_lp
from fcn_SuData import raw_specpb_lp
from fcn_SuData import fcn_compute_avgRunSpeed_inTrials
from fcn_SuData import fcn_determine_runningTrials
from fcn_SuData import fcn_pupilPercentile_to_pupilSize
from fcn_SuData import fcn_getTrials_in_pupilRange
from fcn_SuData import fcn_makeTrials
from fcn_SuData import fcn_compute_pupilMeasure_eachTrial

#%% unpack settings
window_length_percentileComputation = settings.window_length_percentileComputation
split_based_on_evokedData = settings.split_based_on_evokedData
window_length = settings.window_length
dt = settings.dt
inter_window_interval = settings.inter_window_interval
df_array = settings.df_array
stim_duration = settings.stim_duration
nTrials_thresh = settings.nTrials_thresh
n_subsamples = settings.n_subsamples
pupilBlock_size = settings.pupilBlock_size
pupilBlock_step = settings.pupilBlock_step
pupilSplit_method = settings.pupilSplit_method
pupilSize_method = settings.pupilSize_method
restOnly = settings.restOnly
trialMatch = settings.trialMatch
runThresh = settings.runThresh
runSpeed_method = settings.runSpeed_method
runBlock_size = settings.runBlock_size
runBlock_step = settings.runBlock_step
runSplit_method = settings.runSplit_method
avg_type = settings.avg_type
data_path = settings.data_path
outpath = settings.outpath


#%% checks

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


#%% parse data by pre-stimulus percentiles computed from 100 ms window
### want to use same percentile splits for all analyses

if split_based_on_evokedData == True:
    
    ### load data and define trials
    
    session_info = fcn_processedh5data_to_dict(session_name, data_path)
    
    nCells = session_info['nCells']
    
    trial_window_evoked = [-window_length_percentileComputation, 100e-3]
    
    ### update session info
    
    session_info['trial_window'] = trial_window_evoked
    
    session_info['pupilSize_method'] = pupilSize_method
    session_info['pupilSplit_method'] = pupilSplit_method
    session_info['pupilBlock_size'] = pupilBlock_size
    session_info['pupilBlock_step'] = pupilBlock_step
    
    session_info['runSpeed_method'] = runSpeed_method
    session_info['runSplit_method'] = runSplit_method
    session_info['runBlock_size'] = runBlock_size
    session_info['runBlock_step'] = runBlock_step
    
    print('session updated')
    
    
    ### make trials
    session_info = fcn_makeTrials(session_info)
    
    ### compute pupil measure in each trial
    session_info = fcn_compute_pupilMeasure_eachTrial(session_info)
    
    ### running info 
    
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
    
    ## if we are only doing resting trials, then set running trials to nan
    if restOnly == True:
        
        # nan out running trials from pupil data
        session_info['trial_pupilMeasure'][run_trials] = np.nan
    
    
    ## pupil size corresponding to each percentile block
    
    # pupil size corresponding ot beginning and end of each pupil percentile bine
    pupilSize_percentileBlocks_evoked = fcn_pupilPercentile_to_pupilSize(session_info)
    
    # pupil sizes corresponding to beginning and end of each pupil block
    pupilBin_centers_evoked = np.mean(pupilSize_percentileBlocks_evoked, 0)
    
    # number of pupil blocks
    n_pupilBlocks = np.size(pupilBin_centers_evoked)
    
    ### delete session dictionary
    del session_info

else:
    
    sys.exit()


#%% analyze spontaneous data

# load data
session_info = fcn_processedh5data_to_dict(session_name, data_path)

# number of cells
nCells = session_info['nCells']

# spontaneous blocks
spont_blocks = session_info['spont_blocks']

# update session dictionary
session_info['spontBlock_start'] = spont_blocks[0,:].copy()
session_info['spontBlock_end'] = spont_blocks[1,:].copy()
session_info['n_spontBlocks'] = np.size(session_info['spontBlock_start'])


#%% make trials

session_info = fcn_makeTrials_spont(session_info, window_length, inter_window_interval)
trial_start = session_info['trial_start'].copy()
trial_end = session_info['trial_end'].copy()


#%% spike times
session_info = fcn_spikeTimes_trials_cells_spont(session_info)


#%% spike counts of all cells in all trials
session_info = fcn_compute_spikeCnts_inTrials(session_info)


#%% average pupil size across window
avg_pupilSize = fcn_compute_avgPupilSize_inTrials(session_info, trial_start, trial_end)
session_info['trial_pupilMeasure'] = avg_pupilSize


#%% running

session_info = fcn_compute_avgRunSpeed_inTrials(session_info, trial_start, trial_end)

session_info = fcn_determine_runningTrials(session_info, runThresh)

# running trials
run_trials = np.nonzero(session_info['running'])[0]
rest_trials = np.nonzero(session_info['running']==0)[0]

if restOnly == True:
    
    # nan out running trials from pupil data
    session_info['trial_pupilMeasure'][run_trials] = np.nan


#%% rebin good trials according to evoked percentiles

pupil_block_trials = np.zeros(n_pupilBlocks, dtype='object')
nTrials_in_pupilBlocks = np.zeros((n_pupilBlocks))

for indPupil in range(0, n_pupilBlocks):

    lowPupil = pupilSize_percentileBlocks_evoked[0, indPupil]
    highPupil = pupilSize_percentileBlocks_evoked[1, indPupil]
    pupil_block_trials[indPupil] = fcn_getTrials_in_pupilRange(session_info, lowPupil, highPupil)    
    nTrials_in_pupilBlocks[indPupil] = np.size( pupil_block_trials[indPupil] )

session_info['pupil_block_trials'] = pupil_block_trials.copy()
        
avg_pupilSize_allTrials_spont = session_info['trial_pupilMeasure'].copy()


#%% number of trials to subsample in each pupil block

nTrials_subsample = np.min(nTrials_in_pupilBlocks).astype(int)
    
if nTrials_subsample < nTrials_thresh:
    print(nTrials_subsample)
    sys.exit('not enough trials')

#%% quantities for spectral analysis

singleTrial_spikeCounts_spont = session_info['spkCounts_trials_cells'].copy()
all_trials_spont = session_info['pupil_block_trials'].copy()
singleTrial_spikeTimes_spont = session_info['spikeTimes_trials_cells'].copy()



#%% clear session info
session_info.clear()


#%% quantities to compute


# spike count bins
bins = np.arange(0, window_length+dt, dt)

# number of frequency points in spectra
nFreq_spectra = int(window_length/(2*dt) + 1)

# number of df
n_df = np.size(df_array)

# spectra of every cell for each frequency in each pupil block
power_spectra = np.ones((nCells, n_pupilBlocks, n_subsamples, nFreq_spectra, n_df, 2))*np.nan
norm_power_spectra = np.ones((nCells, n_pupilBlocks, n_subsamples, nFreq_spectra, n_df, 2))*np.nan

power_spectra_raw = np.ones((nCells, n_pupilBlocks, n_subsamples, nFreq_spectra, 2))*np.nan
norm_power_spectra_raw = np.ones((nCells, n_pupilBlocks, n_subsamples, nFreq_spectra, 2))*np.nan

# spike counts
spont_trialAvg_spikeCount = np.zeros((nCells, n_pupilBlocks))

# average pupil size of low and high pupil trials
avg_pupilSize_trials = np.zeros((n_pupilBlocks, n_subsamples))

# samples
for indSample in range(0, n_subsamples):

    # pupils
    for pupilInd in range(0, n_pupilBlocks):
                
        # trials
        trial_inds_spont = np.random.choice(pupil_block_trials[pupilInd].astype(int), size = int(nTrials_subsample), replace = False).astype(int)
            
        # avg pupil size in trials
        avg_pupilSize_trials[pupilInd, indSample] = np.mean(avg_pupilSize[trial_inds_spont])
        
        # for each cell
        for cellInd in range(0,nCells):

        
            # spontaneous spike counts
            all_trials_spont_spike_counts = singleTrial_spikeCounts_spont[all_trials_spont[pupilInd], cellInd].copy()

            spont_spike_counts = singleTrial_spikeCounts_spont[trial_inds_spont, cellInd].copy()
  
            # trial average spike counts
            spont_trialAvg_spikeCount[cellInd, pupilInd] = np.mean(all_trials_spont_spike_counts)
            
            # spontaneous spike times
            spont_spike_times = singleTrial_spikeTimes_spont[trial_inds_spont, cellInd].copy()
            
            # histogram
            spont_spikes_binned = np.ones((nTrials_subsample, len(bins)-1))*np.nan
            
            for indTrial in range(0, nTrials_subsample):
                
                # align times to beginning of trial
                spont_spike_times[indTrial] = spont_spike_times[indTrial] - trial_start[trial_inds_spont[indTrial]]
                # binning
                spont_spikes_binned[indTrial, :], _ = np.histogram(spont_spike_times[indTrial], bins)
            
            spont_spikes_binned[spont_spikes_binned >= 1] = 1

            # compute spectra
            for ind_df, df in enumerate(df_array):
                
                frequency_spectra, power_spectra[cellInd, pupilInd, indSample, :, ind_df, 0], norm_power_spectra[cellInd, pupilInd, indSample, :, ind_df, 0] = \
                    mt_specpb_lp(spont_spikes_binned, 1/dt, round(window_length*df/2), True, 0)
                    
                frequency_spectra, power_spectra[cellInd, pupilInd, indSample, :, ind_df, 1], norm_power_spectra[cellInd, pupilInd, indSample, :, ind_df, 1] = \
                    mt_specpb_lp(spont_spikes_binned, 1/dt, round(window_length*df/2), True, 1)


                frequency_spectra_raw, power_spectra_raw[cellInd, pupilInd, indSample, :, 0], norm_power_spectra_raw[cellInd, pupilInd, indSample, :, 0] = \
                    raw_specpb_lp(spont_spikes_binned, 1/dt, True, 0)
                    
                frequency_spectra_raw, power_spectra_raw[cellInd, pupilInd, indSample, :, 1], norm_power_spectra_raw[cellInd, pupilInd, indSample, :, 1] = \
                    raw_specpb_lp(spont_spikes_binned, 1/dt, True, 1)


# average
avg_pupilSize_trials = np.mean(avg_pupilSize_trials, 1)

if avg_type == 1:
    power_spectra = np.mean(power_spectra, 2)
    norm_power_spectra = np.mean(norm_power_spectra, 2)
    power_spectra_raw = np.mean(power_spectra_raw, 2)
    norm_power_spectra_raw = np.mean(norm_power_spectra_raw, 2)
elif avg_type == 2:
    power_spectra = np.nanmean(power_spectra, 2)
    norm_power_spectra = np.nanmean(norm_power_spectra, 2)    
    power_spectra_raw = np.nanmean(power_spectra_raw, 2)
    norm_power_spectra_raw = np.nanmean(norm_power_spectra_raw, 2) 


#%% SAVE RESULTS


params = {'session_path':         data_path, \
          'session_name':         session_name, \
          'stim_duration':        stim_duration, \
          'window_length':        window_length, \
          'trial_window_evoked':  trial_window_evoked, \
          'window_length_pctileComp': window_length_percentileComputation, \
          'pupilBlock_size':      pupilBlock_size, \
          'pupilBlock_step':      pupilBlock_step, \
          'pupilSplit_method':    pupilSplit_method, \
          'pupilSize_method':     pupilSize_method, \
          'nSubsamples':          n_subsamples, \
          'restOnly':             restOnly, \
          'runThresh':            runThresh, \
          'trialMatch':           trialMatch, \
          'runSpeed_method':      runSpeed_method, \
          'runBlock_size':        runBlock_size, \
          'runBlock_step':        runBlock_step, \
          'nTrials_thresh':       nTrials_thresh, \
          'dt':                   dt, \
          'df_array':             df_array}


    
results = {'params':                                params, \
           'nTrials_subsample':                     nTrials_subsample, \
           'avg_pupilSize_trials':                  avg_pupilSize_trials, \
           'pupilBin_centers':                      pupilBin_centers_evoked, \
           'pupilSize_percentileBlocks':            pupilSize_percentileBlocks_evoked, \
           'spont_trialAvg_spikeCount':             spont_trialAvg_spikeCount, \
           'frequency_spectra':                             frequency_spectra, \
           'power_spectra':                    power_spectra, \
           'norm_power_spectra':                norm_power_spectra, \
           'frequency_spectra_raw':                  frequency_spectra_raw, \
           'power_spectra_raw':                    power_spectra_raw, \
           'norm_power_spectra_raw':                norm_power_spectra_raw}           
        
if ( (restOnly == True) and (trialMatch == False) ):
    fName_end = 'restOnly'
        
else:
    fName_end = ''
    

save_filename = ( (outpath + 'spont_powerSpectra_pupilPercentile_%s_windLength%0.3fs_%s_raw%d.mat') % (session_name, window_length, fName_end, avg_type) )      
savemat(save_filename, results) 
