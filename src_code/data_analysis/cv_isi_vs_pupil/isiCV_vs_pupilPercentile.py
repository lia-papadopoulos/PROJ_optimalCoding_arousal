
#%% basic imports
import sys     
import numpy as np
import numpy.matlib
import argparse
from scipy.io import savemat

#%% settings
import isiCV_vs_pupilPercentile_settings as settings

#%% functions
sys.path.append(settings.func_path1)        
from fcn_processedh5data_to_dict import fcn_processedh5data_to_dict
from fcn_SuData import fcn_makeTrials
from fcn_SuData import fcn_trialInfo_eachFrequency
from fcn_SuData import fcn_compute_pupilMeasure_eachTrial
from fcn_SuData import fcn_compute_avgRunSpeed_inTrials
from fcn_SuData import fcn_determine_runningTrials
from fcn_SuData import fcn_makeTrials_spont
from fcn_SuData import fcn_spikeTimes_trials_cells_spont
from fcn_SuData import fcn_compute_spikeCnts_inTrials
from fcn_SuData import fcn_compute_avgPupilSize_inTrials
from fcn_SuData import fcn_pupilPercentile_to_pupilSize
from fcn_SuData import fcn_getTrials_in_pupilRange
from fcn_SuData import fcn_get_trials_in_pupilBlocks


#%% unpack settings
data_path = settings.data_path
outpath = settings.outpath
bins_from_evokedTrials = settings.bins_from_evokedTrials
window_length_percentileComputation = settings.window_length_percentileComputation
window_length = settings.window_length
window_step = settings.window_step
inter_window_interval = settings.inter_window_interval
stim_duration = settings.stim_duration
pupilBlock_size = settings.pupilBlock_size
pupilBlock_step = settings.pupilBlock_step
pupilSplit_method = settings.pupilSplit_method
global_pupilNorm = settings.global_pupilNorm
cellSelection = settings.cellSelection
highDownSample = settings.highDownsample
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

#%% load data
data_name = '' + cellSelection + '_globalPupilNorm'*global_pupilNorm + '_downSampled'*highDownSample


#%% parse data by pre-stimulus percentiles computed from 100 ms window
### want to use same percentile splits for all analyses

# make session dictionary
session_info = fcn_processedh5data_to_dict(session_name, data_path, fname_end = data_name)

# pupil blocks
if bins_from_evokedTrials == True:


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
    
    
    ### make trials
    session_info = fcn_makeTrials(session_info)
    
    ###trials of each frequency
    session_info = fcn_trialInfo_eachFrequency(session_info)
    
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
    
    
    # pupil size corresponding ot beginning and end of each pupil percentile bine
    pupilSize_percentileBlocks = fcn_pupilPercentile_to_pupilSize(session_info)
    
    # pupil sizes corresponding to beginning and end of each pupil block
    pupilBin_centers = np.mean(pupilSize_percentileBlocks, 0)
    
    # number of pupil blocks
    n_pupilBlocks = np.size(pupilBin_centers)
    
    ### clear
    del session_info


#%% re-make session dictionary
session_info = fcn_processedh5data_to_dict(session_name, data_path)
nCells = session_info['nCells']


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

avg_pupilSize = fcn_compute_avgPupilSize_inTrials(session_info, trial_start, trial_end)
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
 
    
    
#%% bin trials according to precomputed percentile bins

if bins_from_evokedTrials == True:

    pupil_block_trials = np.zeros(n_pupilBlocks, dtype='object')
    nTrials_in_pupilBlocks = np.zeros((n_pupilBlocks))
    
    
    for indPupil in range(0, n_pupilBlocks):
    
        lowPupil = pupilSize_percentileBlocks[0, indPupil]
        highPupil = pupilSize_percentileBlocks[1, indPupil]
        
        pupil_block_trials[indPupil] = fcn_getTrials_in_pupilRange(session_info, lowPupil, highPupil)
        
        nTrials_in_pupilBlocks[indPupil] = np.size( pupil_block_trials[indPupil] )
    
    
    session_info['pupil_block_trials'] = pupil_block_trials.copy()
    
else:
    
    session_info = fcn_get_trials_in_pupilBlocks(session_info)
    
    # pupil size corresponding ot beginning and end of each pupil percentile bine
    pupilSize_percentileBlocks = fcn_pupilPercentile_to_pupilSize(session_info)
    
    # pupil sizes corresponding to beginning and end of each pupil block
    pupilBin_centers = np.mean(pupilSize_percentileBlocks, 0)

    
#%% number of trials in each pupil block

n_pupilBlocks = np.size(session_info['pupil_block_trials'] )

nTrials_in_pupilBlocks = np.zeros(n_pupilBlocks)

for indPupil in range(0, n_pupilBlocks):
    
    nTrials_in_pupilBlocks[indPupil] = np.size(session_info['pupil_block_trials'][indPupil])
    
    
#%% number of trials to subsample in each pupil block
nTrials_subsample_spont = np.min(nTrials_in_pupilBlocks)
nTrials_subsample_spont = int(nTrials_subsample_spont)

#%% average pupil size of spont trials in each block
avg_pupilSize_allTrials_spont = session_info['trial_pupilMeasure'].copy()


#%% quantities for isi
all_trials_spont = session_info['pupil_block_trials'].copy()
singleTrial_spikeTimes_spont = session_info['spikeTimes_trials_cells'].copy()
singleTrial_spikeCounts_spont = session_info['spkCounts_trials_cells'].copy()


#%% clear session info
del session_info

#%% number of trials to subsample

nTrials_subsample = nTrials_subsample_spont
    
if nTrials_subsample < nTrials_thresh:
    print(nTrials_subsample)
    sys.exit('not enough trials')



#%% quantities to compute

spont_cv_isi = np.ones((nCells, n_pupilBlocks, n_subsamples, nTrials_subsample))*np.nan
spont_cv_isi_trialAggregate = np.ones((nCells, n_pupilBlocks, n_subsamples))*np.nan
spont_trialAvg_spikeCount = np.ones((nCells, n_pupilBlocks))*np.nan
avg_pupilSize_spontTrials = np.ones((n_pupilBlocks, n_subsamples))*np.nan

#%%

for indSample in range(0, n_subsamples):

    for pupilInd in range(0, n_pupilBlocks):
                
        # sample trials
        trial_inds_spont = np.random.choice(all_trials_spont[pupilInd], size = nTrials_subsample, replace = False).astype(int)
        
        # average pupil size of sampled trials
        avg_pupilSize_spontTrials[pupilInd, indSample] = np.mean(avg_pupilSize_allTrials_spont[trial_inds_spont])

        for cellInd in range(0,nCells):
            
            # spontaneous spike counts
            all_trials_spont_spike_counts = singleTrial_spikeCounts_spont[all_trials_spont[pupilInd], cellInd].copy()
            spont_spike_counts = singleTrial_spikeCounts_spont[trial_inds_spont, cellInd].copy()
            
            # trial average spike counts
            spont_trialAvg_spikeCount[cellInd, pupilInd] = np.mean(all_trials_spont_spike_counts)
            
            # isi
            isi_all_trials = np.array([])
            
            # loop over trials
            for trialCount, indTrial in enumerate(trial_inds_spont):
                
                spk_times = singleTrial_spikeTimes_spont[indTrial, cellInd].copy()
                isi = np.diff(spk_times)
                isi_all_trials = np.append(isi_all_trials, isi)
                
                # cv2 
                spont_cv_isi[cellInd, pupilInd, indSample, trialCount] = np.std(isi)/(np.mean(isi))

            
            spont_cv_isi_trialAggregate[cellInd, pupilInd, indSample] = np.std(isi_all_trials)/(np.mean(isi_all_trials))
                

#%% averaging

spont_cv_isi = np.nanmean(spont_cv_isi, axis = (2, 3))
spont_cv_isi_trialAggregate = np.nanmean(spont_cv_isi_trialAggregate, axis=(2))
avg_pupilSize_spontTrials = np.mean(avg_pupilSize_spontTrials, 1)


#%% SAVE DATA


params = {'session_path':         data_path, \
          'session_name':         session_name, \
          'stim_duration':        stim_duration, \
          'window_length':        window_length, \
          'pupilBlock_size':      pupilBlock_size, \
          'pupilBlock_step':      pupilBlock_step, \
          'pupilSplit_method':    pupilSplit_method, \
          'restOnly':             restOnly, \
          'runThresh':            runThresh, \
          'runBlock_size':        runBlock_size, \
          'runBlock_step':        runBlock_step, \
          'trialMatch':           trialMatch, \
          'nTrials_thresh':       nTrials_thresh, \
          'window_step':          window_step, \
          'window_length_pctileComp': window_length_percentileComputation}

    
    
results = {'params':                                params, \
           'nTrials_subsample':                     nTrials_subsample, \
           'spont_cv_isi':                      spont_cv_isi, \
           'spont_cv_isi_trialAggreate':           spont_cv_isi_trialAggregate, \
           'spont_trialAvg_spikeCount':             spont_trialAvg_spikeCount, \
           'avg_pupilSize_spontTrials':                    avg_pupilSize_spontTrials, \
           'pupilBin_centers':                pupilBin_centers, \
           'pupilSize_percentileBlocks':            pupilSize_percentileBlocks}
    

        

if ( (restOnly == True) and (trialMatch == False) ):
    fName_end = 'restOnly'
        
else:
    fName_end = ''
    

save_filename = ( (outpath + 'spont_cvISI_pupilPercentile_raw_%s_windLength%0.3fs_%s%s.mat') % (session_name, window_length, fName_end, data_name) )      
savemat(save_filename, results) 


