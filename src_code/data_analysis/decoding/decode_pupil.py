'''
decoding vs pupil
'''

# basic imports
import sys        
import numpy as np
import argparse
from scipy.io import savemat

# decoding parameters
import decoding_params as decode_params

# path to functions
func_path1 = decode_params.func_path1
func_path2 = decode_params.func_path2
sys.path.append(func_path1)        
sys.path.append(func_path2)     

# main functions
from fcn_processedh5data_to_dict import fcn_processedh5data_to_dict
from fcn_SuData import fcn_makeTrials
from fcn_SuData import fcn_trialInfo_eachFrequency
from fcn_SuData import fcn_compute_pupilMeasure_eachTrial
from fcn_SuData import fcn_get_trials_in_pupilBlocks
from fcn_SuData import fcn_trials_perFrequency_perPupilBlock
from fcn_SuData import fcn_spikeTimes_trials_cells
from fcn_SuData import fcn_decoding_multiclass
from fcn_SuData import fcn_compute_avgRunSpeed_inTrials
from fcn_SuData import fcn_determine_runningTrials
from fcn_SuData import fcn_pupilPercentile_to_pupilSize
from fcn_SuData import fcn_trials_perFrequency_perRunBlock


# decoding parameters
import decoding_params as decode_params



#%% PATHS

data_path = decode_params.data_path
decode_outpath = decode_params.decode_outpath_pupil

global_pupilNorm = decode_params.global_pupilNorm
rateDrift_cellSelection = decode_params.rateDrift_cellSelection

trial_window = decode_params.trial_window
windSize = decode_params.windSize
windStep = decode_params.windStep
decoderType = decode_params.decoderType
crossvalType = decode_params.crossvalType
runShuffle = decode_params.runShuffle
n_decodeReps = decode_params.n_decodeReps
n_kFolds = decode_params.n_kFolds
n_kFold_reps = decode_params.n_kFold_reps
nShuffles = decode_params.nShuffles
shuffle_percentile = decode_params.shuffle_percentile

lda_solver = decode_params.lda_solver

pupilSplit_method = decode_params.pupilSplit_method
pupilSize_method = decode_params.pupilSize_method
pupilBlock_size = decode_params.pupilBlock_size
pupilBlock_step = decode_params.pupilBlock_step

run_thresh = decode_params.run_thresh
rest_only = decode_params.rest_only
trialMatch = decode_params.trialMatch
runBlock_size = decode_params.runBlock_size
runBlock_step = decode_params.runBlock_step
runSpeed_method = decode_params.runSpeed_method
runSplit_method = decode_params.runSplit_method

nTrials_thresh = decode_params.nTrials_thresh


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

session_info['pupilSize_method'] = pupilSize_method
session_info['pupilSplit_method'] = pupilSplit_method
session_info['pupilBlock_size'] = pupilBlock_size
session_info['pupilBlock_step'] = pupilBlock_step

session_info['runSpeed_method'] = runSpeed_method
session_info['runSplit_method'] = runSplit_method
session_info['runBlock_size'] = runBlock_size
session_info['runBlock_step'] = runBlock_step


#%% make trials
session_info = fcn_makeTrials(session_info)

#%% get trials corresponding to each stimulus frequency
session_info = fcn_trialInfo_eachFrequency(session_info)

#%% compute spike times of each cell in every trial
session_info = fcn_spikeTimes_trials_cells(session_info)

#%% compute pupil measure in this session
session_info = fcn_compute_pupilMeasure_eachTrial(session_info)


#%% for decoding with running trials
    
# average running speed in each trial
session_info = fcn_compute_avgRunSpeed_inTrials(session_info, session_info['trial_start'], session_info['trial_start'] + np.abs(trial_window[0]) )

# classify trials as running or resting
session_info = fcn_determine_runningTrials(session_info, run_thresh)

# running trials
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

if rest_only == True:
    
    # nan out running trials from pupil data
    session_info['trial_pupilMeasure'][run_trials] = np.nan
    

#%% get trials in each pupil block
session_info = fcn_get_trials_in_pupilBlocks(session_info)


#%% determine number of trials of each stimulus type in each pupil block
session_info = fcn_trials_perFrequency_perPupilBlock(session_info)


#%% get pupil sizes corresponding to beginning and end of each pupil block
pupilSize_percentileBlocks = fcn_pupilPercentile_to_pupilSize(session_info)


#%% get pupil measure in each trial
pupilMeasure_eachTrial = session_info['trial_pupilMeasure']


#%%

n_pupilBlocks = len(session_info['pupil_block_trials'])

# average pupil size of trials used for decoding in each pupil block
avg_pupilSize_decodingTrials_pupilBlocks = np.zeros((n_pupilBlocks, n_decodeReps))
med_pupilSize_decodingTrials_pupilBlocks = np.zeros((n_pupilBlocks, n_decodeReps))


# average pupil size of trials used for decoding in each run block
avg_pupilSize_decodingTrials_runBlocks = np.ones((1, n_decodeReps))*np.nan
med_pupilSize_decodingTrials_runBlocks = np.ones((1, n_decodeReps))*np.nan


#%%

# run decoding

# loop over reps
for repInd in range(0, n_decodeReps):
    
    # number of frequencies to decode
    nFreqs_decode = session_info['n_frequencies']
    
    # unique frequencies
    freqVals = session_info['unique_frequencies']   
   
    # initialize lists
    t_decoding_run = np.ones(1,dtype='object')*np.nan
    accuracy_run = np.ones(1,dtype='object')*np.nan
    mean_accuracy_shuffle_run = np.ones(1,dtype='object')*np.nan
    sd_accuracy_shuffle_run = np.ones(1,dtype='object')*np.nan
    lowPercentile_accuracy_shuffle_run = np.ones(1,dtype='object')*np.nan
    highPercentile_accuracy_shuffle_run = np.ones(1,dtype='object')*np.nan
    p_accuracy_run = np.ones(1,dtype='object')*np.nan
    confusion_mat_run = np.ones(1,dtype='object')*np.nan
    
    
    # peak accuracy
    peak_accuracy_run = np.ones(1)*np.nan
    tInd_peakAccuracy_run = np.ones(1, dtype='uint64')*np.nan
    pctCorrect_runBlock = np.ones((1, nFreqs_decode))*np.nan
        
    
    ###########################################################################
    # if we are matching trials, then execute decoding for running data
    if ( (trialMatch == True) and (rest_only == True) ):
        
        # number of trials to subsample of each frequency in each run block
        nTrials_subsample = np.min([int(session_info['max_nTrials']), int(session_info['max_nTrials_runBlocks'])])
        
        # if we dont have enough running trials, exit here
        if nTrials_subsample <= nTrials_thresh:
            print(nTrials_subsample)
            break
    
        
        # decoding for running data

        # trials for decoding (one set per frequency)
        trials_for_decoding = np.zeros(nFreqs_decode, dtype='object')
        
        # pupil size of trials used for decoding
        pupilSize_decodingTrials = np.zeros(0)
        
        # loop over stimuli and get trials
        for freqInd in range(0,nFreqs_decode):
            
            # all trials for this frequency and run block
            all_trials = session_info['trials_perFreq_perRunBlock'][freqInd, 0]
             
            # subsample
            trials_for_decoding[freqInd] = np.random.choice(all_trials, \
                                                            size = nTrials_subsample, \
                                                            replace = False)
                
            # pupil size of trials used for decoding
            pupilSize_decodingTrials = np.append(pupilSize_decodingTrials, \
                                                 pupilSize_runTrials[trials_for_decoding[freqInd]])
                
        # average pupil size of trials used for decoding in this pupil block
        avg_pupilSize_decodingTrials_runBlocks[0, repInd] = np.mean(pupilSize_decodingTrials)
        med_pupilSize_decodingTrials_runBlocks[0, repInd] = np.median(pupilSize_decodingTrials)


        # decoding
        t_decoding_run[0], accuracy_run[0], mean_accuracy_shuffle_run[0], sd_accuracy_shuffle_run[0],\
        lowPercentile_accuracy_shuffle_run[0], \
        highPercentile_accuracy_shuffle_run[0], \
        p_accuracy_run[0], confusion_mat_run[0] = fcn_decoding_multiclass(session_info, trials_for_decoding, \
                                                                                        windSize, windStep, decoderType, \
                                                                                            runShuffle, nShuffles, shuffle_percentile, \
                                                                                                crossvalType, lda_solver, nFolds=n_kFolds, nReps=n_kFold_reps)
    
        # compute peak accuracy 
        
        peak_accuracy_run[0] = np.max(accuracy_run[0])
        tInd_peakAccuracy_run[0] = int(np.argmax(accuracy_run[0]))
        
        for indFreq in range(0,nFreqs_decode):
            pctCorrect_runBlock[0,indFreq] = confusion_mat_run[0][indFreq, indFreq, int(tInd_peakAccuracy_run[0])]
    
    
        print('session running info computed')
    
    
    ###########################################################################

    ######################## decoding in each pupil block #####################


    # number of pupil blocks
    n_pupilBlocks = len(session_info['pupil_block_trials'])
    
    # number of trials to subsample of each frequency in each pupil block
    if (rest_only and trialMatch):
        nTrials_subsample = np.min([int(session_info['max_nTrials']), int(session_info['max_nTrials_runBlocks'])])

    else:
        nTrials_subsample = int(session_info['max_nTrials'])
        
    
    # initialize lists
    t_decoding = np.zeros(n_pupilBlocks,dtype='object')
    accuracy = np.zeros(n_pupilBlocks,dtype='object')
    mean_accuracy_shuffle = np.zeros(n_pupilBlocks,dtype='object')
    sd_accuracy_shuffle = np.zeros(n_pupilBlocks,dtype='object')
    lowPercentile_accuracy_shuffle = np.zeros(n_pupilBlocks,dtype='object')
    highPercentile_accuracy_shuffle = np.zeros(n_pupilBlocks,dtype='object')
    p_accuracy = np.zeros(n_pupilBlocks,dtype='object')
    confusion_mat = np.zeros(n_pupilBlocks,dtype='object')
    

    
    # peak accuracy
    peak_accuracy = np.zeros(n_pupilBlocks)
    tInd_peakAccuracy = np.zeros(n_pupilBlocks, dtype='uint64')
    pctCorrect_pupilBlock = np.zeros((n_pupilBlocks, nFreqs_decode))
    
    # if we dont have enough running trials, exit here
    if nTrials_subsample <= nTrials_thresh:
        break
    
    # loop over pupil blocks and run decoding for each one
    for blockInd in range(0, n_pupilBlocks, 1):  
        
        # trials for decoding (one set per frequency)
        trials_for_decoding = np.zeros(nFreqs_decode, dtype='object')
        
        # pupil size of trials used for decoding
        pupilSize_decodingTrials = np.zeros(0)
        
        # loop over stimuli and get trials
        for freqInd in range(0,nFreqs_decode):
            
            # all trials for this frequency and pupil block
            all_trials = session_info['trials_perFreq_perPupilBlock'][freqInd, blockInd]
             
            # subsample
            trials_for_decoding[freqInd] = np.random.choice(all_trials, \
                                                            size = nTrials_subsample, \
                                                            replace = False)
                
            # pupil size of trials used for decoding
            pupilSize_decodingTrials = np.append(pupilSize_decodingTrials, \
                                                 pupilMeasure_eachTrial[trials_for_decoding[freqInd]])
                
                
                
        # average pupil size of trials used for decoding in this pupil block
        avg_pupilSize_decodingTrials_pupilBlocks[blockInd,repInd] = np.mean(pupilSize_decodingTrials)
        med_pupilSize_decodingTrials_pupilBlocks[blockInd,repInd] = np.median(pupilSize_decodingTrials)
    
    
        # decoding
        t_decoding[blockInd], accuracy[blockInd], mean_accuracy_shuffle[blockInd], sd_accuracy_shuffle[blockInd], \
        lowPercentile_accuracy_shuffle[blockInd], \
        highPercentile_accuracy_shuffle[blockInd], \
        p_accuracy[blockInd], confusion_mat[blockInd] = fcn_decoding_multiclass(session_info, trials_for_decoding, \
                                                                                    windSize, windStep, decoderType, \
                                                                                        runShuffle, nShuffles, shuffle_percentile, \
                                                                                            crossvalType, lda_solver, nFolds=n_kFolds, nReps=n_kFold_reps)
            
        print(blockInd)
        
        
    # compute peak accuracy for each pupil block
    # compute percent correct for each frequency in each pupil block
    for blockInd in range(0,n_pupilBlocks):
        
        peak_accuracy[blockInd] = np.max(accuracy[blockInd])
        tInd_peakAccuracy[blockInd] = int(np.argmax(accuracy[blockInd]))
        
        for indFreq in range(0,nFreqs_decode):
            pctCorrect_pupilBlock[blockInd,indFreq] = confusion_mat[blockInd][indFreq, indFreq, tInd_peakAccuracy[blockInd]]
    
    
    
    #%% SAVE DECODING DATA
         
    parameters_dictionary = {'session_path':         data_path, \
                             'session_name':         session_name, \
                             'pupilBlock_size':      pupilBlock_size, \
                             'pupilBlock_step':      pupilBlock_step, \
                             'runBlock_size':        runBlock_size, \
                             'runBlock_step':        runBlock_step, \
                             'windSize':             windSize,\
                             'windStep':             windStep, \
                             'pupilSize_method':     pupilSize_method,\
                             'pupilSplit_method':    pupilSplit_method, \
                             'runSpeed_method':      runSpeed_method,\
                             'runSplit_method':      runSplit_method, \
                             'trial_window':         trial_window, \
                             'freqVals':             freqVals, \
                             'nClasses':             nFreqs_decode,\
                             'nTrials_decoding':     nTrials_subsample, \
                             'decoderType':          decoderType, \
                             'lda_solver':           lda_solver, \
                             'crossvalType':         crossvalType, \
                             'runShuffle':           runShuffle, \
                             'nShuffles':            nShuffles, \
                             'n_kFold_reps':         n_kFold_reps, \
                             'n_kFolds':             n_kFolds, \
                             'run_thresh':           run_thresh,\
                             'rest_only':            rest_only, \
                             'func_path1':           func_path1, \
                             'func_path2':           func_path2, 
                             'n_decodeReps':         n_decodeReps, 
                             'shuffle_percentile':   shuffle_percentile
                        
    }
          

        
    results_dictionary = {'t_decoding':                                 t_decoding, \
                          'nTrials_subsample':                          nTrials_subsample, \
                          'accuracy':                                   accuracy, \
                          'confusion_mat':                              confusion_mat, \
                          'mean_accuracy_shuffle':                      mean_accuracy_shuffle, \
                          'sd_accuracy_shuffle':                        sd_accuracy_shuffle, \
                          'lowPercentile_accuracy_shuffle':             lowPercentile_accuracy_shuffle,\
                          'highPercentile_accuracy_shuffle':            highPercentile_accuracy_shuffle, \
                          'p_accuracy':                                 p_accuracy,\
                          'peak_accuracy':                              peak_accuracy, \
                          'pctCorrect_pupilBlock':                      pctCorrect_pupilBlock, \
                          'tInd_peakAccuracy':                          tInd_peakAccuracy, \
                          'pupilSize_percentileBlocks':                 pupilSize_percentileBlocks, \
                          'avg_pupilSize_decodingTrials_pupilBlocks':   avg_pupilSize_decodingTrials_pupilBlocks, \
                          'med_pupilSize_decodingTrials_pupilBlocks':   med_pupilSize_decodingTrials_pupilBlocks, \
                          't_decoding_run':                             t_decoding_run, \
                          'accuracy_run':                               accuracy_run, \
                          'confusion_mat_run':                          confusion_mat_run, \
                          'mean_accuracy_shuffle_run':                  mean_accuracy_shuffle_run, \
                          'sd_accuracy_shuffle_run':                    sd_accuracy_shuffle_run, \
                          'lowPercentile_accuracy_shuffle_run':         lowPercentile_accuracy_shuffle_run,\
                          'highPercentile_accuracy_shuffle_run':        highPercentile_accuracy_shuffle_run, \
                          'p_accuracy_run':                             p_accuracy_run,\
                          'peak_accuracy_run':                          peak_accuracy_run, \
                          'pctCorrect_runBlock':                        pctCorrect_runBlock, \
                          'tInd_peakAccuracy_run':                      tInd_peakAccuracy_run, \
                          'avg_pupilSize_decodingTrials_runBlocks':     avg_pupilSize_decodingTrials_runBlocks, \
                          'med_pupilSize_decodingTrials_runBlocks':     med_pupilSize_decodingTrials_runBlocks, \
                          'params':                                     parameters_dictionary
    }
    
        
    fname_pupilNorm = '' + '_rateDrift_cellSelection'*rateDrift_cellSelection + '_globalPupilNorm'*global_pupilNorm

        
    if (rest_only == True and trialMatch == True):   
        
        fname_end = (('_sweep_pupilSize_pupilSplit%s_pupilBlock%0.3f_pupilStep%0.3f_windSize%0.3fs_windStep%0.3fs_decoder%s_crossVal%s_nFreqs%d_restOnly_trialMatch%s') % \
                     (pupilSplit_method, pupilBlock_size, pupilBlock_step, windSize, windStep, decoderType, crossvalType, nFreqs_decode, fname_pupilNorm))
            
    elif (rest_only == True and trialMatch == False):   
        
        fname_end = (('_sweep_pupilSize_pupilSplit%s_pupilBlock%0.3f_pupilStep%0.3f_windSize%0.3fs_windStep%0.3fs_decoder%s_crossVal%s_nFreqs%d_restOnly%s') % \
                     (pupilSplit_method, pupilBlock_size, pupilBlock_step, windSize, windStep, decoderType, crossvalType, nFreqs_decode, fname_pupilNorm))            
    else:
        
        fname_end = (('_sweep_pupilSize_pupilSplit%s_pupilBlock%0.3f_pupilStep%0.3f_windSize%0.3fs_windStep%0.3fs_decoder%s_crossVal%s_nFreqs%d%s') % \
                     (pupilSplit_method, pupilBlock_size, pupilBlock_step, windSize, windStep, decoderType, crossvalType, nFreqs_decode, fname_pupilNorm))        
        
    
    save_filename = ( (decode_outpath + 'decode_toneFreq_session%s' + fname_end + '_rep%d.mat') % (session_name, repInd) )
            
    savemat(save_filename, results_dictionary)
    
    print(repInd)
    
    
    
    
    
    