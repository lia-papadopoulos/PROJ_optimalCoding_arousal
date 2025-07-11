
#%%

# basic imports
import sys        
import numpy as np
import argparse
from scipy.io import savemat

# decoding parameters
import decoding_params as decode_params

func_path1 = decode_params.func_path1
func_path2 = decode_params.func_path2

# path to functions
sys.path.append(func_path1)        
sys.path.append(func_path2)        

# main functions
from fcn_SuData import fcn_makeTrials
from fcn_SuData import fcn_trialInfo_eachFrequency
from fcn_SuData import fcn_spikeTimes_trials_cells
from fcn_SuData import fcn_decoding_multiclass
from fcn_processedh5data_to_dict import fcn_processedh5data_to_dict


#%% UNPACK DECODING PARAMETERS

data_path = decode_params.data_path
decode_outpath = decode_params.decode_outpath_allTrials
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

session_info = fcn_processedh5data_to_dict(session_name, data_path)

 
#%% update session info

session_info['trial_window'] = trial_window

#%% make trials
session_info = fcn_makeTrials(session_info)


#%% get trials corresponding to each stimulus frequency
session_info = fcn_trialInfo_eachFrequency(session_info)


#%% compute spike times of each cell in every trial
session_info = fcn_spikeTimes_trials_cells(session_info)


#%% run decoding without splitting up based on pupil size

# loop over decoding repetitions
for repInd in range(0, n_decodeReps):

    # number of trials to subsample of each frequency in each pupil block
    nTrials_subsample = int(np.min(session_info['nTrials_eachFrequency']))
    # number of frequencies to decode
    nFreqs_decode = session_info['n_frequencies']
    # unique frequencies
    freqVals = session_info['unique_frequencies']
    
    # trials for decoding (one set per frequency)
    trials_for_decoding = np.zeros(nFreqs_decode, dtype='object')
        
    # loop over stimuli and get trials
    for freqInd in range(0,nFreqs_decode):
        
        # all trials for this frequency
        all_trials = session_info['trialInds_eachFrequency'][freqInd]
                
        # subsample
        trials_for_decoding[freqInd] = np.random.choice(all_trials, \
                                                        size = nTrials_subsample, \
                                                        replace = False) 
            
    # decoding      
    t_decoding_allTrials, accuracy_allTrials, mean_accuracy_shuffle_allTrials, sd_accuracy_shuffle_allTrials, \
    lowPercentile_accuracy_shuffle_allTrials, \
    highPercentile_accuracy_shuffle_allTrials, \
    p_accuracy_allTrials, confusion_mat_allTrials = fcn_decoding_multiclass(session_info, trials_for_decoding, \
                                                                            windSize, windStep, decoderType, \
                                                                                runShuffle, nShuffles, shuffle_percentile, \
                                                                                    crossvalType, lda_solver, nFolds=n_kFolds, nReps=n_kFold_reps)


    #%% SAVE DECODING DATA
         
    parameters_dictionary = {'session_path':            data_path, \
                             'session_name':            session_name, \
                             'windSize':                windSize,\
                             'windStep':                windStep, \
                             'trial_window':            trial_window, \
                             'freqVals':                freqVals, \
                             'nClasses':                nFreqs_decode,\
                             'nTrials_decoding':        nTrials_subsample, \
                             'decoderType':             decoderType, \
                             'crossvalType':            crossvalType, \
                             'runShuffle':              runShuffle, \
                             'nShuffles':               nShuffles, \
                             'n_kFold_reps':            n_kFold_reps, \
                             'n_kFolds':                n_kFolds, \
                             'lda_solver':              lda_solver, \
                             'func_path1':              func_path1, \
                             'func_path2':              func_path2, \
                             'n_decodeReps':            n_decodeReps, \
                             'shuffle_percentile':      shuffle_percentile
    }
          
        
    results_dictionary = {'t_decoding':                             t_decoding_allTrials, \
                          'accuracy':                               accuracy_allTrials, \
                          'confusion_mat':                          confusion_mat_allTrials, \
                          'mean_accuracy_shuffle':                  mean_accuracy_shuffle_allTrials, \
                          'sd_accuracy_shuffle':                    sd_accuracy_shuffle_allTrials, \
                          'lowPercentile_accuracy_shuffle':         lowPercentile_accuracy_shuffle_allTrials,\
                          'highPercentile_accuracy_shuffle':        highPercentile_accuracy_shuffle_allTrials, \
                          'p_accuracy':                             p_accuracy_allTrials,\
                          'params':                                 parameters_dictionary
    }
    
                
    fname_end = (('_allTrials_windSize%0.3fs_windStep%0.3fs_decoder%s_crossVal%s_nFreqs%d') % \
                 (windSize, windStep, decoderType, crossvalType, nFreqs_decode))
    
    save_filename = ( (decode_outpath + 'decode_toneFreq_session%s' + fname_end + '_rep%d.mat') % (session_name, repInd) )
            
    savemat(save_filename, results_dictionary)  
    
    
    
    
    
    
    
    