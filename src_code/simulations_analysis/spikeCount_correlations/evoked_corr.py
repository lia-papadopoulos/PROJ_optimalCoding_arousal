

#%% basic imports
import sys
import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
import argparse
import importlib

#%% import settings
import evoked_corr_settings as settings

#%% functions
func_path0 = settings.func_path0
func_path1 = settings.func_path1
func_path2 = settings.func_path2
sim_params_path = settings.sim_params_path

sys.path.append(sim_params_path)
sys.path.append(func_path0)
sys.path.append(func_path1)
sys.path.append(func_path2)

from fcn_simulation_loading import fcn_set_sweepParam_string
from fcn_simulation_setup import fcn_define_arousalSweep
from fcn_SuData import fcn_trial_shuffled_spikeCounts
from fcn_compute_firing_stats import fcn_compute_spkCounts
from fcn_compute_firing_stats import Dict2Class
from fcn_statistics import fcn_zscore

#%% unpack settings
simParams_fname = settings.simParams_fname
net_type = settings.net_type
nNetworks = settings.nNetworks
sweep_param_name = settings.sweep_param_name
windL = settings.windL
windStep = settings.windStep
baseWind_burn = settings.baseWind_burn
nCells_sample = settings.nCells_sample
nShuffles = settings.nShuffles
nSamples = settings.nSamples
load_path = settings.load_path
save_path = settings.save_path
decode_path = settings.decode_path
decode_windL = settings.decode_windL
decode_ensembleSize = settings.decode_ensembleSize
decode_rateThresh = settings.decode_rateThresh
decode_classifier = settings.decode_classifier
use_decode_window = settings.use_decode_window

#%% load sim parameters
params = importlib.import_module(simParams_fname) 
s_params = params.sim_params

#%% arousal sweep
s_params = fcn_define_arousalSweep(s_params)

#%% unpack simulation parameters
simID = s_params['simID']
n_sweepParams = s_params['nParams_sweep']
swept_params_dict = s_params['swept_params_dict']
simID = s_params['simID']
nTrials = s_params['n_ICs']
nStim = s_params['nStim']
stim_shape = s_params['stim_shape']
stim_type = s_params['stim_type']
stim_rel_amp = s_params['stim_rel_amp']

del params
del s_params

#%% argparser
parser = argparse.ArgumentParser() 
parser.add_argument('-net_indx', '--net_indx', type=int)    
args = parser.parse_args()
indNetwork = args.net_indx

#%% filenames
fname_begin = ( '%s%s_%s_sweep_%s_network%d_IC%d_stim%d_stimType_%s_stim_rel_amp%0.3f_simulationData.mat')
fname_begin_decode = ( '%s%s_sweep_%s_network%d_windL%dms_ensembleSize%d_rateThresh%0.2fHz_%s.mat') 

#%% load one simulation to set everything up

# number of arousal avlues
nParam_vals = np.size(swept_params_dict['param_vals1'])

# load in one simulation get simulation parameters
sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, 0) 
params_tuple = (load_path, simID, net_type, sweep_param_str, indNetwork, 0, 0, stim_shape, stim_rel_amp)
filename = ( (fname_begin) % (params_tuple) )

# load data
data = loadmat(filename, simplify_cells=True)  

# clustered E cells
Ne_clusters = np.sum(data['popSize_E'][:-1]) 

# sim_params         
s_params = Dict2Class(data['sim_params'])

# spikes
spikes = data['spikes']

# unpack
Ne = s_params.N_e
stimOnset = s_params.stim_onset

# spike counts
spkCounts_E, spkCounts_I, t_window = fcn_compute_spkCounts(s_params, spikes, baseWind_burn, windL, windStep)   

# print
print(t_window)

#%% MAIN ANALYSIS BLOCK

# t eval
tEval_allArousal = np.zeros((nParam_vals))

# average spike count all trials subsample
sampled_cells = np.zeros((nCells_sample, nSamples))
avg_spkCount_allTrials_subsample = np.zeros((nCells_sample, nStim, nSamples))

# correlations
corr_eachStim_eachArousal_subsample = np.zeros((nCells_sample, nCells_sample, nStim, nParam_vals, nSamples))
corr_eachStim_eachArousal_subsample_shuffle = np.zeros((nCells_sample, nCells_sample, nStim, nParam_vals, nSamples, nShuffles))
corr_allTrials_subsample = np.zeros((nCells_sample, nCells_sample, nStim, nSamples))
shuffle_corr_allTrials_subsample = np.zeros((nCells_sample, nCells_sample, nStim, nSamples, nShuffles))

# loop over stimuli and arousal level
for indStim in range(0,nStim,1):

    # spike counts for this frequency
    zscore_spkCounts_allTrials = np.array([])
    zscore_spkCounts_allTrials.shape = (0, Ne)
    spkCounts_allTrials = np.array([])
    spkCounts_allTrials.shape = (0, Ne)
    

    for indParam in range(0, nParam_vals):
    
        # sweep param string
        sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, indParam) 
    
        # load decoding data
        if use_decode_window:
            decode_fname = ( (fname_begin_decode) % (decode_path, simID, sweep_param_str, indNetwork, decode_windL*1000, decode_ensembleSize, decode_rateThresh, decode_classifier) )
            decode_data = loadmat(decode_fname, simplify_cells=True)
            accuracy = decode_data['accuracy'].copy()
            tAccuracy = decode_data['t_window'].copy()
            t_eval = tAccuracy[np.argmax(accuracy)]
        else:
            t_eval = stimOnset + windL
            
        tEval_allArousal[indParam] = t_eval
        # time window index
        tEval_ind = np.nonzero(t_window == t_eval)[0]
        print(t_window[tEval_ind])
    
        # spike counts
        spkCounts_allTrials_singleParam = np.array([])
        spkCounts_allTrials_singleParam.shape = (0, Ne)
                 
        for indTrial in range(0,nTrials, 1):
            
            print(indParam, indStim, indTrial)
            
            params_tuple = (load_path, simID, net_type, sweep_param_str, indNetwork, indTrial, indStim, stim_shape, stim_rel_amp)
            filename = ( (fname_begin) % (params_tuple) )
             
            # load data
            data = loadmat(filename, simplify_cells=True)                
            s_params = Dict2Class(data['sim_params'])
            spikes = data['spikes']
            
            # spike counts
            spkCounts_E, _, _ = fcn_compute_spkCounts(s_params, spikes, baseWind_burn, windL, windStep)  
            
            # add to spike counts for all trials
            spkCounts_allTrials_singleParam = np.vstack((spkCounts_allTrials_singleParam, spkCounts_E[tEval_ind, :]))
            
        # z-score the spike counts for this parameter
        zscore_spkCounts_allTrials_singleParam = np.zeros(np.shape(spkCounts_allTrials_singleParam))
        for indCell in range(0, Ne):
            zscore_spkCounts_allTrials_singleParam[:,indCell] = fcn_zscore(spkCounts_allTrials_singleParam[:, indCell])
    
        zscore_spkCounts_allTrials = np.vstack((zscore_spkCounts_allTrials, zscore_spkCounts_allTrials_singleParam))
        spkCounts_allTrials = np.vstack((spkCounts_allTrials, spkCounts_allTrials_singleParam))
          
        
    for indSample in range(0, nSamples):
        
        # random number generator
        rng = np.random.default_rng(indSample)
        
        # cells to sample [clusters only]
        indCells_sample = rng.choice(Ne_clusters, nCells_sample, replace=False)
        sampled_cells[:, indSample] = indCells_sample.copy()
    
        # subsampled spike counts
        zscore_spkCounts_allTrials_subsampled = zscore_spkCounts_allTrials[:, indCells_sample].copy()
        spkCounts_allTrials_subsampled = spkCounts_allTrials[:, indCells_sample].copy()
        
        avg_spkCount_allTrials_subsample[:, indStim, indSample] = np.mean(spkCounts_allTrials_subsampled, 0)
    
        corr_allTrials_subsample[:,:,indStim,indSample] = np.corrcoef(zscore_spkCounts_allTrials_subsampled, rowvar=False)
        
        # shuffle
        for indShuffle in range(0, nShuffles):
            
            print(indSample, indShuffle)
                            
            zscore_spkCounts_allTrials_subsample_shuffle, _ = fcn_trial_shuffled_spikeCounts(zscore_spkCounts_allTrials_subsampled)
    
            shuffle_corr_allTrials_subsample[:,:,indStim,indSample,indShuffle] = np.corrcoef(zscore_spkCounts_allTrials_subsample_shuffle, rowvar=False)


        # single parameter
        for indParam in range(0, nParam_vals):
                
            startIndx = indParam*nTrials 
            endIndx = startIndx + nTrials
                
            corr_eachStim_eachArousal_subsample[:, :, indStim, indParam, indSample] = np.corrcoef(zscore_spkCounts_allTrials_subsampled[startIndx:endIndx, :], rowvar=False)   
                
            # could also run shuffle for this
            for indShuffle in range(0, nShuffles):
                zscore_spkCounts_allTrials_subsampled_shuffle, _ = fcn_trial_shuffled_spikeCounts(zscore_spkCounts_allTrials_subsampled[startIndx:endIndx, :])
                corr_eachStim_eachArousal_subsample_shuffle[:, :, indStim, indParam, indSample, indShuffle] = np.corrcoef(zscore_spkCounts_allTrials_subsampled_shuffle, rowvar=False)   
                
                                                  
        
# average over stimuli
avg_spkCount_allTrials_subsample = np.mean(avg_spkCount_allTrials_subsample, axis=1)

# correlations
corr_allTrials_subsample = np.nanmean(corr_allTrials_subsample, axis=2)
shuffle_corr_allTrials_subsample = np.nanmean(shuffle_corr_allTrials_subsample, axis=2)
corr_stimAvg_arousalAvg = np.nanmean(corr_eachStim_eachArousal_subsample, axis=(2,3))
corr_stimAvg_arousalAvg_shuffle = np.nanmean(corr_eachStim_eachArousal_subsample_shuffle, axis=(2,3))

#%% SAVE RESULTS

parameters_dictionary = {'simID':               simID, \
                         'net_type':            net_type, \
                         'nNetworks':           nNetworks, \
                         'nTrials':             nTrials, \
                         'sweep_param_name':    sweep_param_name, \
                         'swept_params_dict':  swept_params_dict, \
                         'indNetwork':          indNetwork, \
                         'nStim':               nStim, \
                         'stim_shape':          stim_shape, \
                         'stim_type':           stim_type, \
                         'stim_rel_amp':        stim_rel_amp, \
                         'baseWind_burn':       baseWind_burn, \
                         'windL':               windL, \
                         'windStep':            windStep, \
                          'indCells_sample':    sampled_cells, 
                          'nShuffles':          nShuffles, \
                         'nCells_all':          Ne_clusters, \
                         'decode_path':         decode_path, \
                         'decode_windL':        decode_windL, \
                         'decode_ensembleSize':     decode_ensembleSize, 
                         'decode_rateThresh':       decode_rateThresh, \
                         'decode_classifier':   decode_classifier, \
                         'use_decode_window':   use_decode_window
                         }


    
results_dictionary = {
                      'corr_allTrials_subsample':           corr_allTrials_subsample, \
                      'corr_stimAvg_arousalAvg':           corr_stimAvg_arousalAvg, \
                      'corr_allTrials_subsample_shuffle':           shuffle_corr_allTrials_subsample, \
                      'avg_spkCount_allTrials_subsample':             avg_spkCount_allTrials_subsample, \
                      't_window':                       t_window, \
                      'tEval_allArousal':                   tEval_allArousal, \
                      'parameters':                         parameters_dictionary, \
                      'corr_stimAvg_arousalAvg_shuffle':    corr_stimAvg_arousalAvg_shuffle}


save_name = ('%s%s_%s_sweep_%s_network%d_stimType_%s_stim_rel_amp%0.3f_evoked_corr_windL%0.3f.mat' % \
             (save_path, simID, net_type, sweep_param_name, indNetwork, stim_shape, stim_rel_amp, windL) )
    
savemat(save_name, results_dictionary)
