

import sys
import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
import argparse
import importlib

import decode_settings as settings
func_path = settings.func_path

sys.path.append(func_path)
from fcn_compute_firing_stats import fcn_compute_spkCounts
from fcn_compute_firing_stats import Dict2Class
from fcn_decoding import fcn_repeated_stratified_kFold_crossVal
from fcn_make_network_cluster import fcn_compute_cluster_assignments
from fcn_decoding import fcn_draw_neurons


#%% SETUP

# load settings
load_from_simParams = settings.load_from_simParams
net_type = settings.net_type
load_path = settings.load_path
save_path = settings.save_path

burnTime = settings.burnTime
windStep = settings.windStep
classifier= settings.classifier
lda_solver = settings.lda_solver
nFolds = settings.nFolds
nReps = settings.nReps
compute_shuffleDist = settings.compute_shuffleDist
nShuffles = settings.nShuffles
sig_level = settings.sig_level
shuffle_percentile = settings.shuffle_percentile
saveName_short = settings.saveName_short
rate_thresh = settings.rate_thresh
nSamples = settings.nSamples
drawStimNeurons = settings.drawStimNeurons
seed = settings.seed
draw_equalPerCluster = settings.draw_equalPerCluster
testing = settings.testing

if load_from_simParams == True:
    sim_params_path = settings.sim_params_path
    simParams_fname = settings.simParams_fname
else:
    simID = settings.simID
    nTrials = settings.nTrials
    stim_shape = settings.stim_shape
    stim_type = settings.stim_type
    stim_rel_amp = settings.stim_rel_amp
    n_sweepParams = settings.n_sweepParams
    swept_params_dict = settings.swept_params_dict


#%% load sim parameters

if load_from_simParams == True:

    sys.path.append(sim_params_path)
    params = importlib.import_module(simParams_fname) 
    s_params = params.sim_params
    
    simID = s_params['simID']
    nTrials = s_params['n_ICs']
    stim_shape = s_params['stim_shape']
    stim_type = s_params['stim_type']
    stim_rel_amp = s_params['stim_rel_amp']
    
    del params
    del s_params


#%% SET PARAMETERS THAT CAN BE PASSED IN
    
parser = argparse.ArgumentParser() 


# swept parameter name
parser.add_argument('-sweep_param_name', '--sweep_param_name', \
                    type=str, required=True)
    
# swept parameter name + value as string
parser.add_argument('-sweep_param_str_val', '--sweep_param_str_val', \
                    type=str, required=True)
    
    
# window length
parser.add_argument('-windL', '--windL', type=float, required=True)

# ensemble size
parser.add_argument('-ensembleSize', '--ensembleSize', type=int, required=True)
                
# network ID
parser.add_argument('-ind_network', '--ind_network', type=int, required=True)

# arguments of parser
args = parser.parse_args()


#%% GET ARG PARSER VARIABLES FOR LATER USE

# name of swept parameter
sweep_param_name = args.sweep_param_name
# name of swept parameter with value as a string
sweep_param_str_val = args.sweep_param_str_val
# network index
ind_network = args.ind_network
# window length
windL = args.windL
# ensemble size
ensembleSize = args.ensembleSize

#%% SET FILENAMES


if sweep_param_name == 'stim_rel_amp':
    fname = ( '%s%s_%s_sweep_%s_network%d_IC%d_stim%d_simulationData.mat')
    params_tuple = (load_path, simID, net_type, sweep_param_str_val, ind_network, 0, 0)

else:
    if stim_type == 'stim_noStim':
        fname = ( '%s%s_%s_sweep_%s_network%d_IC%d_stim%d_stimType_%s_stim_rel_amp%0.3f_%s_simulationData.mat')
        params_tuple = (load_path, simID, net_type, sweep_param_str_val, ind_network, 0, 0, stim_shape, stim_rel_amp, stim_type)

    else:
        fname = ( '%s%s_%s_sweep_%s_network%d_IC%d_stim%d_stimType_%s_stim_rel_amp%0.3f_simulationData.mat')
        params_tuple = (load_path, simID, net_type, sweep_param_str_val, ind_network, 0, 0, stim_shape, stim_rel_amp)
        
filename = ( (fname) %  (params_tuple) )


#%% LOAD ONE SIMULATION TO SET EVERYTHING UP

# load data
data = loadmat(filename, simplify_cells=True)   
# cluster sizes

if 'popSize_E' in data:
    clustSize_E = data['popSize_E']
else:
    clustSize_E = data['clust_sizeE']
    

if 'popSize_I' in data:
    clustSize_I = data['popSize_I']
else:
    clustSize_I = data['clust_sizeI']
    

# sim_params         
s_params = Dict2Class(data['sim_params'])
# spikes
spikes = data['spikes']
# simulation parameters
N = s_params.N
Ne = s_params.N_e
nClu = s_params.p
stimOn = s_params.stim_onset
stimOff = stimOn + s_params.stim_duration
nStim = s_params.nStim
# get spike counts
_, _, t_window =  fcn_compute_spkCounts(s_params, spikes, burnTime, windL, windStep)
# number of time windows
Nwindows = np.size(t_window)

# baseline windows
baseWindow = np.nonzero( (t_window >= burnTime) & (t_window <= stimOn) )[0]

#%% COMPUTE SPIKE COUNTS OF EACH CELL ACROSS TIME, FOR EACH TRIAL AND STIMULUS
#   CONDITION

# all 
all_spkCnts_E = np.zeros((Nwindows, Ne, nTrials, nStim))

all_stimCells = np.array([])

Ecluster_inds, _ = fcn_compute_cluster_assignments(clustSize_E, clustSize_I)


for indTrial in range(0,nTrials,1):
    
    for indStim in range(0,nStim,1):
        
        if sweep_param_name == 'stim_rel_amp':
            params_tuple = (load_path, simID, net_type, sweep_param_str_val, ind_network, indTrial, indStim)

        else:
            if stim_type == 'stim_noStim':
                params_tuple = (load_path, simID, net_type, sweep_param_str_val, ind_network, indTrial, indStim, stim_shape, stim_rel_amp, stim_type)

            else:
                params_tuple = (load_path, simID, net_type, sweep_param_str_val, ind_network, indTrial, indStim, stim_shape, stim_rel_amp)
        
        # filename
        filename = ( (fname) %  (params_tuple) )
                 
        # load data
        data = loadmat(filename, simplify_cells=True)                
        s_params = Dict2Class(data['sim_params'])
        spikes = data['spikes']
        
        if indTrial == 0:
            stimCells = np.nonzero(s_params.stim_Ecells == 1)[0]
            all_stimCells = np.append(all_stimCells, stimCells)
        
        # spike counts
        spkCounts_E, _, _ = fcn_compute_spkCounts(s_params, spikes, burnTime, windL, windStep)       
        all_spkCnts_E[:, :, indTrial, indStim] = spkCounts_E
        
# cells with good firing rate
trialAvg_rate = np.mean(all_spkCnts_E, axis=(2,3))/windL
avg_baselineRate = np.mean(trialAvg_rate[baseWindow, :], 0)
goodRate_cells = np.nonzero(avg_baselineRate >= rate_thresh)[0]

all_stimCells = np.unique(all_stimCells).astype(int)
all_stimCells = np.intersect1d(goodRate_cells, all_stimCells)            


#%% DECODING

# initialize arrays for decoding analysis
accuracy_samples = np.zeros((Nwindows, nSamples))
mean_accuracy_shuffle_samples = np.zeros((Nwindows, nSamples))
sd_accuracy_shuffle_samples = np.zeros((Nwindows, nSamples))
lowPercentile_accuracy_shuffle_samples = np.zeros((Nwindows, nSamples))
highPercentile_accuracy_shuffle_samples = np.zeros((Nwindows, nSamples))
p_accuracy_samples =  np.zeros((Nwindows, nSamples))
#confusion_mat_samples = np.zeros((nStim,nStim,Nwindows, nSamples))

subsampled_cells = np.zeros((ensembleSize, nSamples))

#---------------LOOP OVER RANDOM DRAWS OF NEURONS -----------------------------#
for indSample in range(0, nSamples):

    # subsample from excitatory cells
    if drawStimNeurons:
        ensembleInds = fcn_draw_neurons(all_stimCells, Ecluster_inds, ensembleSize, seed, draw_equalPerCluster)
        
    else:
        ensembleInds = fcn_draw_neurons(goodRate_cells, Ecluster_inds, ensembleSize, seed, draw_equalPerCluster)
        
    
    ensemble_spkCnts_E = all_spkCnts_E[:, ensembleInds, :, :].copy()

    subsampled_cells[:, indSample] = ensembleInds


    #---------------LOOP ACROSS TIME WINDOWS---------------------------------------#
    for tInd in range(0,Nwindows,1):
        
        #---------------SETUP FOR DECODING-----------------------------------------#
        
        # initialize array to hold spike count data for all trials, stim conditions, 
        # and cells in the ensemble
        X = np.zeros((nTrials*nStim, ensembleSize))
        
        # initialize vector to hold the stimulus label for each trial
        classLabels = np.zeros((nTrials*nStim))
            
        # loop over trials and stimulus conditions and build data/label arrays
        for indTrial in range(0,nTrials,1):
            
            for indStim in range(0,nStim,1):
                
                # row index for this trial/stim condition
                rowInd = nTrials*indStim + indTrial
                
                # data [vector of size nTrials*nStim x ensembleSize]
                X[rowInd,:] = ensemble_spkCnts_E[tInd,:,indTrial, indStim]
                
                # class (stim condition) label
                classLabels[rowInd] = indStim    
                
                
        #---------------RUN DECODING-------------------------------------------# 
        accuracy_samples[tInd, indSample], \
        mean_accuracy_shuffle_samples[tInd, indSample], sd_accuracy_shuffle_samples[tInd, indSample], \
        lowPercentile_accuracy_shuffle_samples[tInd, indSample], \
        highPercentile_accuracy_shuffle_samples[tInd, indSample], \
        p_accuracy_samples[tInd, indSample], _ = \
            fcn_repeated_stratified_kFold_crossVal(X, classLabels, classifier, \
                                                   nFolds, nReps, \
                                                   compute_shuffleDist, nShuffles, \
                                                   shuffle_percentile, lda_solver)

#%% AVERAGE OVER SAMPLES

accuracy = np.mean(accuracy_samples, 1)
mean_accuracy_shuffle = np.mean(mean_accuracy_shuffle_samples, 1)
sd_accuracy_shuffle = np.mean(sd_accuracy_shuffle_samples, 1)
lowPercentile_accuracy_shuffle = np.mean(lowPercentile_accuracy_shuffle_samples, 1)
highPercentile_accuracy_shuffle = np.mean(highPercentile_accuracy_shuffle_samples, 1)
p_accuracy = np.mean(p_accuracy_samples, 1)

    
#%% SAVE DATA

parameters_dictionary = {'simID':               simID, \
                         'net_type':            net_type, \
                         'ind_network':         ind_network, \
                         'nTrials':             nTrials, \
                         'nStim':               nStim, \
                         'stim_shape':          stim_shape, \
                         'stim_type':           stim_type, \
                         'stim_rel_amp':        stim_rel_amp, \
                         'burnTime':            burnTime, \
                         'windL':               windL, \
                         'windStep':            windStep, \
                         'classifier':          classifier, \
                         'lda_solver':          lda_solver,\
                         'compute_shuffleDist': compute_shuffleDist, \
                         'nFolds':              nFolds, \
                         'nReps':               nReps, \
                         'ensembleSize':        ensembleSize, \
                         'subsampledCells':     subsampled_cells, \
                         'nShuffles':           nShuffles, \
                         'sig_level':           sig_level, \
                         'shuffle_percentile':  shuffle_percentile, \
                         'stimOn':              stimOn, \
                         'stimOff':             stimOff, \
                         'seed':                seed, \
                         'nSamples':            nSamples}
    
results_dictionary = {'t_window':                               t_window, \
                      'accuracy':                               accuracy, \
                      'mean_accuracy_shuffle':                  mean_accuracy_shuffle, \
                      'lowPercentile_accuracy_shuffle':         lowPercentile_accuracy_shuffle,\
                      'highPercentile_accuracy_shuffle':        highPercentile_accuracy_shuffle, \
                      'p_accuracy':                             p_accuracy, \
                      'accuracy_samples':                       accuracy_samples, \
                      'mean_accuracy_shuffle_samples':          mean_accuracy_shuffle_samples, \
                      'sd_accuracy_shuffle_samples':            sd_accuracy_shuffle_samples, \
                      'lowPercentile_accuracy_shuffle_samples': lowPercentile_accuracy_shuffle_samples,\
                      'highPercentile_accuracy_shuffle_samples':highPercentile_accuracy_shuffle_samples, \
                      'p_accuracy_samples':                     p_accuracy_samples, \
                      'parameters':                             parameters_dictionary}



if drawStimNeurons:
    drawNeurons_str = ('_stimCellsOnly_rateThresh%0.2fHz' % rate_thresh)
else:
    drawNeurons_str = ('_rateThresh%0.2fHz' % rate_thresh)
        
if sweep_param_name == 'stim_rel_amp':
    fname = ( '%s%s_%s_sweep_%s_network%d_windL%dms_ensembleSize%d%s_%s_decoding.mat')
    params_tuple = (save_path, simID, net_type, sweep_param_str_val, ind_network, windL*1000, ensembleSize, drawNeurons_str, classifier)

else:
    if stim_type == 'stim_noStim':
        fname = ( '%s%s_%s_sweep_%s_network%d_stimType_%s_stim_rel_amp%0.3f_%s_windL%dms_ensembleSize%d%s_%s_decoding.mat')
        params_tuple = (save_path, simID, net_type, sweep_param_str_val, ind_network, stim_shape, stim_rel_amp, stim_type, windL*1000, ensembleSize, drawNeurons_str, classifier)

    else:
        if saveName_short:
            if testing:
                fname = ( '%s%s_sweep_%s_network%d_windL%dms_ensembleSize%d%s_%s_TESTING.mat')
                params_tuple = (save_path, simID, sweep_param_str_val, ind_network, windL*1000, ensembleSize, drawNeurons_str, classifier)
            else:
                fname = ( '%s%s_sweep_%s_network%d_windL%dms_ensembleSize%d%s_%s.mat')
                params_tuple = (save_path, simID, sweep_param_str_val, ind_network, windL*1000, ensembleSize, drawNeurons_str, classifier)

        else:
            fname = ( '%s%s_%s_sweep_%s_network%d_stimType_%s_stim_rel_amp%0.3f_windL%dms_ensembleSize%d%s_%s_decoding.mat')
            params_tuple = (save_path, simID, net_type, sweep_param_str_val, ind_network, stim_shape, stim_rel_amp, windL*1000, ensembleSize, drawNeurons_str, classifier)
        
save_filename = ( (fname) % (params_tuple) )
savemat(save_filename, results_dictionary)
        


