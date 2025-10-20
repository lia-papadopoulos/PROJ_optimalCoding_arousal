

#%% imports
import sys
import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
import argparse
import importlib

#%% settings
import spont_cvISI_vsPerturbation_settings as settings

#%% functions
func_path = settings.func_path
sys.path.append(func_path)
from fcn_compute_firing_stats import fcn_compute_total_spkCount
from fcn_compute_firing_stats import Dict2Class

#%% unpack settings
sim_params_path = settings.sim_params_path
simParams_fname = settings.simParams_fname
net_type = settings.net_type
load_path = settings.load_path
save_path = settings.save_path
nNetworks = settings.nNetworks
sweep_param_name = settings.sweep_param_name
windL = settings.windL

#%% load sim parameters
sys.path.append(sim_params_path)
params = importlib.import_module(simParams_fname) 
s_params = params.sim_params
simID = s_params['simID']
nTrials = s_params['n_ICs']
nStim = s_params['nStim']
stim_shape = s_params['stim_shape']
stim_type = s_params['stim_type']
stim_rel_amp = s_params['stim_rel_amp']

#%% argparser

# initialize
parser = argparse.ArgumentParser() 

# swept parameter name + value as string
parser.add_argument('-sweep_param_str_val', '--sweep_param_str_val', type=str, required = True)
    
# arguments of parser
args = parser.parse_args()

#-------------------- argparser values for later use -------------------------#

# name of swept parameter with value as a string
sweep_param_str_val = args.sweep_param_str_val


#%% filename
fname_begin = ( '%s%s_%s_sweep_%s_network%d_IC%d_stim%d_stimType_%s_stim_rel_amp%0.3f_' )

#%% load one simulation to set everything up

# parameters
params_tuple = (load_path, simID, net_type, sweep_param_str_val, 0, 0, 0, stim_shape, stim_rel_amp)

# filename
filename = ( (fname_begin + 'simulationData.mat') % params_tuple )

# load data
data = loadmat(filename, simplify_cells=True)   

# sim_params         
s_params = Dict2Class(data['sim_params'])

# spikes
spikes = data['spikes']

# simulation parameters
N = s_params.N
Ne = s_params.N_e

#%% MAIN ANALYSIS BLOCK

# spike counts
avg_spkCount = np.zeros((N, nNetworks, nStim))

# cvISI_eachTrial of every cell for each frequency in each pupil block
cvISI_eachTrial = np.ones((N, nNetworks, nStim, nTrials))*np.nan
cvISI_trialAggregate = np.ones((N, nNetworks, nStim))*np.nan

# loop over networks, stimuli, trials
for indNetwork in range(0, nNetworks, 1):
        
    for indStim in range(0,nStim,1):
        
        spkCounts_allTrials = np.zeros((N, nTrials))
        
        diff_spkTimes_allTrials = np.ones((N), dtype='object')*np.nan
        
        for indTrial in range(0,nTrials, 1):
            
            # fname begin
            params_tuple = (load_path, simID, net_type, sweep_param_str_val, indNetwork, indTrial, indStim, stim_shape, stim_rel_amp)

    
            # filename
            filename = ( (fname_begin + 'simulationData.mat') % params_tuple )
                     
            # load data
            data = loadmat(filename, simplify_cells=True)                
            s_params = Dict2Class(data['sim_params'])
            spikes = data['spikes']
            
            # start and end of analysis window
            window_start = s_params.TF - windL
            window_end = s_params.TF
            
            # get spikes in window
            spikeTimes = spikes[0,:].copy()
            window_inds = np.nonzero( (spikeTimes <= window_end) & (spikeTimes >= window_start))[0]
            spikes = spikes[:, window_inds]
            
            # spike counts
            spkCounts = fcn_compute_total_spkCount(s_params, spikes)       

            # save
            spkCounts_allTrials[:, indTrial] = spkCounts

                                          
            # for each cell
            for indCell in range(0, N):
                
                cell_spikeInds = np.nonzero(spikes[1,:] == indCell)[0]
                cell_spikeTimes_raw = spikes[0,cell_spikeInds].copy()
                cell_spikeTimes = cell_spikeTimes_raw - window_start
 
 
                # compute single trial cv isis
                diff_spkTimes = np.diff(cell_spikeTimes)
                cvISI_eachTrial[indCell, indNetwork, indStim, indTrial] = np.std(diff_spkTimes)/np.mean(diff_spkTimes)
                
                # add to running list across all trials
                diff_spkTimes_allTrials[indCell] = np.append(diff_spkTimes_allTrials[indCell], diff_spkTimes)


        # avg spike count
        avg_spkCount[:, indNetwork, indStim] = np.mean(spkCounts_allTrials,1)
        
        # compute trial aggregate cv ISI
        for indCell in range(0, N):
            
            cvISI_trialAggregate[indCell, indNetwork, indStim] = np.nanstd(diff_spkTimes_allTrials[indCell])/np.nanmean(diff_spkTimes_allTrials[indCell])
                

# average over trials
avg_cvISI_eachTrial = np.nanmean(cvISI_eachTrial, 3)
            
 
#%% save the results

parameters_dictionary = {'load_path':           load_path, \
                         'save_path':           save_path, \
                         'func_path':           func_path, \
                         'simID':               simID, \
                         'net_type':            net_type, \
                         'nNetworks':           nNetworks, \
                         'nTrials':             nTrials, \
                         'sweep_param_name':    sweep_param_name, \
                         'sweep_param_str_val': sweep_param_str_val, \
                         'nStim':               nStim, \
                         'stim_shape':          stim_shape, \
                         'stim_type':           stim_type, \
                         'stim_rel_amp':        stim_rel_amp, \
                         'windL':               windL, \
                         'Ne':                  Ne, \
                         'N':                   N}

results_dictionary = {'avg_spkCount':             avg_spkCount, \
                      'avg_cvISI_eachTrial':      avg_cvISI_eachTrial, \
                      'cvISI_trialAggregate':     cvISI_trialAggregate, \
                      'parameters':               parameters_dictionary}


save_name = ('%s%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f_spont_cvISI_windL%0.3f.mat' % \
                (save_path, simID, net_type, sweep_param_str_val, stim_shape, stim_rel_amp, windL) )
    
savemat(save_name, results_dictionary)
