

import sys
import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
import argparse
import importlib

import dPrime_settings as settings
func_path = settings.func_path

sys.path.append(func_path)
from fcn_compute_firing_stats import fcn_compute_spkCounts
from fcn_compute_firing_stats import Dict2Class

#%% SETUP

# load settings
sim_params_path = settings.sim_params_path
simParams_fname = settings.simParams_fname
net_type = settings.net_type
load_path = settings.load_path
save_path = settings.save_path
sweep_param_name = settings.sweep_param_name

nNetworks = settings.nNetworks
windL = settings.windL
windStep = settings.windStep


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

del params
del s_params

#%% ARGPARSER

parser = argparse.ArgumentParser() 
    
# swept parameter name + value as string
parser.add_argument('-sweep_param_str_val', '--sweep_param_str_val', type=str, required = True)
              

# arguments of parser
args = parser.parse_args()

#-------------------- argparser values for later use -------------------------#

# name of swept parameter with value as a string
sweep_param_str_val = args.sweep_param_str_val


#%% FILENAMES

# beginning of filename
fname_begin = ( '%s%s_%s_sweep_%s_network%d_IC%d_stim%d_stimType_%s_stim_rel_amp%0.3f_' )


#%% LOAD ONE SIMULATION TO SET EVERYTHING UP

# params tuple
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
stimOn = s_params.stim_onset
# get spike counts
_, _, t_window =  fcn_compute_spkCounts(s_params, spikes, 0, windL, windStep)

# align to stimulus
t_window = t_window - stimOn

# number of time windows
Nwindows = np.size(t_window)


#%% INITIALIZE


dprime = np.zeros((Nwindows, N, nStim, nStim, nNetworks))
firing_rate = np.zeros((Nwindows, N, nStim, nNetworks))
stimCells = np.zeros((nStim, nNetworks), dtype='object')

for indNetwork in range(0, nNetworks, 1):
    
    mu_spkCount = np.zeros((Nwindows, N, nStim))
    var_spkCount = np.zeros((Nwindows, N, nStim))
    
    for indStim in range(0,nStim,1):
        
        all_spkCnts = np.zeros((Nwindows, N, nTrials))
        
        for indTrial in range(0,nTrials, 1):
            
            # params tuple
            params_tuple = (load_path, simID, net_type, sweep_param_str_val, indNetwork, indTrial, indStim, stim_shape, stim_rel_amp)

            # filename
            filename = ( (fname_begin + 'simulationData.mat') % params_tuple )
                     
            # load data
            data = loadmat(filename, simplify_cells=True)                
            s_params = Dict2Class(data['sim_params'])
            spikes = data['spikes']
            
            if indTrial == 0:
                stim_Ecells = np.nonzero(s_params.stim_Ecells == 1)[0]
                stim_Icells = np.nonzero(s_params.stim_Icells == 1)[0]
                stimCells[indStim, indNetwork] = np.append(stim_Ecells, stim_Icells)
                    
            # spike counts
            spkCounts_E, spkCounts_I, _ = fcn_compute_spkCounts(s_params, spikes, 0, windL, windStep)       
            all_spkCnts[:, :, indTrial] = np.hstack((spkCounts_E, spkCounts_I))
                                              
          
        # average and variance of spike counts
        mu_spkCount[:, :, indStim] = np.mean(all_spkCnts, 2)
        var_spkCount[:, :, indStim] = np.var(all_spkCnts, 2)
        
        # firing rate
        firing_rate[:, :, indStim, indNetwork] = mu_spkCount[:, :, indStim]/windL


    # compute dprime for each pair of stimuli
    for iStim in range(0,nStim,1):
        for jStim in range(0,nStim,1):
            
            dprime[:, :, iStim, jStim, indNetwork] = \
                np.abs(mu_spkCount[:, :, iStim] - mu_spkCount[:, :, jStim])/( np.sqrt( (1/2)*(var_spkCount[:, :, iStim] + var_spkCount[:, :, jStim]) ) )
            

#%% averaging

freqAvg_dprime = np.nanmean(dprime, axis=(2,3)) # time, neurons, networks

stimAvg_firingRate = np.mean(firing_rate, axis=(2)) # (time, neurons, networks)


#%% SAVE DATA

parameters_dictionary = {'simID':               simID, \
                         'net_type':            net_type, \
                         'nNetworks':           nNetworks, \
                         'nTrials':             nTrials, \
                         'sweep_param_name':    sweep_param_name, \
                         'sweep_param_str_val': sweep_param_str_val, \
                         'nStim':               nStim, \
                         'stim_shape':          stim_shape, \
                         'stim_rel_amp':        stim_rel_amp, \
                         'windL':               windL, \
                         'windStep':            windStep}

    
results_dictionary = {'freqAvg_dprime':                     freqAvg_dprime, \
                      'stimAvg_firingRate':                 stimAvg_firingRate, \
                      't_window':                           t_window, \
                      'stimCells':                          stimCells, \
                      'parameters':                         parameters_dictionary}


save_name = ('%s%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f_singleCell_dPrime_windL%0.3f.mat' % \
             (save_path, simID, net_type, sweep_param_str_val, stim_shape, stim_rel_amp, windL) )
    
savemat(save_name, results_dictionary)

