

#%%

import sys
import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
import argparse
import importlib

import FF_evoked_vs_perturbation_settings as settings
func_path = settings.func_path

sys.path.append(func_path)
from fcn_compute_firing_stats import fcn_compute_spkCounts
from fcn_compute_firing_stats import Dict2Class


#%% unpack settings

load_from_simParams = settings.load_from_simParams
net_type = settings.net_type
load_path = settings.load_path
save_path = settings.save_path
sweep_param_name = settings.sweep_param_name

nNetworks = settings.nNetworks
windL = settings.windL
windStep = settings.windStep

#%% load sim parameters

if load_from_simParams == True:
    sim_params_path = settings.sim_params_path
    simParams_fname = settings.simParams_fname
else:
    simID = settings.simID
    nStim = settings.nStim
    nTrials = settings.nTrials
    stim_shape = settings.stim_shape
    stim_type = settings.stim_type
    stim_rel_amp = settings.stim_rel_amp

#%% load sim parameters

if load_from_simParams == True:

    sys.path.append(sim_params_path)
    params = importlib.import_module(simParams_fname) 
    s_params = params.sim_params
    
    simID = s_params['simID']
    nStim = s_params['nStim']
    nTrials = s_params['n_ICs']
    stim_shape = s_params['stim_shape']
    stim_type = s_params['stim_type']
    stim_rel_amp = s_params['stim_rel_amp']
    
    del params
    del s_params



#%% ARGPARSER

parser = argparse.ArgumentParser() 


# swept parameter name + value as string
parser.add_argument('-sweep_param_str_val', '--sweep_param_str_val', \
                    type=str, default = 'sd_nu_ext_e_pert0.000')
    
# index of swept parameter
parser.add_argument('-param_indx', '--param_indx', type=int)    


# arguments of parser
args = parser.parse_args()


#-------------------- argparser values for later use -------------------------#

# name of swept parameter with value as a string
sweep_param_str_val = args.sweep_param_str_val
# index of swept parameter
inputParam_indx = args.param_indx


#%% LOAD ONE SIMULATION TO SET EVERYTHING UP

# fname begin
fname_begin = ( '%s_%s_sweep_%s' % (simID, net_type, sweep_param_str_val) )

# middle of filename
fname_middle = ( '_network%d_IC%d_stim%d' % (0, 0 ,0))

# end of filename
fname_end = ( '_stimType_%s_stim_rel_amp%0.3f_' % (stim_shape, stim_rel_amp) )

# filename
filename = ( load_path + fname_begin + fname_middle + fname_end + 'simulationData.mat' )

# load data
data = loadmat(filename, simplify_cells=True)   

# sim_params         
s_params = Dict2Class(data['sim_params'])

# spikes
spikes = data['spikes']

# simulation parameters
N = s_params.N

# spike counts
spkCounts_E, spkCounts_I, t_window = fcn_compute_spkCounts(s_params, spikes, 0, windL, windStep)   

# number of time window
n_windows = len(t_window)


#%% COMPUTE SPIKE COUNTS OF EACH CELL ACROSS TIME, FOR EACH TRIAL AND STIMULUS CONDITION

fanofactor = np.ones((N, nStim, nNetworks, n_windows))*np.nan

avg_spkCount = np.ones((N, nStim, nNetworks, n_windows))*np.nan

for indNetwork in range(0, nNetworks, 1):
        
    
    for indStim in range(0,nStim,1):
        
        spkCounts_allTrials = np.zeros((N, nTrials, n_windows))

        for indTrial in range(0,nTrials, 1):
            
            # beginning of filename
            fname_begin = ( '%s_%s_sweep_%s' % (simID, net_type, sweep_param_str_val) )
    
            # middle of filename
            fname_middle = ( '_network%d_IC%d_stim%d' % (indNetwork, indTrial, indStim) )
    
            # end of filename
            fname_end = ( '_stimType_%s_stim_rel_amp%0.3f_' % (stim_shape, stim_rel_amp) )
    
            # filename
            filename = ( load_path + fname_begin + fname_middle  + fname_end + 'simulationData.mat' )
                     
            # load data
            data = loadmat(filename, simplify_cells=True)                
            s_params = Dict2Class(data['sim_params'])
            spikes = data['spikes']
            
            
            # spike counts
            spkCounts_E, spkCounts_I, t_window = fcn_compute_spkCounts(s_params, spikes, 0, windL, windStep)       
            
            # all spike counts
            all_spkCounts = np.vstack(( np.transpose(spkCounts_E), np.transpose(spkCounts_I)  )) # N, windows
            
            spkCounts_allTrials[:, indTrial, :] = all_spkCounts
                                          
      
        fanofactor[:, indStim, indNetwork, :] = np.var(spkCounts_allTrials,1)/np.mean(spkCounts_allTrials,1)

        avg_spkCount[:, indStim, indNetwork, :] = np.mean(spkCounts_allTrials,1)

            

#%% SAVE DATA

parameters_dictionary = {'simID':               simID, \
                         'net_type':            net_type, \
                         'nNetworks':           nNetworks, \
                         'nTrials':             nTrials, \
                         'sweep_param_name':    sweep_param_name, \
                         'sweep_param_str_val': sweep_param_str_val, \
                         'inputParam_indx':     inputParam_indx, \
                         'nStim':               nStim, \
                         'stim_shape':          stim_shape, \
                         'stim_type':           stim_type, \
                         'stim_rel_amp':        stim_rel_amp, \
                         'windL':               windL, \
                         'windStep':            windStep}

    
results_dictionary = {'fanofactor':           fanofactor, \
                      'avg_spkCount':             avg_spkCount, \
                      't_window':                           t_window, \
                      'parameters':                         parameters_dictionary}


save_name = ('%s%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f_FanofactorRaw_timecourse_windL%0.3f.mat' % \
             (save_path, simID, net_type, sweep_param_str_val, stim_shape, stim_rel_amp, windL) )
    
savemat(save_name, results_dictionary)
