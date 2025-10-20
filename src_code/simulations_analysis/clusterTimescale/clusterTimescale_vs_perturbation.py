

#%% imports

import sys
import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
import argparse
import importlib

#%% from settings

import clusterTimescale_vs_perturbation_settings as settings
func_path = settings.func_path

sys.path.append(func_path)
from fcn_compute_firing_stats import Dict2Class
from fcn_compute_firing_stats import fcn_compute_clusterRates_vs_time
from fcn_compute_firing_stats import fcn_compute_time_resolved_rate_gaussian
from fcn_compute_firing_stats import fcn_compute_cluster_activationTimes
from fcn_compute_firing_stats import fcn_compute_avg_clusterTimescale
from fcn_compute_firing_stats import fcn_compute_avg_interactivationTimescale

#%% unpack settings

# load settings
load_from_simParams = settings.load_from_simParams
net_type = settings.net_type
sweep_param_name = settings.sweep_param_name
load_path = settings.load_path
save_path = settings.save_path
nNets = settings.nNetworks
burnTime_begin = settings.burnTime_begin
burnTime_end = settings.burnTime_end
window_step = settings.window_step
window_std = settings.window_std
rate_thresh_array = settings.rate_thresh_array    
gain_based = settings.gain_based

if load_from_simParams:
    sim_params_path = settings.sim_params_path
    simParams_fname = settings.simParams_fname

else:
    simID = settings.simID
    nTrials = settings.nTrials
    nStim = settings.nStim
    stim_shape = settings.stim_shape
    stim_type = settings.stim_type
    stim_rel_amp = settings.stim_rel_amp


#%% load sim parameters

if load_from_simParams:
    
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

#%% argparser

parser = argparse.ArgumentParser() 

parser.add_argument('-sweep_param_str_val', '--sweep_param_str_val', \
                    type=str, required = True)
              
args = parser.parse_args()


#-------------------- argparser values for later use -------------------------#

# name of swept parameter with value as a string
sweep_param_str_val = args.sweep_param_str_val

#%% set filenames

# beginning of filename
fname_begin = ( '%s_%s_sweep_%s' % (simID, net_type, sweep_param_str_val) )

# middle of filename
fname_middle = ( '_network%d_IC%d_stim%d' )

# end of filename
fname_end = ( '_stimType_%s_stim_rel_amp%0.3f_' % (stim_shape, stim_rel_amp) )


#%% number of window lengths and thresholds

n_rateThresh = np.size(rate_thresh_array)

#%% initialize variables for analysis
    
avg_clusterTimescale_E = np.zeros((nNets, nTrials, nStim, n_rateThresh))
avg_clusterIAI_E = np.zeros((nNets, nTrials, nStim, n_rateThresh))


#%% start main analysis block

for net_ind in range(0,nNets,1):
    
    print(net_ind)

    # load one simulation to set up
    
    # filename
    filename = ( (load_path + fname_begin + \
                 fname_middle + fname_end + 'simulationData.mat') % \
                 (net_ind, 0, 0) )
    
    # load data
    data = loadmat(filename, simplify_cells=True)                
    s_params = Dict2Class(data['sim_params'])
    
    spikes = data['spikes']  

    nClus = s_params.p
    to = s_params.T0
    tf = s_params.TF

    # time resolved firing rate of each neuron given window_width, window_step
    tRates, Erates, Irates = fcn_compute_time_resolved_rate_gaussian(s_params, spikes, to, tf, window_std, window_step)  
    
    indBurn_begin = np.argmin(np.abs(tRates - (to + burnTime_begin)))
    indBurn_end = np.argmin(np.abs(tRates - (tf - burnTime_end)))
            
    # update rates
    tRates = tRates[indBurn_begin:indBurn_end]
    nTpts = np.size(tRates)
    
    # initialize cluster rates
    clusterRates_vs_time = np.zeros((nTrials, nStim, nClus, nTpts))


    for trial_ind in range(0,nTrials,1):
                                    
        print('trial_ind %d' % trial_ind)

        for stim_ind in range(0,nStim):
            
            print('stim_ind %d' % stim_ind)
        
            # filename
            filename = ( (load_path + fname_begin + \
                         fname_middle + fname_end + 'simulationData.mat') % \
                         (net_ind, trial_ind, stim_ind) )
            
            # load data
            data = loadmat(filename, simplify_cells=True)                
            s_params = Dict2Class(data['sim_params'])
            
            spikes = data['spikes']  
            
            nClus = s_params.p
            to = s_params.T0
            tf = s_params.TF
            
            clustSize_E = data['popSize_E']
            clustSize_I = data['popSize_I']   
  

            # get cluster assignments
            cluLabels = 0
            cluLabels = np.append(cluLabels, np.cumsum(clustSize_E))
            

            # time resolved firing rate of each neuron given window_width, window_step
            tRates, Erates, Irates = fcn_compute_time_resolved_rate_gaussian(\
                                 s_params, spikes, \
                                 to, tf, window_std, window_step)
        
                
            # time resolved E cluster rates
            Erates_clu = fcn_compute_clusterRates_vs_time(clustSize_E, Erates)
            # remove background rate
            Erates_clu = Erates_clu[:nClus, :]
            
            # get rid of burn time at the beginning and end
            indBurn_begin = np.argmin(np.abs(tRates - (to + burnTime_begin)))
            indBurn_end = np.argmin(np.abs(tRates - (tf - burnTime_end)))
            
            # update rates
            tRates = tRates[indBurn_begin:indBurn_end]
            Erates_clu = Erates_clu[:,indBurn_begin:indBurn_end]
            
            # save time dependnet cluster rates
            clusterRates_vs_time[trial_ind, stim_ind, :, :] = Erates_clu.copy()
            
    # avg cluster rates
    avg_clusterRates = np.mean(clusterRates_vs_time, axis=(0,1,3))
    
    # loop over trials and compute cluster timescale
    for trial_ind in range(0,nTrials,1):
        
        print('trial_ind %d' % trial_ind)
                                    
        for stim_ind in range(0,nStim):
            
            print('stim_ind %d' % stim_ind)
            
            # compute cluster gain
            clusterGain_vs_time = np.zeros((nClus, nTpts))
            
            if gain_based == True:
                for indClu in range(0, nClus):
                    clusterGain_vs_time[indClu, :] = clusterRates_vs_time[trial_ind, stim_ind, indClu, :] - avg_clusterRates[indClu]
            else:
                clusterGain_vs_time[:] = clusterRates_vs_time[trial_ind, stim_ind, :, :].copy()
            

            for iThresh in range(0, n_rateThresh):
                
                print('thresh %d' % iThresh)
                
                # rate threshold
                rate_thresh = rate_thresh_array[iThresh]
            
                # compute cluster activation times
                Eclu_on_off = fcn_compute_cluster_activationTimes(tRates, clusterGain_vs_time, rate_thresh)
                    
                # compute avg cluster activation timescale for each cluster
                Eclu_timescales = np.zeros(nClus)
                
                # compute avg interactivation timescale for each cluster
                Eclu_IAI = np.zeros(nClus)
                
                # loop over clusters
                for cluInd in range(0,nClus,1):
                    
                    Eclu_timescales[cluInd] = fcn_compute_avg_clusterTimescale(Eclu_on_off[cluInd])
                    Eclu_IAI[cluInd] = fcn_compute_avg_interactivationTimescale(Eclu_on_off[cluInd])
                    
                # compute mean across clusters
                # dont average over clusters who never became active (nanmean does this)
                avg_clusterTimescale_E[net_ind, trial_ind, stim_ind, iThresh] = np.nanmean(Eclu_timescales)
               
                avg_clusterIAI_E[net_ind, trial_ind, stim_ind, iThresh] = np.nanmean(Eclu_IAI)

                       
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------                 
                               
# average over trials and stimuli
trialAvg_clusterTimescale_E = np.nanmean(avg_clusterTimescale_E, axis=(1,2))
trialAvg_clusterIAI_E = np.nanmean(avg_clusterIAI_E, axis=(1,2))

# average over networks
netAvg_clusterTimescale_E = np.nanmean(trialAvg_clusterTimescale_E, axis=0)
netStd_clusterTimescale_E = np.nanstd(trialAvg_clusterTimescale_E, axis=0)  

netAvg_clusterIAI_E = np.nanmean(trialAvg_clusterIAI_E, axis=0)
netStd_clusterIAI_E = np.nanstd(trialAvg_clusterIAI_E, axis=0)  

                   
#%% save results
    
results = {}
parameters = {}

parameters['load_path'] = load_path
parameters['save_path'] = save_path
parameters['simID'] = simID
parameters['net_type'] = net_type
parameters['nNets'] = nNets
parameters['nTrials'] = nTrials
parameters['nStim'] = nStim
parameters['stim_shape']  =  stim_shape
parameters['stim_type'] = stim_type
parameters['stim_rel_amp'] = stim_rel_amp
parameters['sweep_param_name'] =    sweep_param_name
parameters['sweep_param_str_val'] = sweep_param_str_val
parameters['burnTime_begin'] = burnTime_begin
parameters['burnTime_end'] = burnTime_end
parameters['window_std'] = window_std
parameters['window_step']  =   window_step
parameters['rate_thresh'] = rate_thresh_array
parameters['gain_based'] = gain_based

results['parameters'] = parameters

results['trialAvg_clusterTimescale_E'] = trialAvg_clusterTimescale_E
results['netAvg_clusterTimescale_E'] = netAvg_clusterTimescale_E
results['netStd_clusterTimescale_E'] = netStd_clusterTimescale_E

results['trialAvg_clusterIAI_E'] = trialAvg_clusterIAI_E
results['netAvg_clusterIAI_E'] = netAvg_clusterIAI_E
results['netStd_clusterIAI_E'] = netStd_clusterIAI_E

if gain_based == False:
    savemat(('%s%s%s_clusterTimescale.mat' % (save_path, fname_begin, fname_end)), results)
else:
    savemat(('%s%s%s_clusterTimescale_gainBased.mat' % (save_path, fname_begin, fname_end)), results)

