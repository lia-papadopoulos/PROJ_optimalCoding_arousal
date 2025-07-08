
"""
compute significance of stimulus responses
"""

# basic imports
import sys
import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
import argparse
import importlib

# import parameters file
import psth_settings as settings
func_path0 = settings.func_path0
func_path1 = settings.func_path1
sim_params_path = settings.sim_params_path

sys.path.append(sim_params_path)
sys.path.append(func_path0)
sys.path.append(func_path1)

from fcn_simulation_loading import fcn_set_sweepParam_string
from fcn_simulation_setup import fcn_define_arousalSweep
from fcn_compute_firing_stats import Dict2Class
from fcn_compute_firing_stats import fcn_compute_spkCounts
from fcn_statistics import fcn_MannWhitney_twoSided
from fcn_statistics import fcn_Wilcoxon


#%% settings

load_path = settings.load_path
save_path = settings.save_path
simParams_fname = settings.simParams_fname
net_type = settings.net_type
sweep_param_name = settings.sweep_param_name

base_window = settings.base_window
stim_window = settings.stim_window
binSize = settings.binSize
stepSize = settings.stepSize
burnTime = settings.burnTime



#%% argparser

parser = argparse.ArgumentParser() 
parser.add_argument('-ind_network', '--ind_network', type=int, default=0)
parser.add_argument('-ind_stim', '--ind_stim', type=int, default=0)

# arguments of parser
args = parser.parse_args()

# argparse parameters
ind_network = args.ind_network
ind_stim = args.ind_stim


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
stim_shape = s_params['stim_shape']
stim_rel_amp = s_params['stim_rel_amp']

del params
del s_params


#%% beginning of filename
fname_begin = ( '%s%s_%s_sweep_%s_network%d_IC%d_stim%d_stimType_%s_stim_rel_amp%0.3f_simulationData.mat')

#%% load one simulation to set everything up

# number of arousal avlues
n_baseMod = np.size(swept_params_dict['param_vals1'])

# load in one simulation get simulation parameters
sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, 0) 
params_tuple = (load_path, simID, net_type, sweep_param_str, ind_network, 0, ind_stim, stim_shape, stim_rel_amp)
filename = ( (fname_begin) % (params_tuple) )

# load data
data = loadmat(filename, simplify_cells=True)                
s_params = Dict2Class(data['sim_params'])
spikes = data['spikes']

# get relevant parameters
N_e = s_params.N_e
N_i = s_params.N_i
Tf = s_params.TF
To = s_params.T0
nClus = s_params.p
stimOnset = s_params.stim_onset

# example psth
spkCounts_E, spkCounts_I, t_window = fcn_compute_spkCounts(s_params, spikes, burnTime, binSize, stepSize)
n_bins = len(t_window)

evoked_bins = np.nonzero( (t_window <= stim_window[1] + stimOnset)  & (t_window > stim_window[0] + stimOnset) )[0]
base_bins = np.nonzero( (t_window <= base_window[1] + stimOnset)  & (t_window > base_window[0] + stimOnset) )[0]

# pre and post stim window
preStim_wind = np.argmin(np.abs(t_window - stimOnset))
postStim_wind = np.argmin(np.abs(t_window - (stimOnset + binSize)))



print(preStim_wind)
print(postStim_wind)
print(t_window[preStim_wind])
print(t_window[postStim_wind])
print(base_bins)
print(evoked_bins)

print(len(base_bins))
print(len(evoked_bins))


# initialize all quantities
singleTrial_psth = np.zeros((nTrials, N_e + N_i, n_bins, n_baseMod))
singleTrial_gain = np.zeros((nTrials, N_e + N_i, n_bins, n_baseMod))

psth_pval_allBaseMod = np.ones((N_e + N_i, n_bins))*np.inf
psth_pval_eachBaseMod = np.ones((N_e + N_i, n_bins, n_baseMod))*np.inf
pval_preStim_vs_postStim_allBaseMod = np.ones((N_e + N_i))*np.inf



#------------------------ main analysis block --------------------------------#
        

# loop over baseline modulation values
for ind_baseMod in range(0, n_baseMod):
        
    # loop over trials    
    for ind_trial in range(0,nTrials,1):
                          
        # filename
        sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, ind_baseMod) 
        fname = ( '%s%s_%s_sweep_%s_network%d_IC%d_stim%d_stimType_%s_stim_rel_amp%0.3f_simulationData.mat')
        params_tuple = (load_path, simID, net_type, sweep_param_str, ind_network, ind_trial, ind_stim, stim_shape, stim_rel_amp)     
        filename = ( (fname) %  (params_tuple) )            

        # load data
        data = loadmat(filename, simplify_cells=True)                
        s_params = Dict2Class(data['sim_params'])
        spikes = data['spikes'] 
        stimOnset = s_params.stim_onset
        
 
        # spike times
        spike_times = spikes[0,:]
                
        # neuron IDs
        neuron_IDs = spikes[1,:].astype(int)
        
                
        # compute rate of each cell vs time
        spkCounts_E, spkCounts_I, _ = fcn_compute_spkCounts(s_params, spikes, burnTime, binSize, stepSize)
        rateE_bins = np.transpose(spkCounts_E/binSize)
        rateI_bins = np.transpose(spkCounts_I/binSize)
        
        # single trial psth and gain
        singleTrial_psth[ind_trial, :, :, ind_baseMod] = np.vstack((rateE_bins, rateI_bins))
        
        for indCell in range((N_e + N_i)):
        
            singleTrial_gain[ind_trial, indCell, :, ind_baseMod] = singleTrial_psth[ind_trial, indCell, :, ind_baseMod] - np.mean(singleTrial_psth[ind_trial, indCell, base_bins, ind_baseMod])
            
            
        print(ind_baseMod, ind_trial)
            
        
#-----------------------------------------------------------------------------#

# statistical signficance of stimulus response for each cell

# combine data across all baseline modulations
          
# baseline psth for all trials, time points, baseline modulations

for indCell in range((N_e + N_i)):

    base_psth =  singleTrial_psth[:, indCell, base_bins, :].flatten()
    
    # loop over stimulus time points
    for _, indT in enumerate(evoked_bins):
                       
        # get psth at this time point
        stim_psth = singleTrial_psth[:, indCell, indT, :].flatten()
                       
        # if stim and base psth's are the same, continue to next time point
        if ( np.all(stim_psth == 0) and np.all(base_psth == 0) ):
                
            continue
            
        # run statistical test at this time point
        _, pval = fcn_MannWhitney_twoSided(base_psth, stim_psth)

        # store pval
        psth_pval_allBaseMod[indCell, indT] = pval
        
        print(indCell, indT)
        
        

    # compare single trial responses in static baseline window to those in static stimulus window
    preStim_psth = singleTrial_psth[:, indCell, preStim_wind, :].flatten()
    postStim_psth = singleTrial_psth[:, indCell, postStim_wind, :].flatten()
        
    # run statistical test at this time point
    stat_test_data = fcn_Wilcoxon(preStim_psth, postStim_psth)
    pval_preStim_vs_postStim_allBaseMod[indCell] = stat_test_data['pVal_2sided']
        
# corrected p values for multiple comparisons
psth_pval_corrected = psth_pval_allBaseMod*np.size(evoked_bins)       
     

#-----------------------------------------------------------------------------#

# statistical signficance of stimulus response for each cell

# consider each baseline modulation separately
        
for indCell in range((N_e + N_i)):
    
    for ind_baseMod in range(0, n_baseMod):

        base_psth =  singleTrial_psth[:, indCell, base_bins, ind_baseMod].flatten()
        
        # loop over stimulus time points
        for _, indT in enumerate(evoked_bins):
                           
            # get psth at this time point
            stim_psth = singleTrial_psth[:, indCell, indT, ind_baseMod].flatten()
                           
            # if stim and base psth's are the same, continue to next time point
            if ( np.all(stim_psth == 0) and np.all(base_psth == 0) ):
                    
                continue
                
            # run statistical test at this time point
            _, pval = fcn_MannWhitney_twoSided(base_psth, stim_psth)
    
            # store pval
            psth_pval_eachBaseMod[indCell, indT, ind_baseMod] = pval
            
            print(indCell, ind_baseMod, indT)
        
        
# corrected p values for multiple comparisons
psth_pval_eachBaseMod_corrected = psth_pval_eachBaseMod*np.size(evoked_bins)   
        
        
        
        
#------------------------ save the results ------------------------------------#
        
        
# trial average psth and gain

trialAvg_psth_eachBaseMod = np.mean(singleTrial_psth, 0)
trialAvg_gain_eachBaseMod = np.mean(singleTrial_gain, 0)
    
trialAvg_psth_allBaseMod = np.mean(singleTrial_psth, axis = (0, 3) )
trialAvg_gain_allBaseMod = np.mean(singleTrial_gain, axis = (0, 3) )
   
trialVar_psth_eachBaseMod = np.var(singleTrial_psth, 0)
trialVar_gain_eachBaseMod = np.var(singleTrial_gain, 0)
    
trialVar_psth_allBaseMod = np.var(singleTrial_psth, axis = (0, 3) )
trialVar_gain_allBaseMod = np.var(singleTrial_gain, axis = (0, 3) )     
        
        
#------------------------ save the results ------------------------------------#

parameters_dictionary = {'simID':               simID, \
                         'net_type':            net_type, \
                         'nTrials':             nTrials, \
                         'stim_shape':          stim_shape, \
                         'stim_rel_amp':        stim_rel_amp, \
                         'base_window':         base_window, \
                         'stim_window':         stim_window, \
                         'binSize':             binSize, \
                         'stepSize':            stepSize, \
                         'sweep_param_name':    sweep_param_name, \
                         'n_sweepParams':        n_sweepParams, \
                         'swept_params_dict':   swept_params_dict}
    
    
results_dictionary = {
                      'bin_times':                                          t_window - stimOnset, \
                      'preStim_wind':                                       preStim_wind, \
                      'postStim_wind':                                      postStim_wind, \
                      'pval_preStim_vs_postStim_allBaseMod':                pval_preStim_vs_postStim_allBaseMod, \
                      'psth_pval_corrected':                                psth_pval_corrected, \
                      'psth_pval_eachBaseMod_corrected':                    psth_pval_eachBaseMod_corrected, \
                      'trialAvg_psth_eachBaseMod':                          trialAvg_psth_eachBaseMod, \
                      'trialAvg_gain_eachBaseMod':                          trialAvg_gain_eachBaseMod, \
                      'trialAvg_psth_allBaseMod':                           trialAvg_psth_allBaseMod, \
                      'trialAvg_gain_allBaseMod':                           trialAvg_gain_allBaseMod, \
                      'trialVar_psth_eachBaseMod':                          trialVar_psth_eachBaseMod, \
                      'trialVar_gain_eachBaseMod':                          trialVar_gain_eachBaseMod, \
                      'trialVar_psth_allBaseMod':                           trialVar_psth_allBaseMod, \
                      'trialVar_gain_allBaseMod':                           trialVar_gain_allBaseMod, \
                      'parameters':                                         parameters_dictionary}


# save data
savename_begin = ( '%s%s_%s_sweep_%s_network%d_stim%d_stimType_%s_stim_rel_amp%0.3f_psth_windSize%0.3fs.mat' )
params_tuple = ( save_path, simID, net_type, sweep_param_name, ind_network, ind_stim, stim_shape, stim_rel_amp, binSize)
save_filename = ( (savename_begin) % (params_tuple) )
savemat(save_filename, results_dictionary) 

print('data saved')
