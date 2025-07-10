
"""
compute rates of selective (targeted by stimulus) and nonselective (not targeted by stimulus) clusters
in the window around peak decoding performance
"""


#%% standard imports
import sys
import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
import importlib


#%% imports from functions folder

sys.path.append('../../')
import global_settings

func_path0 = global_settings.path_to_src_code + 'functions/'
func_path1 = global_settings.path_to_src_code + 'run_simulations/'
                 
sys.path.append(func_path0) 
from fcn_compute_firing_stats import Dict2Class
from fcn_compute_firing_stats import fcn_compute_time_resolved_rate_gaussian
from fcn_compute_firing_stats import fcn_compute_clusterRates_vs_time
from fcn_simulation_loading import fcn_set_sweepParam_string

sys.path.append(func_path1)
from fcn_simulation_setup import fcn_define_arousalSweep


#%% settings

sim_params_path = global_settings.path_to_src_code + 'run_simulations/'
load_path = global_settings.path_to_sim_output + ''
outpath = global_settings.path_to_sim_output + 'deltaRate_selective_nonselective/'
decoding_path = global_settings.path_to_sim_output + 'decoding_analysis/'   


simParams_fname = 'simParams_051325_clu'
sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'
net_type = 'baseEIclu'
nNetworks = 10

# window for computing spike counts
window_length = 100e-3
window_step = 1e-3
window_std = 25e-3

# decoding stuff
decode_ensembleSize = 160
decode_windowSize = 100e-3
decode_type = 'LinearSVC'
rate_thresh = 0.
drawNeurons_str = ('rateThresh%0.2fHz' % rate_thresh)

# filenames
fname_begin = ( '%s_%s_sweep_%s_network%d_IC%d_stim%d_stimType_%s_stim_rel_amp%0.3f_simulationData.mat')
decoding_filename = ('%s_sweep_%s_network%d_windL%dms_ensembleSize%d_%s_%s.mat')


#%% load parameters
sys.path.append(sim_params_path)
params = importlib.import_module(simParams_fname) 
s_params = params.sim_params

#%% arousal sweep
s_params = fcn_define_arousalSweep(s_params)

#%% unpack sim params
simID = s_params['simID']
nTrials = s_params['n_ICs']
nStim = s_params['nStim']
stim_shape = s_params['stim_shape']
stim_rel_amp = s_params['stim_rel_amp']
n_sweepParams = s_params['nParams_sweep']
swept_params_dict = s_params['swept_params_dict']

del s_params
del params


#%% RUN ANALYSIS

#------------------------ setting up -----------------------------------------#

# number of parameters
n_arousalLevels = np.size(swept_params_dict['param_vals1'])
    

#------------------------ initialize -----------------------------------------#
avg_Ecluster_rates_selective = np.zeros((n_arousalLevels, nNetworks, nTrials, nStim))
avg_Ecluster_rates_nonselective = np.zeros((n_arousalLevels, nNetworks, nTrials, nStim))
avg_Ecluster_diff_rates = np.zeros((n_arousalLevels, nNetworks, nTrials, nStim))


#------------------------ main analysis block --------------------------------#

#------------------------- loop over parameters --------------------------#
for param_ind in range(0,n_arousalLevels):

    sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, param_ind) 

#------------------------ loop over networks -----------------------------#
    for net_ind in range(0,nNetworks,1):  
        
        
        # decoding filename
        decode_filename = ( (decoding_path + decoding_filename) % \
                          ( simID, sweep_param_str, net_ind, \
                            decode_windowSize*1000, decode_ensembleSize, drawNeurons_str, decode_type) )
    
        # load decoding data
        decode_data = loadmat(decode_filename, simplify_cells=True)
    
        # time of peak decoding accuracy
        t_decoding = decode_data['t_window']
        accuracy = decode_data['accuracy']
        t_peakAccuracy = t_decoding[np.argmax(accuracy)]    
        
        # stimulation time window
        tfStim = t_peakAccuracy
        toStim = tfStim - window_length
        
#------------------------ loop over initial conditions -------------------#
        for IC_ind in range(0,nTrials,1):
            
#------------------------ loop over stimulation------- -------------------#
            for stimInd in range(0,nStim,1):
                                            
                #------------------------ get key simulation data ------------------------#
            
                # filename
                filename = ((load_path + fname_begin) \
                        % (simID, net_type, sweep_param_str, net_ind, IC_ind, stimInd, stim_shape, stim_rel_amp)) 
                
                # load data
                data = loadmat(filename, simplify_cells=True)                
                
                # sim params
                s_params = Dict2Class(data['sim_params'])
                
                # unpack data
                spikes = data['spikes'] 
                clustSize_E = data['popSize_E']
                clustSize_I = data['popSize_I'] 
                
                # unpack sim params
                nClus = s_params.p
                to = s_params.T0
                tf = s_params.TF
            
                # targeted and non-targeted clusters
                targetedClusters = s_params.selectiveClusters
                non_targetedClusters = np.setdiff1d(np.arange(0,nClus,1),targetedClusters)

                # time resolved firing rate of each neuron given window_width, window_step
                tRates, Erates, Irates = fcn_compute_time_resolved_rate_gaussian(\
                                     s_params, spikes, \
                                     to, tf, window_std, window_step)


                # time resolved E cluster rates
                Erates_clu = fcn_compute_clusterRates_vs_time(clustSize_E, Erates)
            
                # remove background rate
                Erates_clu = Erates_clu[:nClus, :]

                # decoding window time indices
                ind_to_stim = np.argmin(np.abs(tRates - toStim))
                ind_tf_stim = np.argmin(np.abs(tRates - tfStim))
            
                # cluster rates during decoding window
                Erates_clu_stim = Erates_clu[:, ind_to_stim:ind_tf_stim].copy()

                # selective and non selective rates
                avg_Ecluster_rates_selective[param_ind, net_ind, IC_ind, stimInd] = np.mean(Erates_clu_stim[targetedClusters, :])
                
                avg_Ecluster_rates_nonselective[param_ind, net_ind, IC_ind, stimInd] = np.mean(Erates_clu_stim[non_targetedClusters, :])
                
                avg_Ecluster_diff_rates[param_ind, net_ind, IC_ind, stimInd] =  np.mean(Erates_clu_stim[targetedClusters, :]) - np.mean(Erates_clu_stim[non_targetedClusters, :])
                
            print(param_ind, net_ind, IC_ind)



# average across stimuli and trials
rates_selective_trialAvg_stimAvg = np.mean(np.mean(avg_Ecluster_rates_selective,3),2)
rates_nonselective_trialAvg_stimAvg = np.mean(np.mean(avg_Ecluster_rates_nonselective,3),2)
rates_diff_trialAvg_stimAvg = np.mean(np.mean(avg_Ecluster_diff_rates,3),2)

# average and standard deviation across networks
rates_selective_trialAvg_stimAvg_netAvg = np.mean(rates_selective_trialAvg_stimAvg,1)
rates_nonselective_trialAvg_stimAvg_netAvg = np.mean(rates_nonselective_trialAvg_stimAvg,1)
rates_diff_trialAvg_stimAvg_netAvg = np.mean(rates_diff_trialAvg_stimAvg,1)

rates_selective_trialAvg_stimAvg_netSd = np.std(rates_selective_trialAvg_stimAvg,1)
rates_nonselective_trialAvg_stimAvg_netSd = np.std(rates_nonselective_trialAvg_stimAvg,1)
rates_diff_trialAvg_stimAvg_netSd = np.std(rates_diff_trialAvg_stimAvg,1)



#------------------------ save the results ------------------------------------#

parameters_dictionary = {'simID':               simID, \
                         'net_type':            net_type, \
                         'nTrials':             nTrials, \
                         'nNetworks':           nNetworks, \
                         'nStim':               nStim, \
                         'stim_shape':          stim_shape, \
                         'stim_rel_amp':        stim_rel_amp, \
                         'swept_params_dict':    swept_params_dict, \
                         'sweep_param_name':    sweep_param_name, \
                         'n_sweepParams':       n_sweepParams, \
                         'window_length':       window_length, \
                         'window_step':         window_step, \
                         'window_std':          window_std, \
                         'decoding_path':       decoding_path, \
                         'decode_ensembleSize': decode_ensembleSize, \
                         'decode_windowSize':   decode_windowSize, \
                         'decode_type':         decode_type}


results_dictionary = {'rates_selective_trialAvg_stimAvg_netAvg':    rates_selective_trialAvg_stimAvg_netAvg, \
                      'rates_nonselective_trialAvg_stimAvg_netAvg': rates_nonselective_trialAvg_stimAvg_netAvg, \
                      'rates_selective_trialAvg_stimAvg_netSd':     rates_selective_trialAvg_stimAvg_netSd, \
                      'rates_nonselective_trialAvg_stimAvg_netSd':  rates_nonselective_trialAvg_stimAvg_netSd, \
                      'rates_diff_trialAvg_stimAvg_netAvg':          rates_diff_trialAvg_stimAvg_netAvg, \
                      'rates_diff_trialAvg_stimAvg_netSd':          rates_diff_trialAvg_stimAvg_netSd, \
                      'params':                                     parameters_dictionary}


    
save_filename = ( outpath +  '%s_%s_sweep_%s_stimType%s_selective_nonselective_rates.mat' % (simID, net_type, sweep_param_name, stim_shape))

savemat(save_filename, results_dictionary) 

print('data saved')