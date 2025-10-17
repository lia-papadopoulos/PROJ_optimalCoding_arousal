
'''
This script generates
    Fig7A
'''


#%% standard imports
import sys
import numpy as np
import os
from scipy.io import loadmat
import numpy.matlib
import importlib

#%% import global settings file
sys.path.append('../../../')
import global_settings

#%% plotting
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')

#%% imports from functions folder

func_path0 = global_settings.path_to_src_code + 'functions/'
func_path1 = global_settings.path_to_src_code + 'run_simulations/'
                 
sys.path.append(func_path0) 
from fcn_compute_firing_stats import Dict2Class
from fcn_compute_firing_stats import fcn_compute_time_resolved_rate_gaussian
from fcn_compute_firing_stats import fcn_compute_clusterRates_vs_time
from fcn_compute_firing_stats import fcn_compute_clusterActivation
from fcn_simulation_loading import fcn_set_sweepParam_string

sys.path.append(func_path1)
from fcn_simulation_setup import fcn_define_arousalSweep


#%% settings

fig_path = global_settings.path_to_manuscript_figs_final + 'active_inactive_cluster_rasters/'
sim_params_path = global_settings.path_to_src_code + 'run_simulations/'
load_path = global_settings.path_to_sim_output

simParams_fname = 'simParams_051325_clu'
sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'
net_type = 'baseEIclu'
nNetworks = 10
nTrials = 30
nStim = 5

# arousal values to plot
arousalPlot = [0, 0.5, 1.]

# trials to plot
trialsPlot_low = [10, 11]
trialsPlot_mid = [5, 8]
trialsPlot_high = [1, 14]

# number of clusters
nClus = 18

# analysis parameters [all times in seconds]
preStim_burn = 200e-3
window_length = 100e-3
window_step = 1e-3
window_std = 25e-3
gain_thresh = 0
rate_thresh = 15
lifetimeThresh = 25e-3
active_rate_array = np.array([50, 25, 14])
inactive_rate_array = np.array([1, 3, 11])

# decoding stuff
decoding_path = global_settings.path_to_sim_output + 'decoding_analysis/'   
decode_ensembleSize = 160
decode_windowSize = 100e-3
decode_type = 'LinearSVC'
rate_thresh = 0.
drawNeurons_str = ('rateThresh%0.2fHz' % rate_thresh)

# filenames
fname_begin = ( '%s_%s_sweep_%s_network%d_IC%d_stim%d_stimType_%s_stim_rel_amp%0.3f_simulationData.mat')
decoding_filename = ('%s_sweep_%s_network%d_windL%dms_ensembleSize%d_%s_%s.mat')

# figure ids
figID_low = 'Fig7A_low'
figID_mid = 'Fig7A_mid'
figID_high = 'Fig7A_high'


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
arousal_level = s_params['arousal_levels']

del s_params
del params

#%% figure path

# make output directory
if os.path.isdir(fig_path) == False:
    os.makedirs(fig_path)

#%% find arousal indices to plot

arousalInds = np.array([], dtype=int)
for i in arousalPlot:
    ind = np.argmin(np.abs(arousal_level - i))
    arousalInds = np.append(arousalInds, ind)


#%% start main analysis blocks

for count, indParam in enumerate(arousalInds):
        
    # swept parameters string
    sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, indParam) 

    # active and inactive rates
    active_rate = active_rate_array[count]
    inactive_rate = inactive_rate_array[count]
   
    # networks
    for net_ind in range(0,1,1):
                
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
        tfStim = t_peakAccuracy
        toStim = t_peakAccuracy - 1.5*decode_windowSize
        
        # initialize baseline rate
        baselineRate = np.zeros((nTrials, nStim, nClus))
    
        # trials
        for trial_ind in range(0,nTrials,1):
    
            # stimuli
            for stim_ind in range(0,nStim):
                        
                # filename
                filename = ((load_path + fname_begin) % (simID, net_type, sweep_param_str, net_ind, trial_ind, stim_ind, stim_shape, stim_rel_amp)) 
                
                # load data
                data = loadmat(filename, simplify_cells=True)                
                
                # sim params
                s_params = Dict2Class(data['sim_params'])
                
                # spikes
                spikes = data['spikes']  
                
                # unpack sim params
                nClus = s_params.p
                to = s_params.T0
                tf = s_params.TF
                tStim = s_params.stim_onset
                
                # get cluster assignments
                clustSize_E = data['popSize_E']
                clustSize_I = data['popSize_I']   

                # baseline time window
                toBase = to + preStim_burn
                tfBase = tStim - window_length
                
                # time resolved firing rate of each neuron given window_width, window_step
                tRates, Erates, Irates = fcn_compute_time_resolved_rate_gaussian(\
                                         s_params, spikes, \
                                         to, tf, window_std, window_step)
                    
                # baseline window time indices
                inds_t_base = np.nonzero( (tRates <= tfBase) & ( tRates >= toBase))[0]
                
                # time resolved E cluster rates
                Erates_clu = fcn_compute_clusterRates_vs_time(clustSize_E, Erates)
                
                # remove background rate
                Erates_clu = Erates_clu[:nClus, :]
        
                # baseline rates
                Erates_clu_base = Erates_clu[:, inds_t_base].copy()
                baselineRate[trial_ind, stim_ind, :] = np.mean(Erates_clu_base, 1)
    
                print(indParam, trial_ind, stim_ind)
        
        
        # trial avg baseline rate
        trialAvg_baselineRate = np.mean(baselineRate, axis=(0,1))
    
    
        # start plotting
        for stim_ind in range(0,1):
            
            # trials    
            if count == 0:
                trialInds_plot = trialsPlot_low
                figID = figID_low
            elif count == 1:
                trialInds_plot = trialsPlot_mid
                figID = figID_mid
            elif count == 2:
                trialInds_plot = trialsPlot_high
                figID = figID_high
            
            
            for trial_count, trial_ind in enumerate(trialInds_plot):
        
                # filename
                filename = ((load_path + fname_begin) % (simID, net_type, sweep_param_str, net_ind, trial_ind, stim_ind, stim_shape, stim_rel_amp)) 
                
                # load data
                data = loadmat(filename, simplify_cells=True) 
                
                # parameters
                s_params = Dict2Class(data['sim_params'])
                
                # spikes
                spikes = data['spikes']  
                
                # unpack sim params
                nClus = s_params.p
                to = s_params.T0
                tf = s_params.TF
                tStim = s_params.stim_onset
                
                # get cluster assignments
                clustSize_E = data['popSize_E']
                clustSize_I = data['popSize_I']   
                
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
                
                # gain
                Egain_clu = np.zeros(np.shape(Erates_clu))
                for indClu in range(0, nClus):
                    Egain_clu[indClu,:] = Erates_clu[indClu, :] - trialAvg_baselineRate[indClu]
                
                
                
                #### gain based activation
                
                # decoding window
                inds_t_stim = np.nonzero( (tRates <= tfStim) & ( tRates >= toStim))[0]
                                

                ind_peakAccuracy_f = np.argmin(np.abs(tRates[inds_t_stim] - t_peakAccuracy))
                ind_peakAccuracy_i = np.argmin(np.abs(tRates[inds_t_stim] - ( t_peakAccuracy - decode_windowSize)))
                print(tRates[inds_t_stim][ind_peakAccuracy_f])
                
                ind_stimOnset = np.nonzero(tRates[inds_t_stim] == tStim)[0]
                print(tRates[inds_t_stim][ind_stimOnset])

                # rates for schematic
                rates_schematic = np.ones((nClus, len(tRates))) * inactive_rate
                
                # loop over clusters and fill in rates array
                indClu = -1
                for clu in non_targetedClusters:
                    indClu += 1
                    clu_binarized = fcn_compute_clusterActivation(Egain_clu[clu,:], gain_thresh)
                    tInds_active = np.nonzero(clu_binarized == 1)[0]
                    rates_schematic[indClu, tInds_active] = active_rate
                for clu in targetedClusters:
                    indClu += 1
                    clu_binarized = fcn_compute_clusterActivation(Egain_clu[clu,:], gain_thresh)
                    tInds_active = np.nonzero(clu_binarized == 1)[0]
                    rates_schematic[indClu, tInds_active] = active_rate
                        
                # just extract decoding window
                rates_schematic = rates_schematic[:, inds_t_stim]
                        
                # plot
                plt.rcParams.update({'font.size': 8})
                fig = plt.figure(figsize=((0.95,1.13)))  
                ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 
                ax.imshow(rates_schematic, cmap='Greys', aspect='auto', vmin=0, vmax=50, origin='lower', interpolation='nearest')
                ax.plot([0, len(inds_t_stim)-1], [8.5, 8.5], linestyle='dashed', color='k', linewidth=0.75)
                ax.plot([ind_peakAccuracy_i, ind_peakAccuracy_i], [-0.5, 17.5], color='darkorange', linewidth=0.75)
                tickLocs = np.arange(0, len(inds_t_stim), 20)
                tickLabels = np.round(tRates[inds_t_stim[tickLocs]],2)
                plt.xticks(tickLocs, labels=tickLabels)
                plt.xticks([])
                plt.yticks([])
                plt.savefig( (('%s%s%s.pdf') % (fig_path, figID, trial_count) ), bbox_inches='tight', pad_inches=0, transparent=True)
                plt.close()                

                
