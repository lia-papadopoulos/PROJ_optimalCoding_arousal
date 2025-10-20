
#%% imports
import numpy as np
import scipy.stats
import numpy.matlib
from scipy.io import savemat
from scipy.io import loadmat
import sys
import argparse
import importlib

#%% settings
import settings as params

#%% functions
func_path = params.func_path
sys.path.append(func_path)    
from fcn_compute_firing_stats import Dict2Class
from fcn_compute_firing_stats import fcn_compute_clusterRates_vs_time
from fcn_compute_firing_stats import fcn_compute_time_resolved_rate_gaussian
from fcn_compute_firing_stats import fcn_compute_num_activeClusters
from fcn_compute_firing_stats import fcn_compute_cluster_activation_lifetimes

#%% settings

# paths for loading and saving data
sim_params_path = params.sim_params_path
load_path = params.load_path
save_path = params.save_path
sim_params_path = params.sim_params_path
simParams_fname = params.simParams_fname
net_type = params.net_type
nNets = params.nNetworks

# analysis parameters
zscore = params.zscore
preStim_burn = params.preStim_burn
window_length = params.window_length
window_step = params.window_step
window_std = params.window_std
gain_thresh_array = params.gain_thresh_array
lifetimeThresh = params.lifetimeThresh

# decoding 
decoding_path = params.decoding_path
decode_ensembleSize = params.decode_ensembleSize
decode_windowSize = params.decode_windowSize
decode_type = params.decode_type
decode_rateThresh = params.decode_rateThresh

#%% load sim parameters

sys.path.append(sim_params_path)
sim_params = importlib.import_module(simParams_fname) 
s_params = sim_params.sim_params

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
parser.add_argument('-sweep_param_str_val', '--sweep_param_str_val', type=str)
    
# arguments of parser
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

# decoding filename
decoding_filename = ('%s_sweep_%s_network%d_windL%dms_ensembleSize%d_rateThresh%0.2fHz_%s.mat')


#%% more setup

n_gainThresh = np.size(gain_thresh_array)

# filename
filename = ( (load_path + fname_begin + \
             fname_middle + fname_end + 'simulationData.mat') % \
             (0, 0, 0) )

# load data
data = loadmat(filename, simplify_cells=True)                
s_params = Dict2Class(data['sim_params'])
spikes = data['spikes']  

# unpack sim params
nClus = s_params.p
to = s_params.T0
tf = s_params.TF
tStim = s_params.stim_onset
 
# targeted and non-targeted clusters
targetedClusters = s_params.selectiveClusters
non_targetedClusters = np.setdiff1d(np.arange(0,nClus,1),targetedClusters)

# baseline time window
toBase = to + preStim_burn
tfBase = tStim
            
# time resolved firing rate of each neuron given window_width, window_step
tRates, Erates, Irates = fcn_compute_time_resolved_rate_gaussian(\
                                     s_params, spikes, \
                                     to, tf, window_std, window_step)
                    
nTpts = np.size(tRates)

#%% initialize all quantities for analysis

targetedCluster_gain_stimWindow = np.zeros((nNets, nTrials, nStim))
nontargetedCluster_gain_stimWindow = np.zeros((nNets, nTrials, nStim))
targeted_minus_nontargeted_gain_stimWindow = np.zeros((nNets, nTrials, nStim))
                                      
f_activeClusters_base = np.zeros((nNets, nTrials, nStim, n_gainThresh))
f_activeClusters_stim = np.zeros((nNets, nTrials, nStim, n_gainThresh))

frac_targetedClusters_active = np.zeros((nNets, nTrials, nStim, n_gainThresh))
frac_nontargetedClusters_active = np.zeros((nNets, nTrials, nStim, n_gainThresh))    
frac_targeted_minus_num_nontargeted_active = np.zeros((nNets, nTrials, nStim, n_gainThresh))

f_active_targtedClusters_inWindow = np.zeros((nNets, nTrials, nStim, n_gainThresh))
f_active_nontargtedClusters_inWindow = np.zeros((nNets, nTrials, nStim, n_gainThresh))
f_active_targeted_minus_nontargeted_clusters_inWindow = np.zeros((nNets, nTrials, nStim, n_gainThresh))

f_active_clusters_evokedWindow = np.zeros((nNets, nTrials, nStim, n_gainThresh))
f_active_clusters_baselineWindow = np.zeros((nNets, nTrials, nStim, n_gainThresh))
 
fracTrials_targetedClusters_active_inWindow = np.zeros((len(targetedClusters), nNets, nStim, n_gainThresh))
fracTrials_nontargetedClusters_active_inWindow = np.zeros((len(non_targetedClusters), nNets, nStim, n_gainThresh))

avg_frac_activeClusters_targeted_base = np.zeros((nNets, nTrials, nStim, n_gainThresh))
avg_frac_activeClusters_targeted_stim = np.zeros((nNets, nTrials, nStim, n_gainThresh))

fracWindow_targeted_active_stim = np.zeros((nNets, nTrials, nStim, n_gainThresh))
fracWindow_nontargeted_active_stim = np.zeros((nNets, nTrials, nStim, n_gainThresh))
fracWindow_targeted_nontargeted_active_stim = np.zeros((nNets, nTrials, nStim, n_gainThresh))

f_targeted_nontargeted_active_avgGain = np.zeros((nNets, nTrials, nStim, n_gainThresh))
f_targeted_active_avgGain = np.zeros((nNets, nTrials, nStim, n_gainThresh))
f_nontargeted_active_avgGain = np.zeros((nNets, nTrials, nStim, n_gainThresh))

f_active_base_avgGain = np.zeros((nNets, nTrials, nStim, n_gainThresh))
f_active_stim_avgGain = np.zeros((nNets, nTrials, nStim, n_gainThresh))

f_active_areTargeted_stim_avgGain = np.zeros((nNets, nTrials, nStim, n_gainThresh))
f_active_areTargeted_base_avgGain = np.zeros((nNets, nTrials, nStim, n_gainThresh))

#%% MAIN ANALYSIS BLOCK

for net_ind in range(0,nNets,1):
            
    # decoding filename
    decode_filename = ( (decoding_path + decoding_filename) % \
                          ( simID, sweep_param_str_val, net_ind, \
                            decode_windowSize*1000, decode_ensembleSize, decode_rateThresh, decode_type) )
    
    # load decoding data
    decode_data = loadmat(decode_filename, simplify_cells=True)
    
    # time of peak decoding accuracy
    t_decoding = decode_data['t_window']
    accuracy = decode_data['accuracy']
    t_peakAccuracy = t_decoding[np.argmax(accuracy)]  

    # stimulation time window
    tfStim = t_peakAccuracy
    toStim = tfStim - window_length

    active_targeted_in_window = np.zeros((nTrials, nStim, n_gainThresh, len(targetedClusters)))
    active_nontargeted_in_window = np.zeros((nTrials, nStim, n_gainThresh, len(non_targetedClusters)))
    
    rate_vs_time = np.zeros((nTrials, nStim, nClus, nTpts))


    for trial_ind in range(0,nTrials,1):
    
        print(trial_ind)
                    
        for stim_ind in range(0,nStim):
            
            print(stim_ind)
        
            # filename
            filename = ( (load_path + fname_begin + \
                         fname_middle + fname_end + 'simulationData.mat') % \
                         (net_ind, trial_ind, stim_ind) )
            
                
            # load data
            data = loadmat(filename, simplify_cells=True)                
            s_params = Dict2Class(data['sim_params'])
            spikes = data['spikes']  
            
            # unpack sim params
            nClus = s_params.p
            to = s_params.T0
            tf = s_params.TF
            tStim = s_params.stim_onset
            
            # get cluster assignments
            cluLabels = 0
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
            rate_vs_time[trial_ind, stim_ind, :, :] = Erates_clu.copy()
    
    # trial avg baseline rate
    baselineRate = rate_vs_time[:,:,:,inds_t_base].copy()
    avg_baselineRate = np.mean(baselineRate, axis=(0,1,3))
    std_baselineRate = np.std(baselineRate, axis=(0,1,3))

    # rate in pre stim window
    toBase = tStim - 2*window_length
    tfBase = toBase + window_length
    inds_t_base = np.nonzero( (tRates <= tfBase) & ( tRates >= toBase))[0]
    Erates_clu_preWindow = np.mean(rate_vs_time[:,:,:,inds_t_base], axis=(3))
    avg_Erates_clu_preWindow = np.mean(Erates_clu_preWindow, axis=(0,1))
    std_Erates_clu_preWindow = np.std(Erates_clu_preWindow, axis=(0,1))

    for trial_ind in range(0,nTrials,1):
    
        print(trial_ind)
                    
        for stim_ind in range(0,nStim):
            
            print(stim_ind)
        
            # filename
            filename = ( (load_path + fname_begin + \
                         fname_middle + fname_end + 'simulationData.mat') % \
                         (net_ind, trial_ind, stim_ind) )
            
            # load data
            data = loadmat(filename, simplify_cells=True)                
            s_params = Dict2Class(data['sim_params'])
            spikes = data['spikes']  
            
            # unpack sim params
            nClus = s_params.p
            to = s_params.T0
            tf = s_params.TF
            tStim = s_params.stim_onset

            # get cluster assignments
            cluLabels = 0
            clustSize_E = data['popSize_E']
            clustSize_I = data['popSize_I']   
 
                
            cluLabels = np.append(cluLabels, np.cumsum(clustSize_E))
            
            # baseline time window
            toBase = tStim - 2*window_length
            tfBase = toBase + window_length
            
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
            inds_t_stim = np.nonzero( (tRates <= tfStim) & ( tRates >= toStim))[0]
            
            # baseline window time indices
            inds_t_base = np.nonzero( (tRates <= tfBase) & ( tRates >= toBase))[0]

            
            # cluster rates
            Erates_clu_stim = Erates_clu[:, inds_t_stim].copy()
            Erates_clu_base = Erates_clu[:, inds_t_base].copy()

            Erates_clu_stimWindow = np.mean(Erates_clu_stim, axis=(1))


            # cluster gain
            Egain_clu_stim = np.zeros(np.shape(Erates_clu_stim))
            Egain_clu_base = np.zeros(np.shape(Erates_clu_base))
            Egain_clu_stimWindow = np.zeros(np.shape(Erates_clu_stimWindow))
            for indClu in range(0, nClus):
                if zscore:
                    Egain_clu_stim[indClu,:] = (Erates_clu_stim[indClu, :] - avg_baselineRate[indClu])/std_baselineRate[indClu]
                    Egain_clu_base[indClu,:] = (Erates_clu_base[indClu, :] - avg_baselineRate[indClu])/std_baselineRate[indClu]
                    Egain_clu_stimWindow[indClu] = (Erates_clu_stimWindow[indClu] - avg_Erates_clu_preWindow[indClu])/std_Erates_clu_preWindow[indClu]
                else:
                    Egain_clu_stim[indClu,:] = Erates_clu_stim[indClu, :] - avg_baselineRate[indClu]
                    Egain_clu_base[indClu,:] = Erates_clu_base[indClu, :] - avg_baselineRate[indClu]
                    Egain_clu_stimWindow[indClu] = (Erates_clu_stimWindow[indClu] - avg_Erates_clu_preWindow[indClu])
                    
            targetedCluster_gain_stimWindow[net_ind, trial_ind, stim_ind] = np.mean(Egain_clu_stimWindow[targetedClusters])
            nontargetedCluster_gain_stimWindow[net_ind, trial_ind, stim_ind] = np.mean(Egain_clu_stimWindow[non_targetedClusters])
            targeted_minus_nontargeted_gain_stimWindow[net_ind, trial_ind, stim_ind] = np.mean(Egain_clu_stimWindow[targetedClusters]) - np.mean(Egain_clu_stimWindow[non_targetedClusters])              
            

            # loop over thresholds
            for iThresh in range(0, n_gainThresh):
                            
                # rate threshold
                gain_thresh = gain_thresh_array[iThresh]

                # compute number of targeted clusters that exceed threshold during stim window 
                num_targeted_active_stim_avgGain = np.sum(np.mean(Egain_clu_stim[targetedClusters, :], 1) > gain_thresh)
                num_nontargeted_active_stim_avgGain = np.sum(np.mean(Egain_clu_stim[non_targetedClusters, :], 1) > gain_thresh)

                # compute number of targeted clusters that exceed threshold during base window 
                num_targeted_active_base_avgGain = np.sum(np.mean(Egain_clu_base[targetedClusters, :], 1) > gain_thresh)
                num_nontargeted_active_base_avgGain = np.sum(np.mean(Egain_clu_base[non_targetedClusters, :], 1) > gain_thresh)
                
                # fraction of targeted and non targeted clusters that are active
                f_targeted_active_avgGain[net_ind, trial_ind, stim_ind, iThresh] = num_targeted_active_stim_avgGain/np.size(targetedClusters)
                f_nontargeted_active_avgGain[net_ind, trial_ind, stim_ind, iThresh] = num_nontargeted_active_stim_avgGain/np.size(non_targetedClusters)
                
                f_targeted_nontargeted_active_avgGain[net_ind, trial_ind, stim_ind, iThresh] = \
                    f_targeted_active_avgGain[net_ind, trial_ind, stim_ind, iThresh] - f_nontargeted_active_avgGain[net_ind, trial_ind, stim_ind, iThresh]

                # fraction of active cluters during stim window and during baseline (using avg gain method)
                num_active_stim_avgGain = np.sum(np.mean(Egain_clu_stim, 1) > gain_thresh)
                num_active_base_avgGain = np.sum(np.mean(Egain_clu_base, 1) > gain_thresh)
                
                f_active_stim_avgGain[net_ind, trial_ind, stim_ind, iThresh] = num_active_stim_avgGain/nClus
                f_active_base_avgGain[net_ind, trial_ind, stim_ind, iThresh] = num_active_base_avgGain/nClus
                
                # fraction of active clusters that belong to stimulus targeted group (using avg gain method)
                f_active_areTargeted_stim_avgGain[net_ind, trial_ind, stim_ind, iThresh] = num_targeted_active_stim_avgGain/num_active_stim_avgGain
                f_active_areTargeted_base_avgGain[net_ind, trial_ind, stim_ind, iThresh] = num_targeted_active_base_avgGain/num_active_base_avgGain
                
                # compute number of active clusters at each baseline time point
                num_active_clusters_base = fcn_compute_num_activeClusters(Egain_clu_base, gain_thresh)  

                # compute number of active clusters at each stim time point
                num_active_clusters_stim = fcn_compute_num_activeClusters(Egain_clu_stim, gain_thresh)  

                # compute number of active targeted clusters at each time point during baseline
                num_active_targeted_clusters_base = fcn_compute_num_activeClusters(Egain_clu_base[targetedClusters,:], gain_thresh)

                # compute number of active targeted clusters at each time point during baseline
                num_active_targeted_clusters_stim = fcn_compute_num_activeClusters(Egain_clu_stim[targetedClusters,:], gain_thresh)
                
                # compute number of active nontargeted clusters at each time point during stimulation
                num_active_nontargeted_clusters_stim = fcn_compute_num_activeClusters(Egain_clu_stim[non_targetedClusters,:], gain_thresh)   
         
                # fraction of active clusters at each time point that are targeted (baseline)
                frac_active_clusters_targeted_base = num_active_targeted_clusters_base/num_active_clusters_base
                
                # fraction of active clusters at each time point that are targeted (stim)
                frac_active_clusters_targeted_stim = num_active_targeted_clusters_stim/num_active_clusters_stim          
                
                # fraction of simultaneously active clusters (time avg)
                f_activeClusters_base[net_ind, trial_ind, stim_ind, iThresh] = np.mean(num_active_clusters_base)/nClus
                f_activeClusters_stim[net_ind, trial_ind, stim_ind, iThresh] = np.mean(num_active_clusters_stim)/nClus

                # fraction of active clusters that are targeted (time avg)
                avg_frac_activeClusters_targeted_base[net_ind, trial_ind, stim_ind, iThresh] = np.mean(frac_active_clusters_targeted_base)
                avg_frac_activeClusters_targeted_stim[net_ind, trial_ind, stim_ind, iThresh] = np.mean(frac_active_clusters_targeted_stim)

                # fraction of targeted and non targeted clusters that are active (time avg)
                frac_targetedClusters_active[net_ind, trial_ind, stim_ind, iThresh] = np.mean(num_active_targeted_clusters_stim)/np.size(targetedClusters)
                frac_nontargetedClusters_active[net_ind, trial_ind, stim_ind, iThresh] = np.mean(num_active_nontargeted_clusters_stim)/np.size(non_targetedClusters)
                frac_targeted_minus_num_nontargeted_active[net_ind, trial_ind, stim_ind, iThresh] =  \
                    frac_targetedClusters_active[net_ind, trial_ind, stim_ind, iThresh] - frac_nontargetedClusters_active[net_ind, trial_ind, stim_ind, iThresh]
                
                # compute number of unique active clusters in evoked winow
                activeLifetimes = fcn_compute_cluster_activation_lifetimes(tRates[inds_t_stim], Egain_clu_stim, gain_thresh)
                
                # longest amount time each cluster is active
                max_cluLifetime = np.zeros(nClus)
                for indClu in range(0,nClus):
                    if np.size(activeLifetimes[indClu]) > 0:
                        max_cluLifetime[indClu] = np.max(activeLifetimes[indClu])
                    else:
                        max_cluLifetime[indClu] = 0
                
                # number active targeted clusters in window
                f_active_targtedClusters_inWindow[net_ind, trial_ind, stim_ind, iThresh] = np.size( np.nonzero(max_cluLifetime[targetedClusters] > lifetimeThresh)[0])/np.size(targetedClusters)
                # number active nontargeted clusters in window
                f_active_nontargtedClusters_inWindow[net_ind, trial_ind, stim_ind, iThresh] = np.size( np.nonzero(max_cluLifetime[non_targetedClusters] > lifetimeThresh)[0])/np.size(non_targetedClusters)
                # difference between number of active targeted and non targeted clusters in window
                f_active_targeted_minus_nontargeted_clusters_inWindow[net_ind, trial_ind, stim_ind, iThresh] = \
                    f_active_targtedClusters_inWindow[net_ind, trial_ind, stim_ind, iThresh] - f_active_nontargtedClusters_inWindow[net_ind, trial_ind, stim_ind, iThresh]
                # total number of clusters that become consistently active
                f_active_clusters_evokedWindow[net_ind, trial_ind, stim_ind, iThresh] = np.size( np.nonzero(max_cluLifetime > lifetimeThresh)[0])/nClus

                # number of trials in which each targeted cluster becomes active in window
                for ind, cluID in enumerate(targetedClusters):
                    fracTrials_targetedClusters_active_inWindow[ind, net_ind, stim_ind, iThresh] += (max_cluLifetime[cluID] > lifetimeThresh).astype(int)/nTrials
                
                for ind, cluID in enumerate(non_targetedClusters):
                    fracTrials_nontargetedClusters_active_inWindow[ind, net_ind, stim_ind, iThresh] += (max_cluLifetime[cluID] > lifetimeThresh).astype(int)/nTrials
                     
                # compute average fraction of window targeted and nontargeted clusters are active
                fracWindow_targeted_active_stim[net_ind, trial_ind, stim_ind, iThresh] = np.mean(np.sum((Egain_clu_stim[targetedClusters,:] >= gain_thresh),1)/np.size(Egain_clu_stim[0,:]))
                fracWindow_nontargeted_active_stim[net_ind, trial_ind, stim_ind, iThresh] = np.mean(np.sum((Egain_clu_stim[non_targetedClusters,:] >= gain_thresh),1)/np.size(Egain_clu_stim[0,:]))
                fracWindow_targeted_nontargeted_active_stim[net_ind, trial_ind, stim_ind, iThresh] = fracWindow_targeted_active_stim[net_ind, trial_ind, stim_ind, iThresh] - \
                                                                                                     fracWindow_nontargeted_active_stim[net_ind, trial_ind, stim_ind, iThresh]

                # compute number of unique active clusters in baseline winow
                activeLifetimes = fcn_compute_cluster_activation_lifetimes(tRates[inds_t_base], Egain_clu_base, gain_thresh)
                
                # longest amount time each cluster is active
                max_cluLifetime = np.zeros(nClus)
                for indClu in range(0,nClus):
                    if np.size(activeLifetimes[indClu]) > 0:
                        max_cluLifetime[indClu] = np.max(activeLifetimes[indClu])
                    else:
                        max_cluLifetime[indClu] = 0
                        
                # total number of clusters that become consistently active
                f_active_clusters_baselineWindow[net_ind, trial_ind, stim_ind, iThresh] = np.size( np.nonzero(max_cluLifetime > lifetimeThresh)[0])/nClus                            


#%% averages


trialAvg_targetedCluster_gain_stimWindow = np.mean(targetedCluster_gain_stimWindow, axis=(1,2))
netAvg_targetedCluster_gain_stimWindow = np.mean(trialAvg_targetedCluster_gain_stimWindow)
netStd_targetedCluster_gain_stimWindow = np.std(trialAvg_targetedCluster_gain_stimWindow)

trialAvg_nontargetedCluster_gain_stimWindow = np.mean(nontargetedCluster_gain_stimWindow, axis=(1,2))
netAvg_nontargetedCluster_gain_stimWindow = np.mean(trialAvg_nontargetedCluster_gain_stimWindow)
netStd_nontargetedCluster_gain_stimWindow = np.std(trialAvg_nontargetedCluster_gain_stimWindow)

trialAvg_targeted_minus_nontargeted_gain_stimWindow = np.mean(targeted_minus_nontargeted_gain_stimWindow, axis=(1,2))
netAvg_targeted_minus_nontargeted_gain_stimWindow = np.mean(trialAvg_targeted_minus_nontargeted_gain_stimWindow)
netStd_targeted_minus_nontargeted_gain_stimWindow = np.std(trialAvg_targeted_minus_nontargeted_gain_stimWindow)
   
trialAvg_frac_activeClusters_base = np.nanmean(f_activeClusters_base, axis=(1,2))
netAvg_frac_activeClusters_base = np.nanmean(trialAvg_frac_activeClusters_base, axis=0)
netStd_frac_activeClusters_base = np.nanstd(trialAvg_frac_activeClusters_base, axis=0)
netSem_frac_activeClusters_base = scipy.stats.sem(trialAvg_frac_activeClusters_base, axis=0)

trialAvg_frac_activeClusters_stim = np.mean(f_activeClusters_stim, axis=(1,2))
netAvg_frac_activeClusters_stim = np.nanmean(trialAvg_frac_activeClusters_stim, axis=0)
netStd_frac_activeClusters_stim = np.nanstd(trialAvg_frac_activeClusters_stim, axis=0)
netSem_frac_activeClusters_stim = scipy.stats.sem(trialAvg_frac_activeClusters_stim, axis=0)   

trialAvg_frac_targetedClusters_active = np.nanmean(frac_targetedClusters_active, axis=(1,2))
netAvg_frac_targetedClusters_active = np.nanmean(trialAvg_frac_targetedClusters_active,axis=0)
netStd_frac_targetedClusters_active = np.nanstd(trialAvg_frac_targetedClusters_active,axis=0)     
netSem_frac_targetedClusters_active = scipy.stats.sem(trialAvg_frac_targetedClusters_active, axis=0, nan_policy='omit') 
    
trialAvg_frac_nontargetedClusters_active = np.nanmean(frac_nontargetedClusters_active, axis=(1,2))
netAvg_frac_nontargetedClusters_active = np.nanmean(trialAvg_frac_nontargetedClusters_active,axis=0)
netStd_frac_nontargetedClusters_active = np.nanstd(trialAvg_frac_nontargetedClusters_active,axis=0)     
netSem_frac_nontargetedClusters_active = scipy.stats.sem(trialAvg_frac_nontargetedClusters_active, axis=0, nan_policy='omit') 

trialAvg_frac_targeted_minus_num_nontargeted_active = np.nanmean(frac_targeted_minus_num_nontargeted_active, axis=(1,2))
netAvg_frac_targeted_minus_num_nontargeted_active = np.nanmean(trialAvg_frac_targeted_minus_num_nontargeted_active,axis=0)
netStd_frac_targeted_minus_num_nontargeted_active = np.nanstd(trialAvg_frac_targeted_minus_num_nontargeted_active,axis=0)     
netSem_frac_targeted_minus_num_nontargeted_active = scipy.stats.sem(trialAvg_frac_targeted_minus_num_nontargeted_active, axis=0, nan_policy='omit') 
    
trialAvg_f_active_clusters_evokedWindow = np.nanmean(f_active_clusters_evokedWindow, axis=(1,2))
netAvg_f_active_clusters_evokedWindow = np.nanmean(trialAvg_f_active_clusters_evokedWindow, axis=(0))
netStd_f_active_clusters_evokedWindow = np.nanstd(trialAvg_f_active_clusters_evokedWindow, axis=(0))
netSem_f_active_clusters_evokedWindow = scipy.stats.sem(trialAvg_f_active_clusters_evokedWindow, axis=(0))

trialAvg_f_active_clusters_baselineWindow = np.nanmean(f_active_clusters_baselineWindow, axis=(1,2))
netAvg_f_active_clusters_baselineWindow = np.nanmean(trialAvg_f_active_clusters_baselineWindow, axis=(0))
netStd_f_active_clusters_baselineWindow = np.nanstd(trialAvg_f_active_clusters_baselineWindow, axis=(0))
netSem_f_active_clusters_baselineWindow = scipy.stats.sem(trialAvg_f_active_clusters_baselineWindow, axis=(0))

trialAvg_f_active_targtedClusters_inWindow = np.nanmean(f_active_targtedClusters_inWindow, axis=(1,2))
netAvg_f_active_targtedClusters_inWindow = np.nanmean(trialAvg_f_active_targtedClusters_inWindow,axis=0)
netStd_f_active_targtedClusters_inWindow = np.nanstd(trialAvg_f_active_targtedClusters_inWindow,axis=0)     
netSem_f_active_targtedClusters_inWindow = scipy.stats.sem(trialAvg_f_active_targtedClusters_inWindow, axis=0, nan_policy='omit') 

trialAvg_f_active_nontargtedClusters_inWindow = np.nanmean(f_active_nontargtedClusters_inWindow, axis=(1,2))
netAvg_f_active_nontargtedClusters_inWindow = np.nanmean(trialAvg_f_active_nontargtedClusters_inWindow,axis=0)
netStd_f_active_nontargtedClusters_inWindow = np.nanstd(trialAvg_f_active_nontargtedClusters_inWindow,axis=0)     
netSem_f_active_nontargtedClusters_inWindow = scipy.stats.sem(trialAvg_f_active_nontargtedClusters_inWindow, axis=0, nan_policy='omit') 

trialAvg_f_active_targeted_minus_nontargeted_clusters_inWindow = np.nanmean(f_active_targeted_minus_nontargeted_clusters_inWindow, axis=(1,2)) 
netAvg_f_active_targeted_minus_nontargeted_clusters_inWindow = np.nanmean(trialAvg_f_active_targeted_minus_nontargeted_clusters_inWindow, axis=0)
netStd_f_active_targeted_minus_nontargeted_clusters_inWindow = np.nanstd(trialAvg_f_active_targeted_minus_nontargeted_clusters_inWindow, axis=0)
netSem_f_active_targeted_minus_nontargeted_clusters_inWindow = scipy.stats.sem(trialAvg_f_active_targeted_minus_nontargeted_clusters_inWindow, axis=0, nan_policy='omit')

trialAvg_fracTrials_targetedClusters_active_inWindow = np.nanmean(fracTrials_targetedClusters_active_inWindow, axis=(0,2))
netAvg_fracTrials_targetedClusters_active_inWindow = np.nanmean(trialAvg_fracTrials_targetedClusters_active_inWindow, axis=(0))
netStd_fracTrials_targetedClusters_active_inWindow = np.nanstd(trialAvg_fracTrials_targetedClusters_active_inWindow, axis=(0))

trialAvg_fracTrials_nontargetedClusters_active_inWindow = np.nanmean(fracTrials_nontargetedClusters_active_inWindow, axis=(0,2))
netAvg_fracTrials_nontargetedClusters_active_inWindow = np.nanmean(trialAvg_fracTrials_nontargetedClusters_active_inWindow, axis=(0))
netStd_fracTrials_nontargetedClusters_active_inWindow = np.nanstd(trialAvg_fracTrials_nontargetedClusters_active_inWindow, axis=(0))

trialAvg_avg_frac_activeClusters_targeted_base = np.nanmean(avg_frac_activeClusters_targeted_base, axis=(1,2))
netAvg_avg_frac_activeClusters_targeted_base = np.nanmean(trialAvg_avg_frac_activeClusters_targeted_base, axis=(0))
netStd_avg_frac_activeClusters_targeted_base = np.nanstd(trialAvg_avg_frac_activeClusters_targeted_base, axis=(0))
netSem_avg_frac_activeClusters_targeted_base = scipy.stats.sem(trialAvg_avg_frac_activeClusters_targeted_base, axis=(0))

trialAvg_avg_frac_activeClusters_targeted_stim = np.nanmean(avg_frac_activeClusters_targeted_stim, axis=(1,2))
netAvg_avg_frac_activeClusters_targeted_stim = np.nanmean(trialAvg_avg_frac_activeClusters_targeted_stim, axis=(0))
netStd_avg_frac_activeClusters_targeted_stim = np.nanstd(trialAvg_avg_frac_activeClusters_targeted_stim, axis=(0))
netSem_avg_frac_activeClusters_targeted_stim = scipy.stats.sem(trialAvg_avg_frac_activeClusters_targeted_stim, axis=(0))

trialAvg_fracWindow_targeted_active_stim = np.nanmean(fracWindow_targeted_active_stim, axis=(1,2))
netAvg_fracWindow_targeted_active_stim = np.nanmean(trialAvg_fracWindow_targeted_active_stim, axis=(0))
netStd_fracWindow_targeted_active_stim = np.nanstd(trialAvg_fracWindow_targeted_active_stim, axis=(0))

trialAvg_fracWindow_nontargeted_active_stim = np.nanmean(fracWindow_nontargeted_active_stim, axis=(1,2))
netAvg_fracWindow_nontargeted_active_stim = np.nanmean(trialAvg_fracWindow_nontargeted_active_stim, axis=(0))
netStd_fracWindow_nontargeted_active_stim = np.nanstd(trialAvg_fracWindow_nontargeted_active_stim, axis=(0))

trialAvg_fracWindow_targeted_nontargeted_active_stim = np.nanmean(fracWindow_targeted_nontargeted_active_stim, axis=(1,2))
netAvg_fracWindow_targeted_nontargeted_active_stim = np.nanmean(trialAvg_fracWindow_targeted_nontargeted_active_stim, axis=(0))
netStd_fracWindow_targeted_nontargeted_active_stim = np.nanstd(trialAvg_fracWindow_targeted_nontargeted_active_stim, axis=(0))

trialAvg_f_targeted_nontargeted_active_avgGain = np.nanmean(f_targeted_nontargeted_active_avgGain, axis=(1,2))
netAvg_f_targeted_nontargeted_active_avgGain = np.nanmean(trialAvg_f_targeted_nontargeted_active_avgGain, axis=(0))
netStd_f_targeted_nontargeted_active_avgGain = np.nanstd(trialAvg_f_targeted_nontargeted_active_avgGain, axis=(0))

trialAvg_f_targeted_active_avgGain = np.nanmean(f_targeted_active_avgGain, axis=(1,2))
netAvg_f_targeted_active_avgGain = np.nanmean(trialAvg_f_targeted_active_avgGain, axis=(0))
netStd_f_targeted_active_avgGain = np.nanstd(trialAvg_f_targeted_active_avgGain, axis=(0))

trialAvg_f_nontargeted_active_avgGain = np.nanmean(f_nontargeted_active_avgGain, axis=(1,2))
netAvg_f_nontargeted_active_avgGain = np.nanmean(trialAvg_f_nontargeted_active_avgGain, axis=(0))
netStd_f_nontargeted_active_avgGain = np.nanstd(trialAvg_f_nontargeted_active_avgGain, axis=(0))

trialAvg_f_active_base_avgGain = np.nanmean(f_active_base_avgGain, axis=(1,2))
netAvg_f_active_base_avgGain = np.nanmean(trialAvg_f_active_base_avgGain, axis=(0))
netStd_f_active_base_avgGain = np.nanstd(trialAvg_f_active_base_avgGain, axis=(0))

trialAvg_f_active_stim_avgGain = np.nanmean(f_active_stim_avgGain, axis=(1,2))
netAvg_f_active_stim_avgGain = np.nanmean(trialAvg_f_active_stim_avgGain, axis=(0))
netStd_f_active_stim_avgGain = np.nanstd(trialAvg_f_active_stim_avgGain, axis=(0))

trialAvg_f_active_areTargeted_stim_avgGain = np.nanmean(f_active_areTargeted_stim_avgGain, axis=(1,2))
netAvg_f_active_areTargeted_stim_avgGain = np.nanmean(trialAvg_f_active_areTargeted_stim_avgGain, axis=(0))
netStd_f_active_areTargeted_stim_avgGain = np.nanstd(trialAvg_f_active_areTargeted_stim_avgGain, axis=(0))    

trialAvg_f_active_areTargeted_base_avgGain = np.nanmean(f_active_areTargeted_base_avgGain, axis=(1,2))
netAvg_f_active_areTargeted_base_avgGain = np.nanmean(trialAvg_f_active_areTargeted_base_avgGain, axis=(0))
netStd_f_active_areTargeted_base_avgGain = np.nanstd(trialAvg_f_active_areTargeted_base_avgGain, axis=(0))    


#%% save the results
    
results = {}
parameters = {}

parameters['load_path'] = load_path
parameters['save_path'] = save_path
parameters['simID'] = simID
parameters['net_type'] = net_type
parameters['nNets'] = nNets
parameters['nTrials'] = nTrials
parameters['nStim'] = nStim
parameters['stim_shape']   =  stim_shape
parameters['stim_type'] = stim_type
parameters['stim_rel_amp'] = stim_rel_amp
parameters['sweep_param_str_val'] = sweep_param_str_val
parameters['zscore'] = zscore
parameters['preStim_burn'] = preStim_burn
parameters['window_length'] = window_length
parameters['window_std'] = window_std
parameters['window_step']  =   window_step
parameters['gain_thresh'] = gain_thresh_array
parameters['decoding_path'] = decoding_path
parameters['decode_ensembleSize'] = decode_ensembleSize
parameters['decode_windowSize'] = decode_windowSize
parameters['decode_type'] = decode_type

results['parameters'] = parameters

results['trialAvg_targetedCluster_gain_stimWindow'] = trialAvg_targetedCluster_gain_stimWindow
results['netAvg_targetedCluster_gain_stimWindow'] = netAvg_targetedCluster_gain_stimWindow
results['netStd_targetedCluster_gain_stimWindow'] = netStd_targetedCluster_gain_stimWindow

results['trialAvg_nontargetedCluster_gain_stimWindow'] = trialAvg_nontargetedCluster_gain_stimWindow
results['netAvg_nontargetedCluster_gain_stimWindow'] = netAvg_nontargetedCluster_gain_stimWindow
results['netStd_nontargetedCluster_gain_stimWindow'] = netStd_nontargetedCluster_gain_stimWindow    

results['trialAvg_targeted_minus_nontargeted_gain_stimWindow'] = trialAvg_targeted_minus_nontargeted_gain_stimWindow
results['netAvg_targeted_minus_nontargeted_gain_stimWindow'] = netAvg_targeted_minus_nontargeted_gain_stimWindow
results['netStd_targeted_minus_nontargeted_gain_stimWindow'] = netStd_targeted_minus_nontargeted_gain_stimWindow

results['trialAvg_frac_activeClusters_base'] = trialAvg_frac_activeClusters_base
results['netAvg_frac_activeClusters_base'] = netAvg_frac_activeClusters_base
results['netStd_frac_activeClusters_base'] = netStd_frac_activeClusters_base
results['netSem_frac_activeClusters_base'] = netSem_frac_activeClusters_base

results['trialAvg_frac_activeClusters_stim'] = trialAvg_frac_activeClusters_stim
results['netAvg_frac_activeClusters_stim'] = netAvg_frac_activeClusters_stim
results['netStd_frac_activeClusters_stim'] = netStd_frac_activeClusters_stim
results['netSem_frac_activeClusters_stim'] = netSem_frac_activeClusters_stim

results['trialAvg_frac_targetedClusters_active'] = trialAvg_frac_targetedClusters_active
results['netAvg_frac_targetedClusters_active'] = netAvg_frac_targetedClusters_active
results['netStd_frac_targetedClusters_active'] = netStd_frac_targetedClusters_active
results['netSem_frac_targetedClusters_active'] = netSem_frac_targetedClusters_active

results['trialAvg_frac_nontargetedClusters_active'] = trialAvg_frac_nontargetedClusters_active
results['netAvg_frac_nontargetedClusters_active'] = netAvg_frac_nontargetedClusters_active
results['netStd_frac_nontargetedClusters_active'] = netStd_frac_nontargetedClusters_active
results['netSem_frac_nontargetedClusters_active'] = netSem_frac_nontargetedClusters_active

results['trialAvg_frac_targeted_minus_num_nontargeted_active'] = trialAvg_frac_targeted_minus_num_nontargeted_active
results['netAvg_frac_targeted_minus_num_nontargeted_active'] = netAvg_frac_targeted_minus_num_nontargeted_active
results['netStd_frac_targeted_minus_num_nontargeted_active'] = netStd_frac_targeted_minus_num_nontargeted_active
results['netSem_frac_targeted_minus_num_nontargeted_active'] = netSem_frac_targeted_minus_num_nontargeted_active
    
results['trialAvg_f_active_clusters_evokedWindow'] = trialAvg_f_active_clusters_evokedWindow
results['netAvg_f_active_clusters_evokedWindow'] = netAvg_f_active_clusters_evokedWindow
results['netStd_f_active_clusters_evokedWindow'] = netStd_f_active_clusters_evokedWindow
results['netSem_f_active_clusters_evokedWindow'] = netSem_f_active_clusters_evokedWindow

results['trialAvg_f_active_clusters_baselineWindow'] = trialAvg_f_active_clusters_baselineWindow
results['netAvg_f_active_clusters_baselineWindow'] = netAvg_f_active_clusters_baselineWindow
results['netStd_f_active_clusters_baselineWindow'] = netStd_f_active_clusters_baselineWindow
results['netSem_f_active_clusters_baselineWindow'] = netSem_f_active_clusters_baselineWindow

results['trialAvg_f_active_targtedClusters_inWindow'] = trialAvg_f_active_targtedClusters_inWindow
results['netAvg_f_active_targtedClusters_inWindow'] = netAvg_f_active_targtedClusters_inWindow
results['netStd_f_active_targtedClusters_inWindow'] = netStd_f_active_targtedClusters_inWindow
results['netSem_f_active_targtedClusters_inWindow'] = netSem_f_active_targtedClusters_inWindow

results['trialAvg_f_active_nontargtedClusters_inWindow'] = trialAvg_f_active_nontargtedClusters_inWindow
results['netAvg_f_active_nontargtedClusters_inWindow'] = netAvg_f_active_nontargtedClusters_inWindow
results['netStd_f_active_nontargtedClusters_inWindow'] = netStd_f_active_nontargtedClusters_inWindow
results['netSem_f_active_nontargtedClusters_inWindow'] = netSem_f_active_nontargtedClusters_inWindow    

results['trialAvg_f_active_targeted_minus_nontargeted_clusters_inWindow'] = trialAvg_f_active_targeted_minus_nontargeted_clusters_inWindow
results['netAvg_f_active_targeted_minus_nontargeted_clusters_inWindow'] = netAvg_f_active_targeted_minus_nontargeted_clusters_inWindow
results['netStd_f_active_targeted_minus_nontargeted_clusters_inWindow'] = netStd_f_active_targeted_minus_nontargeted_clusters_inWindow
results['netSem_f_active_targeted_minus_nontargeted_clusters_inWindow'] = netSem_f_active_targeted_minus_nontargeted_clusters_inWindow        

results['trialAvg_fracTrials_targetedClusters_active_inWindow'] = trialAvg_fracTrials_targetedClusters_active_inWindow
results['netAvg_fracTrials_targetedClusters_active_inWindow'] = netAvg_fracTrials_targetedClusters_active_inWindow
results['netStd_fracTrials_targetedClusters_active_inWindow'] = netStd_fracTrials_targetedClusters_active_inWindow

results['trialAvg_fracTrials_nontargetedClusters_active_inWindow'] = trialAvg_fracTrials_nontargetedClusters_active_inWindow
results['netAvg_fracTrials_nontargetedClusters_active_inWindow'] = netAvg_fracTrials_nontargetedClusters_active_inWindow
results['netStd_fracTrials_nontargetedClusters_active_inWindow'] = netStd_fracTrials_nontargetedClusters_active_inWindow

results['trialAvg_avg_frac_activeClusters_targeted_base'] = trialAvg_avg_frac_activeClusters_targeted_base
results['netAvg_avg_frac_activeClusters_targeted_base'] = netAvg_avg_frac_activeClusters_targeted_base
results['netStd_avg_frac_activeClusters_targeted_base'] = netStd_avg_frac_activeClusters_targeted_base
results['netSem_avg_frac_activeClusters_targeted_base'] = netSem_avg_frac_activeClusters_targeted_base
 
results['trialAvg_avg_frac_activeClusters_targeted_stim'] = trialAvg_avg_frac_activeClusters_targeted_stim
results['netAvg_avg_frac_activeClusters_targeted_stim'] = netAvg_avg_frac_activeClusters_targeted_stim
results['netStd_avg_frac_activeClusters_targeted_stim'] = netStd_avg_frac_activeClusters_targeted_stim
results['netSem_avg_frac_activeClusters_targeted_stim'] = netSem_avg_frac_activeClusters_targeted_stim

results['trialAvg_fracWindow_targeted_active_stim'] = trialAvg_fracWindow_targeted_active_stim
results['netAvg_fracWindow_targeted_active_stim'] = netAvg_fracWindow_targeted_active_stim
results['netStd_fracWindow_targeted_active_stim'] = netStd_fracWindow_targeted_active_stim

results['trialAvg_fracWindow_nontargeted_active_stim'] = trialAvg_fracWindow_nontargeted_active_stim
results['netAvg_fracWindow_nontargeted_active_stim'] = netAvg_fracWindow_nontargeted_active_stim
results['netStd_fracWindow_nontargeted_active_stim'] = netStd_fracWindow_nontargeted_active_stim

results['trialAvg_fracWindow_targeted_nontargeted_active_stim'] = trialAvg_fracWindow_targeted_nontargeted_active_stim
results['netAvg_fracWindow_targeted_nontargeted_active_stim'] = netAvg_fracWindow_targeted_nontargeted_active_stim
results['netStd_fracWindow_targeted_nontargeted_active_stim'] = netStd_fracWindow_targeted_nontargeted_active_stim

results['netAvg_f_targeted_nontargeted_active_avgGain'] = netAvg_f_targeted_nontargeted_active_avgGain
results['netStd_f_targeted_nontargeted_active_avgGain'] = netStd_f_targeted_nontargeted_active_avgGain

results['netAvg_f_targeted_active_avgGain'] = netAvg_f_targeted_active_avgGain
results['netStd_f_targeted_active_avgGain'] = netStd_f_targeted_active_avgGain

results['netAvg_f_nontargeted_active_avgGain'] = netAvg_f_nontargeted_active_avgGain
results['netStd_f_nontargeted_active_avgGain'] = netStd_f_nontargeted_active_avgGain

results['netAvg_f_active_stim_avgGain'] = netAvg_f_active_stim_avgGain
results['netStd_f_active_stim_avgGain'] = netStd_f_active_stim_avgGain    

results['netAvg_f_active_base_avgGain'] = netAvg_f_active_base_avgGain
results['netStd_f_active_base_avgGain'] = netStd_f_active_base_avgGain    

results['trialAvg_f_active_areTargeted_stim_avgGain'] = trialAvg_f_active_areTargeted_stim_avgGain    
results['netAvg_f_active_areTargeted_stim_avgGain'] = netAvg_f_active_areTargeted_stim_avgGain
results['netStd_f_active_areTargeted_stim_avgGain'] = netStd_f_active_areTargeted_stim_avgGain

results['trialAvg_f_active_areTargeted_base_avgGain'] = trialAvg_f_active_areTargeted_base_avgGain
results['netAvg_f_active_areTargeted_base_avgGain'] = netAvg_f_active_areTargeted_base_avgGain
results['netStd_f_active_areTargeted_base_avgGain'] = netStd_f_active_areTargeted_base_avgGain
    

if zscore:
    savemat(('%s%s%s_numActive_targeted_vs_nontargeted_clusters_gainBased_zscore.mat' % (save_path, fname_begin, fname_end)), results)
else:
    savemat(('%s%s%s_numActive_targeted_vs_nontargeted_clusters_gainBased.mat' % (save_path, fname_begin, fname_end)), results)

