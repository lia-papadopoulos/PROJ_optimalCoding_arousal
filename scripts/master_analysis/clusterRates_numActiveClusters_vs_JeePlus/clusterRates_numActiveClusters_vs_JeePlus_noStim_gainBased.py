'''
compute the rate of active clusters and the number of active clusters at each
point in time during baseline period of simulations
'''


#-------------------- basic imports ----------------------------------------------#

import numpy as np
import scipy.stats
import numpy.matlib
from scipy.io import savemat
from scipy.io import loadmat
import sys
import argparse
import importlib

import clusterRates_numActiveClusters_vs_JeePlus_noStim_gainBased_settings as settings


#-------------------- functions ----------------------------#

func_path1 = settings.func_path1
sys.path.append(func_path1)
from fcn_compute_firing_stats import Dict2Class
from fcn_compute_firing_stats import fcn_compute_firingRates
from fcn_compute_firing_stats import fcn_compute_clusterRates_vs_time
from fcn_compute_firing_stats import fcn_compute_time_resolved_rate_gaussian
from fcn_compute_firing_stats import fcn_compute_num_activeClusters
from fcn_compute_firing_stats import fcn_compute_clusterRate_XClustersActive
from fcn_compute_firing_stats import fcn_compute_inactive_clusterRate_XClustersActive
from fcn_compute_firing_stats import fcn_compute_backgroundRate_XClustersActive
from fcn_compute_firing_stats import fcn_compute_freq_XActiveClusters
from fcn_compute_firing_stats import fcn_compute_activeCluster_rates_givenBinarized
from fcn_compute_firing_stats import fcn_compute_inactiveCluster_rates_givenBinarized
from fcn_compute_firing_stats import fcn_compute_clusterActivation


#-------------------- path for loading/saving data ----------------------------#
load_path = settings.load_path
save_path = settings.save_path
fig_path = settings.fig_path
sim_params_path = settings.sim_params_path

    
#-------------------- parameters ----------------------------------------------#
simParams_fname = settings.simParams_fname
net_type = settings.net_type
nNets = settings.nNetworks

#-------------------- analysis parameters ----------------------------------------------#
burnTime_begin = settings.burnTime_begin
burnTime_end = settings.burnTime_end
window_step = settings.window_step
window_std = settings.window_std
rate_thresh_array = settings.rate_thresh_array


#%% load sim parameters
sys.path.append(sim_params_path)
params = importlib.import_module(simParams_fname) 
s_params = params.sim_params


#%% unpack sim params
simID = s_params['simID']
nTrials = s_params['n_ICs']
stim_shape = s_params['stim_shape']
stim_rel_amp = s_params['stim_rel_amp']
nStim = s_params['nStim']

del params
del s_params


#%% MAIN
#-------------------- argparser ----------------------------------------------#

parser = argparse.ArgumentParser() 

# swept parameter name + value as string
parser.add_argument('-sweep_param_str_val', '--sweep_param_str_val', type=str, required=True)

# index of swept parameter
parser.add_argument('-param_indx', '--param_indx', type=int, required=True)    

# arguments of parser
args = parser.parse_args()


#-------------------- argparser values for later use -------------------------#

# name of swept parameter with value as a string
sweep_param_str_val = args.sweep_param_str_val

# index of swept parameter
inputParam_indx = args.param_indx


#-------------------- set filenames ------------------------------------------#

# beginning of filename
fname_begin = ( '%s_%s_sweep_%s' % (simID, net_type, sweep_param_str_val) )

# middle of filename
fname_middle = ( '_network%d_IC%d_stim%d' )

# end of filename
fname_end = ( '_stimType_%s_stim_rel_amp%0.3f_' % (stim_shape, stim_rel_amp) )


#-------------------- window sds and rate thresh -------------------------#

n_rateThresh = np.size(rate_thresh_array)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#-------------------- set up ---------------------------------------------#

# filename
filename = ( (load_path + fname_begin + fname_middle + fname_end + 'simulationData.mat') % (0, 0 ,0) )
# load data
data = loadmat(filename, simplify_cells=True)   
# sim_params         
s_params = Dict2Class(data['sim_params'])
nClus = s_params.p
to = s_params.T0
tf = s_params.TF
tStim = s_params.stim_onset
# spikes
spikes = data['spikes']  

# time resolved firing rate of each neuron given window_width, window_step
tRates, Erates, Irates = fcn_compute_time_resolved_rate_gaussian(s_params, spikes, to, tf, window_std, window_step)  

# burn time indices
indBurn_begin = np.argmin(np.abs(tRates - (to + burnTime_begin)))
indBurn_end = np.argmin(np.abs(tRates - (tStim - burnTime_end)))

print(tRates[indBurn_begin])
print(tRates[indBurn_end])
        
# number of analysis time points
tRates = tRates[indBurn_begin:indBurn_end]
nTpts = np.size(tRates)




#-------------------- initialization --------------------------------------#

popAvg_rate_E = np.zeros((nNets, nTrials, nStim))
popAvg_activeCluster_rate_e = np.zeros((nNets, nTrials, nStim, n_rateThresh))
popAvg_inactiveCluster_rate_e = np.zeros((nNets, nTrials, nStim, n_rateThresh))

prob_nActive_clusters_E = np.zeros((nNets, nTrials, nStim, nClus+1, n_rateThresh))
avg_num_active_clusters_E = np.zeros((nNets, nTrials, nStim, n_rateThresh))

avgRate_background_XActiveClusters_E = np.zeros((nNets, nTrials, nStim, nClus+1, n_rateThresh))
avgRate_active_XActiveClusters_E = np.zeros((nNets, nTrials, nStim, nClus+1, n_rateThresh))
avgRate_inactive_XActiveClusters_E = np.zeros((nNets, nTrials, nStim,  nClus+1, n_rateThresh))



#------------------------------------------------------------------------------
# NUMBER OF ACTIVE CLUSTERS AT A GIVEN TIME
#------------------------------------------------------------------------------

for net_ind in range(0,nNets,1):
    
    print(net_ind)
    
    # initialize cluster rates
    clusterRates_vs_time = np.zeros((nTrials, nStim, nClus, nTpts))
    bgRates_vs_time = np.zeros((nTrials, nStim, nTpts))


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
            
            # network average firing rate
            Erates_timeAvg, Irates_timeAvg = fcn_compute_firingRates(s_params, spikes, burnTime_begin)
            popAvg_rate_E[net_ind, trial_ind, stim_ind] = np.mean(Erates_timeAvg)

            #------------------------------------------------------------------------------  
            #------------------------------------------------------------------------------
            
            # time resolved firing rate of each neuron given window_width, window_step
            tRates, Erates, Irates = fcn_compute_time_resolved_rate_gaussian(\
                             s_params, spikes, \
                             to, tf, window_std, window_step)
    
            
            # time resolved E cluster rates
            Erates_clu = fcn_compute_clusterRates_vs_time(clustSize_E, Erates)
            
            # cluster only
            Erates_clu = Erates_clu[:nClus, :]
            
            # background only
            inds_background = np.arange(cluLabels[-2],cluLabels[-1])
            Erates_bg = Erates[inds_background,:]
            
            # update rates
            tRates = tRates[indBurn_begin:indBurn_end]
            Erates_clu = Erates_clu[:,indBurn_begin:indBurn_end]  
            Erates_bg = Erates_bg[:,indBurn_begin:indBurn_end]     
            
            # save time dependnet cluster rates
            clusterRates_vs_time[trial_ind, stim_ind, :, :] = Erates_clu.copy()
            
            # save time dependent background rates
            avgRate_bgE = np.mean(Erates_bg, axis=0) 
            bgRates_vs_time[trial_ind, stim_ind, :] = avgRate_bgE.copy()
           
                                           
    # avg cluster rates
    avg_clusterRates = np.mean(clusterRates_vs_time, axis=(0,1,3))
    
    
    # loop over trials and compute rates of ative and inactive clusters
    for trial_ind in range(0,nTrials,1):
        
        print('trial_ind %d' % trial_ind)
                                    
        for stim_ind in range(0,nStim):
            
            print('stim_ind %d' % stim_ind)
            
            # compute cluster gain
            Erates_clu = clusterRates_vs_time[trial_ind, stim_ind, :, :].copy()
            avgRate_bgE = bgRates_vs_time[trial_ind, stim_ind, :].copy()
            clusterGain_vs_time = np.zeros((nClus, nTpts))
            for indClu in range(0, nClus):
                clusterGain_vs_time[indClu, :] = clusterRates_vs_time[trial_ind, stim_ind, indClu, :] - avg_clusterRates[indClu]
            
                
            for iThresh in range(0, n_rateThresh):
                
                print('thresh %d' % iThresh)


                rate_thresh = rate_thresh_array[iThresh]

                # compute number of active clusters at each time point
                num_active_clusters = fcn_compute_num_activeClusters(clusterGain_vs_time, rate_thresh)
                    
                # compute probability of observing x active clusters
                prob_nActive_clusters_E[net_ind, trial_ind, stim_ind, :, iThresh] = fcn_compute_freq_XActiveClusters(num_active_clusters, nClus)                            
                
                # compute binarized cluster activity
                cluActivity_binarized = fcn_compute_clusterActivation(clusterGain_vs_time, rate_thresh)
                
                # compute avg rate of active clusters from binarized activity
                avgRate_activeClus = fcn_compute_activeCluster_rates_givenBinarized(tRates, Erates_clu, cluActivity_binarized)
                
                # compute avg rate of inactive clusters from binarized activity
                avgRate_inactiveClus = fcn_compute_inactiveCluster_rates_givenBinarized(tRates, Erates_clu, cluActivity_binarized)
                    
                              
                # average rate of in/active clusters, regardless of # active
                popAvg_activeCluster_rate_e[net_ind, trial_ind, stim_ind, iThresh] = np.mean(avgRate_activeClus)
                popAvg_inactiveCluster_rate_e[net_ind, trial_ind, stim_ind, iThresh] = np.mean(avgRate_inactiveClus)
                
                
                # compute avg rate of background as a function of # inactive clusters
                avgRate_background_XActiveClusters_E[net_ind, trial_ind, stim_ind, :, iThresh] = \
                    fcn_compute_backgroundRate_XClustersActive(avgRate_bgE, num_active_clusters, nClus)
                        
                
                # compute avg rate of active clusters as a function of # active clusters
                avgRate_active_XActiveClusters_E[net_ind, trial_ind, stim_ind, :, iThresh] = \
                    fcn_compute_clusterRate_XClustersActive(avgRate_activeClus, num_active_clusters, nClus)      
                        
                    
                # compute avg rate of inactive clusters as a function of # active clusters
                avgRate_inactive_XActiveClusters_E[net_ind, trial_ind, stim_ind, :, iThresh] = \
                    fcn_compute_inactive_clusterRate_XClustersActive(avgRate_inactiveClus, num_active_clusters, nClus)                                       
                        
                        
                # averages
                avg_num_active_clusters_E[net_ind, trial_ind, stim_ind, iThresh] = np.mean(num_active_clusters)
                                                                                                                                   

#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 

# population average rate
trialAvg_popAvg_rate_E = np.mean(popAvg_rate_E, axis=(1,2))
netAvg_popAvg_rate_E = np.mean(trialAvg_popAvg_rate_E, axis=0)
netStd_popAvg_rate_E = np.nanstd(trialAvg_popAvg_rate_E, axis=0)
netSem_popAvg_rate_E = scipy.stats.sem(trialAvg_popAvg_rate_E, axis=0, nan_policy='omit')

# number of active clusters
trialAvg_avg_num_active_clusters_E = np.nanmean(avg_num_active_clusters_E, axis=(1,2))
netAvg_avg_num_active_clusters_E = np.nanmean(trialAvg_avg_num_active_clusters_E,axis=0)
netStd_avg_num_active_clusters_E = np.nanstd(trialAvg_avg_num_active_clusters_E,axis=0)     
netSem_avg_num_active_clusters_E = scipy.stats.sem(trialAvg_avg_num_active_clusters_E, axis=0, nan_policy='omit') 

# grand average rate of active clsuters
trialAvg_popAvg_activeCluster_rate_e = np.nanmean(popAvg_activeCluster_rate_e, axis=(1,2))
netAvg_popAvg_activeCluster_rate_e = np.nanmean(trialAvg_popAvg_activeCluster_rate_e, axis=(0))
netStd_popAvg_activeCluster_rate_e = np.nanstd(trialAvg_popAvg_activeCluster_rate_e, axis=(0))

# grand average rate of inactive clsuters
trialAvg_popAvg_inactiveCluster_rate_e = np.nanmean(popAvg_inactiveCluster_rate_e, axis=(1,2))
netAvg_popAvg_inactiveCluster_rate_e = np.nanmean(trialAvg_popAvg_inactiveCluster_rate_e, axis=(0))
netStd_popAvg_inactiveCluster_rate_e = np.nanstd(trialAvg_popAvg_inactiveCluster_rate_e, axis=(0))

# grand average rate of background rate given X clusters active
trialAvg_avgRate_background_XActiveClusters_E = np.nanmean(avgRate_background_XActiveClusters_E, axis=(1,2))
netAvg_avgRate_background_XActiveClusters_E = np.nanmean(trialAvg_avgRate_background_XActiveClusters_E,axis=0)
netStd_avgRate_background_XActiveClusters_E = np.nanstd(trialAvg_avgRate_background_XActiveClusters_E,axis=0)

# grand average rate of active cluster rate given X clusters active
trialAvg_avgRate_active_XActiveClusters_E = np.nanmean(avgRate_active_XActiveClusters_E, axis=(1,2))
netAvg_avgRate_active_XActiveClusters_E = np.nanmean(trialAvg_avgRate_active_XActiveClusters_E,axis=0)
netStd_avgRate_active_XActiveClusters_E = np.nanstd(trialAvg_avgRate_active_XActiveClusters_E,axis=0)
    
# grand average rate of inactive cluster rate given X clusters active
trialAvg_avgRate_inactive_XActiveClusters_E = np.nanmean(avgRate_inactive_XActiveClusters_E, axis=(1,2))
netAvg_avgRate_inactive_XActiveClusters_E = np.nanmean(trialAvg_avgRate_inactive_XActiveClusters_E,axis=0)
netStd_avgRate_inactive_XActiveClusters_E = np.nanstd(trialAvg_avgRate_inactive_XActiveClusters_E,axis=0)
    

# grand average prob_nActive_clusters
trialAvg_prob_nActive_clusters_E = np.nanmean(prob_nActive_clusters_E, axis=(1,2))
netAvg_prob_nActive_clusters_E = np.nanmean(trialAvg_prob_nActive_clusters_E, axis=0)
netStd_prob_nActive_clusters_E = np.nanstd(trialAvg_prob_nActive_clusters_E, axis=0)
netSem_prob_nActive_clusters_E = scipy.stats.sem(trialAvg_prob_nActive_clusters_E, axis=0, nan_policy='omit')
    
#------------------------------------------------------------------------------
# SAVE RESULTS
#------------------------------------------------------------------------------
    
results = {}
parameters = {}

parameters['load_path'] = load_path
parameters['save_path'] = save_path
parameters['fig_path'] = fig_path
parameters['simID'] = simID
parameters['net_type'] = net_type
parameters['nNets'] = nNets
parameters['nTrials'] = nTrials
parameters['nStim'] = nStim
parameters['stim_shape']   =  stim_shape
parameters['stim_rel_amp'] = stim_rel_amp
parameters['sweep_param_str_val'] = sweep_param_str_val
parameters['burnTime_begin'] = burnTime_begin
parameters['burnTime_end'] = burnTime_end
parameters['window_std'] = window_std
parameters['window_step']  =  window_step
parameters['rate_thresh'] = rate_thresh_array

results['parameters'] = parameters

results['trialAvg_popAvg_rate_E'] = trialAvg_popAvg_rate_E
results['netAvg_popAvg_rate_E'] = netAvg_popAvg_rate_E
results['netStd_popAvg_rate_E'] = netStd_popAvg_rate_E
results['netSem_popAvg_rate_E'] = netSem_popAvg_rate_E

results['trialAvg_avg_num_active_clusters_E'] = trialAvg_avg_num_active_clusters_E
results['netAvg_avg_num_active_clusters_E'] = netAvg_avg_num_active_clusters_E
results['netStd_avg_num_active_clusters_E'] = netStd_avg_num_active_clusters_E
results['netSem_avg_num_active_clusters_E'] = netSem_avg_num_active_clusters_E

results['trialAvg_popAvg_activeCluster_rate_e'] = trialAvg_popAvg_activeCluster_rate_e
results['netAvg_popAvg_activeCluster_rate_e'] = netAvg_popAvg_activeCluster_rate_e
results['netStd_popAvg_activeCluster_rate_e'] = netStd_popAvg_activeCluster_rate_e

results['trialAvg_popAvg_inactiveCluster_rate_e'] = trialAvg_popAvg_inactiveCluster_rate_e
results['netAvg_popAvg_inactiveCluster_rate_e'] = netAvg_popAvg_inactiveCluster_rate_e
results['netStd_popAvg_inactiveCluster_rate_e'] = netStd_popAvg_inactiveCluster_rate_e

results['trialAvg_avgRate_background_XActiveClusters_E'] = trialAvg_avgRate_background_XActiveClusters_E
results['netAvg_avgRate_background_XActiveClusters_E'] = netAvg_avgRate_background_XActiveClusters_E
results['netStd_avgRate_background_XActiveClusters_E'] = netStd_avgRate_background_XActiveClusters_E

results['trialAvg_avgRate_active_XActiveClusters_E'] = trialAvg_avgRate_active_XActiveClusters_E
results['netAvg_avgRate_active_XActiveClusters_E'] = netAvg_avgRate_active_XActiveClusters_E
results['netStd_avgRate_active_XActiveClusters_E'] = netStd_avgRate_active_XActiveClusters_E

results['trialAvg_avgRate_inactive_XActiveClusters_E'] = trialAvg_avgRate_inactive_XActiveClusters_E
results['netAvg_avgRate_inactive_XActiveClusters_E'] = netAvg_avgRate_inactive_XActiveClusters_E
results['netStd_avgRate_inactive_XActiveClusters_E'] = netStd_avgRate_inactive_XActiveClusters_E


results['trialAvg_prob_nActive_clusters_E'] = trialAvg_prob_nActive_clusters_E
results['netAvg_prob_nActive_clusters_E'] = netAvg_prob_nActive_clusters_E
results['netStd_prob_nActive_clusters_E'] = netStd_prob_nActive_clusters_E
results['netSem_prob_nActive_clusters_E'] = netSem_prob_nActive_clusters_E

savemat(('%s%s%s_clusterRates_numActiveClusters_gainBased.mat' % (save_path, fname_begin, fname_end)), results)
