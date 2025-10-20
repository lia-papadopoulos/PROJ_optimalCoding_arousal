

#%% BASIC IMPORTS

import sys
import numpy as np
import scipy.stats
from scipy.io import loadmat
from scipy.io import savemat
import importlib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

#%% SETTINGS 

import settings as settings

#%% FUNCTIONS

func_path = settings.func_path
func_path2 = settings.func_path2
                       
sys.path.append(func_path)
from fcn_compute_firing_stats import Dict2Class
from fcn_simulation_loading import fcn_set_sweepParam_string

sys.path.append(func_path2)
from fcn_simulation_setup import fcn_define_arousalSweep


#%% SETTINGS

load_from_simParams = settings.load_from_simParams
load_path = settings.load_path
save_path = settings.save_path
fig_path = settings.fig_path
    
nNets = settings.nNets
sweep_param_name = settings.sweep_param_name
net_type = settings.net_type  
startTime_base = settings.startTime_base
sig_level = settings.sig_level


if load_from_simParams == True:
    simParams_path = settings.simParams_path
    simParams_fname = settings.simParams_fname
else:
    simID = settings.simID
    nTrials = settings.nTrials
    stim_shape = settings.stim_shape
    stim_type = settings.stim_type
    stim_rel_amp = settings.stim_rel_amp
    n_sweepParams = settings.n_sweepParams
    swept_params_dict = settings.swept_params_dict


#%% SIM PARAMS

if load_from_simParams == True:

    sys.path.append(simParams_path)
    s_params_data = importlib.import_module(simParams_fname) 
    s_params = s_params_data.sim_params

    # arousal sweep
    s_params = fcn_define_arousalSweep(s_params)

    # unpack    
    simID = s_params['simID']
    nTrials = s_params['n_ICs']
    stim_shape = s_params['stim_shape']
    stim_type = s_params['stim_type']
    stim_rel_amp = s_params['stim_rel_amp']
    n_sweepParams = s_params['nParams_sweep']
    swept_params_dict = s_params['swept_params_dict']
    arousal_level = s_params['arousal_levels']

    del s_params_data
    del s_params
    
else:
    
    arousal_level = swept_params_dict['param_vals1']/np.max(swept_params_dict['param_vals1'])
    print('assumming linear arousal model')


#%% NUMBER OF AROUSAL SAMPLES

n_arousalSamples = np.size(arousal_level)

#%% FILENAMES

fname_begin = ( '%s%s_%s_sweep_%s_network%d_IC%d_stim%d_stimType_%s_stim_rel_amp%0.3f_simulationData.mat' )

#%% LOAD ONE SIMULATION TO SET EVERYTHING UP

sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, 0) 

# filename
params_tuple = (load_path, simID, net_type, sweep_param_str, 0, 0, 0, stim_shape, stim_rel_amp)
filename = ( (fname_begin) % (params_tuple) )

# load data
data = loadmat(filename, simplify_cells=True)   
   
# sim_params        
 
s_params = Dict2Class(data['sim_params'])
# spikes

spikes = data['spikes']

# simulation parameters
T0 = s_params.T0
N = s_params.N
stimOn = s_params.stim_onset
nStim = s_params.nStim
nClu = s_params.p

# end of baseline period
endTime_base = stimOn


#%% COMPUTE RATE OF EACH CELL

firingRate_cells_base = np.ones((N, n_arousalSamples, nNets, nTrials, nStim))*np.nan


for indParam in range(0,n_arousalSamples):
    
    for indNet in range(0,nNets):
    
        for indTrial in range(0,nTrials):
            
            for indStim in range(0,nStim):
                
                sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, indParam) 

                # filename
                params_tuple = (load_path, simID, net_type, sweep_param_str, indNet, indTrial, indStim, stim_shape, stim_rel_amp)
                filename = ( (fname_begin) % (params_tuple) )
                         
                # load data
                data = loadmat(filename, simplify_cells=True)                
                s_params = Dict2Class(data['sim_params'])
                
                # spikes
                spikes = data['spikes']
    
                
                # firing rates
                
                # find spikes within baseline time window
                tInds = np.nonzero( (spikes[0,:] <= endTime_base) & (spikes[0,:] >= startTime_base))[0]
                spikes_window = spikes[:,tInds]

                # rate
                neuron_IDs = spikes_window[1,:].astype(int)
                spike_times = spikes_window[0,:]
                spike_cnts = np.zeros(N)      
                
                # loop over spikes array and record spikes
                for i in range(0,len(spike_times),1):
                    
                    t = spike_times[i]
                    n = neuron_IDs[i]
                    spike_cnts[n] += 1
                
                firing_rates = spike_cnts/(endTime_base-startTime_base)
                
                firingRate_cells_base[:, indParam, indNet, indTrial, indStim] = firing_rates.copy()
                               
            
        print(indNet)
        
    print(indParam)
    

#%% average firing rate over trials and stimuli
### since we are considering baseline activity, different stimuli essentially correspond to different trials (we use a random initial condition for all trials)

# average baseline rate over trials and stimuli
avg_firingRate_cells_base = np.mean(np.mean(firingRate_cells_base, 4), 3) # (N, Npert, nNets)

# pop avg baseline rate
popAvg_trialAvg_firingRate_cells_base = np.mean(avg_firingRate_cells_base, 0) # (Npert, nNets)
netAvg_popAvg_trialAvg_firingRate_cells_base = np.mean(popAvg_trialAvg_firingRate_cells_base, 1)
netSd_popAvg_trialAvg_firingRate_cells_base = np.std(popAvg_trialAvg_firingRate_cells_base, 1)

# sd baseline rate
popSd_trialAvg_firingRate_cells_base = np.std(avg_firingRate_cells_base, 0)
netAvg_popSd_trialAvg_firingRate_cells_base = np.mean(popSd_trialAvg_firingRate_cells_base, 1)
netSd_popSd_trialAvg_firingRate_cells_base = np.std(popSd_trialAvg_firingRate_cells_base, 1)


#%% correlation between average firing rate and perturbation strength for each cell and each network
corr_pert_rate_base = np.ones((N, nNets))*np.nan
pval_pert_rate_base = np.ones((N, nNets))*np.nan

for indCell in range(0, N):
    
    for indNet in range(0, nNets):
    
         r, p = scipy.stats.spearmanr(avg_firingRate_cells_base[indCell, :, indNet], arousal_level)

         corr_pert_rate_base[indCell, indNet] = r
         pval_pert_rate_base[indCell, indNet] = p
   
#%% fraction positively and negatively correlated
frac_posCorr_pert_rate = np.zeros(nNets)
frac_negCorr_pert_rate = np.zeros(nNets)

for indNet in range(0, nNets):

    frac_posCorr_pert_rate[indNet] = np.size(np.nonzero( (pval_pert_rate_base[:, indNet] < sig_level) & (corr_pert_rate_base[:, indNet] > 0)  )[0])/(N)
    frac_negCorr_pert_rate[indNet] = np.size(np.nonzero( (pval_pert_rate_base[:, indNet] < sig_level) & (corr_pert_rate_base[:, indNet] < 0)  )[0])/(N)

    
#%% SAVE THE RESULTS

params = {}
results = {}

params['simID'] = simID
params['net_type'] = net_type
params['nNets'] = nNets
params['nTrials'] = nTrials
params['stim_shape'] = stim_shape
params['stim_rel_amp'] = stim_rel_amp
params['n_sweepParams'] = n_sweepParams
params['sweep_param_name'] = sweep_param_name
params['swept_params_dict'] = swept_params_dict
params['arousal_level'] = arousal_level
params['sig_level'] = sig_level

params['startTime_base'] = startTime_base
params['endTime_base'] = endTime_base

results['params'] = params
results['trialAvg_stimAvg_firingRate_cells_base'] = avg_firingRate_cells_base
results['corr_pert_rate_base'] = corr_pert_rate_base 
results['pval_pert_rate_base'] = pval_pert_rate_base
results['popAvg_trialAvg_stimAvg_firingRate_cells_base'] = popAvg_trialAvg_firingRate_cells_base
results['netAvg_popAvg_trialAvg_stimAvg_firingRate_cells_base'] = netAvg_popAvg_trialAvg_firingRate_cells_base
results['netSd_popAvg_trialAvg_stimAvg_firingRate_cells_base'] = netSd_popAvg_trialAvg_firingRate_cells_base
results['popSdtrialAvg_stimAvg_firingRate_cells_base'] = popSd_trialAvg_firingRate_cells_base
results['netAvg_popSd_trialAvg_stimAvg_firingRate_cells_base'] = netAvg_popSd_trialAvg_firingRate_cells_base
results['netSd_popSd_trialAvg_stimAvg_firingRate_cells_base'] = netSd_popSd_trialAvg_firingRate_cells_base


save_name = ('%s%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f_singleCell_tuning_to_perturbation.mat' % \
             (save_path, simID, net_type, sweep_param_name, stim_shape, stim_rel_amp) )
    
savemat(save_name, results)


#%% PLOTTING

fig_name = ( ('%s%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f') % (fig_path, simID, net_type, sweep_param_name, stim_shape, stim_rel_amp) )

         
#%% plot example of positive and negatively modulated cell

indNet = 0

posCell = np.nonzero( (pval_pert_rate_base[:, indNet] < 0.05) & (corr_pert_rate_base[:, indNet] > 0)  )[0][210]
negCell = np.nonzero( (pval_pert_rate_base[:, indNet] < 0.05) & (corr_pert_rate_base[:, indNet] < 0)  )[0][210]

avgRate_posCell = avg_firingRate_cells_base[posCell, :, indNet].copy()
avgRate_negCell = avg_firingRate_cells_base[negCell, :, indNet].copy()


plt.figure()
plt.rcParams['font.size'] = 16
x = arousal_level 
y1 = avgRate_posCell
y2 = avgRate_negCell

plt.plot(x, y1, color='b')
plt.plot(x, y2, color='r')
plt.xlabel('arousal [pct]')
plt.ylabel('firing rate')
plt.tight_layout()
plt.savefig( (fig_name + '_posModCell_negModCell.png') )    



#%% plot distribution of correlations across cells for each network

fig, axs = plt.subplots(1, nNets, figsize=(12, 3))
for indNet in range(0, nNets):
    
    if nNets > 1:
        ax = axs.flat[indNet]
    else:
        ax = axs
    ax.hist(corr_pert_rate_base[:, indNet], np.arange(-1.05,1.25,0.2), density=False)
    ax.set_xlim([-1.2, 1.2])
    ax.set_xlabel('Rs: rate vs %s' % sweep_param_name)
    ax.set_ylabel('count')
    ax.set_title('network %d' % indNet)
plt.tight_layout()
plt.savefig( (fig_name + '_corr_cellRate.png') )    


#%% plot distribution of correlations across cells and networks combined


plt.figure(figsize=(5.5,4))
plt.rcParams['font.size'] = 20
plt.hist(corr_pert_rate_base.flatten(), np.arange(-1.05,1.25,0.2), facecolor='gray', density=False)
plt.xlim([-1.2, 1.2])
plt.xlabel('correlation')
plt.ylabel('count')    
plt.tight_layout()
plt.savefig( (fig_name + '_corr_cellRate_allNetworks.png') )    


#%% plot network average of fraction pos/neg corr cells


plt.figure(figsize=(4,5))
plt.rcParams['font.size'] = 20
y = np.mean(frac_posCorr_pert_rate)
yerr = np.std(frac_posCorr_pert_rate)
plt.bar(0, y, yerr=yerr, color='blue', label='pos.')

y = -np.mean(frac_negCorr_pert_rate)
yerr = np.std(frac_negCorr_pert_rate)
plt.bar(0, y, yerr=yerr, color='red', label='neg.')

plt.ylabel('frac +/- correlation')
plt.xlim([-0.5, 0.5])
plt.ylim([-1.1, 1.1])
plt.xticks([])
plt.yticks([-1, -0.5, 0, 0.5, 1], [1, 0.5, 0, 0.5, 1])
plt.tight_layout()
plt.savefig( (fig_name + '_frac_pos_neg_corr_allNetworks.png') )    


#%% plot population average and standard deviation of firing rate distribution vs perturbation strength

plt.figure()
plt.rcParams['font.size'] = 16
x = arousal_level 
y = netAvg_popAvg_trialAvg_firingRate_cells_base
yerr = netSd_popAvg_trialAvg_firingRate_cells_base
plt.fill_between(x, y+yerr, y-yerr, where=None, color='r', alpha=0.3)
plt.xlabel('%s [pct]' % sweep_param_name)
plt.ylabel('pop avg baseline rate [spks/s]')
plt.tight_layout()
plt.savefig( (fig_name + '_popAvg_baseline_cellRate.png') )    


plt.figure()
plt.rcParams['font.size'] = 16
x = arousal_level 
y = netAvg_popSd_trialAvg_firingRate_cells_base
yerr = netSd_popSd_trialAvg_firingRate_cells_base
plt.fill_between(x, y+yerr, y-yerr, where=None, color='m', alpha=0.3)
plt.xlabel('%s [pct]' % sweep_param_name)
plt.ylabel('pop sd baseline rate [spks/s]')
plt.tight_layout()
plt.savefig( (fig_name + '_popSD_baseline_cellRate.png') )    


plt.close('all')
