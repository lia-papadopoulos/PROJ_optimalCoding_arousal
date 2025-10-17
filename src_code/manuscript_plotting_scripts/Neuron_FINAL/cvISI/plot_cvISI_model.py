
'''
This script generates
    FigS6A
'''

#%% basic imports
import numpy as np
from scipy.io import loadmat
import sys
import os
import scipy.stats
import importlib

#%% import global settings file
sys.path.append('../../../')
import global_settings

#%% plotting
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_path = global_settings.path_to_plotting_font
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()
plt.rcParams["mathtext.default"]="regular"
plt.rcParams['axes.linewidth'] = 0.5

#%% settings

func_path = global_settings.path_to_src_code + 'functions/'
func_path0 = global_settings.path_to_src_code + 'run_simulations/'
sim_params_path = global_settings.path_to_src_code + 'run_simulations/'
sim_path = global_settings.path_to_sim_output
data_path = global_settings.path_to_sim_output + 'spont_cvISI_vsPerturbation/'   
fig_path = global_settings.path_to_manuscript_figs_final + 'cvISI_model/'
simParams_fname = 'simParams_051325_clu_spont'
sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'
net_type = 'baseEIclu'
nNetworks = 2
windL = 2500e-3
rate_thresh = 1.

figID = 'FigS6A'


#%% import custom function

sys.path.append(func_path)
sys.path.append(func_path0)
from fcn_simulation_loading import fcn_set_sweepParam_string
from fcn_simulation_setup import fcn_define_arousalSweep
from fcn_simulation_setup import fcn_compute_cluster_assignments
from fcn_compute_firing_stats import Dict2Class


#%% load sim parameters
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
arousal_level = s_params['arousal_levels']*100


del s_params
del params


#%% number of arousal levels
nArousal_vals = np.size(arousal_level)


#%% make figure directory
if os.path.isdir(fig_path) == False:
    os.makedirs(fig_path)
    
#%% set filenames

figname = ('%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f_windL%0.3f_rateThresh%0.1fHz' % \
          (simID, net_type, sweep_param_name, stim_shape, stim_rel_amp, windL, rate_thresh))

#%% check number of stimuli

if nStim > 1:
    sys.exit('code only works for one stimulus')
    
    
#%% load example simulation for initialization

# filename
sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, 0) 

sim_filename = ( sim_path + '%s_%s_sweep_%s_network%d_IC%d_stim%d_stimType_%s_stim_rel_amp%0.3f_simulationData.mat' % \
               ( simID, net_type, sweep_param_str, 0, 0, 0, stim_shape, stim_rel_amp) )
     
# load data
sim_data = loadmat(sim_filename, simplify_cells=True)                
s_params = Dict2Class(sim_data['sim_params'])
N = s_params.N_e + s_params.N_i
stimOn = s_params.stim_onset    
nClu = s_params.p
Ne = s_params.N_e


#%% get cells in stimulated clusters and cells that pass baseline rate cut

avg_units_allClusters = np.zeros((nNetworks), dtype='object')


for indNetwork in range(0, nNetworks):
    
    # cells with good baseline rate
    baseRate_allBaseMod = np.zeros((N, nArousal_vals))
            
    for indParam in range(0, nArousal_vals):
        
        sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, indParam) 
        
        
        filename = ('%s%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f_spont_cvISI_windL%0.3f.mat' % \
                   (data_path, simID, net_type, sweep_param_str, stim_shape, stim_rel_amp, windL) )

        data = loadmat(filename, simplify_cells=True)

                            
        # spike counts
        spkRate = data['avg_spkCount'][:, indNetwork]/windL
            
        # avg baseline rate
        baseRate_allBaseMod[:, indParam] = spkRate.copy()
            
            
    # cells with good firing rate
    good_rate_cells = np.nonzero(np.all( baseRate_allBaseMod >= rate_thresh, 1 ))[0]


    # get all clustered cells
    
    sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, 0) 

    sim_filename = ( sim_path + '%s_%s_sweep_%s_network%d_IC%d_stim%d_stimType_%s_stim_rel_amp%0.3f_simulationData.mat' % \
                   ( simID, net_type, sweep_param_str, indNetwork, 0, 0, stim_shape, stim_rel_amp) )
         
    # load data
    sim_data = loadmat(sim_filename, simplify_cells=True)                
    s_params = Dict2Class(sim_data['sim_params'])
    if 'popSize_E' in sim_data.keys():
        popSize_E = sim_data['popSize_E'].copy()
        popSize_I = sim_data['popSize_I'].copy()
    else:
        popSize_E = sim_data['clust_sizeE'].copy()
        popSize_I = sim_data['clust_sizeI'].copy()
        
        
    # cluster assignment of every cell
    Ecluster_ids, Icluster_ids = fcn_compute_cluster_assignments(popSize_E, popSize_I)

    # all cluster cells
    allCluster_cells_E = np.nonzero(Ecluster_ids < nClu)[0]
    allCluster_cells_I = np.nonzero(Icluster_ids < nClu)[0] + Ne
    allCluster_cells = np.append(allCluster_cells_E, allCluster_cells_I)
        
    avg_units_allClusters[indNetwork] = np.intersect1d(good_rate_cells, allCluster_cells).astype(int)


#%% initialize quantities

# cvISI
singleCell_cvISI_singleTrial  = np.ones((nNetworks,nArousal_vals), dtype='object')*np.nan
cellAvg_cvISI_singleTrial = np.ones((nNetworks, nArousal_vals))*np.nan
cellSem_cvISI_singleTrial = np.ones((nNetworks, nArousal_vals))*np.nan


singleCell_cvISI_trialAggregate  = np.ones((nNetworks,nArousal_vals), dtype='object')*np.nan
cellAvg_cvISI_trialAggregate = np.ones((nNetworks, nArousal_vals))*np.nan
cellSem_cvISI_trialAggregate = np.ones((nNetworks, nArousal_vals))*np.nan


for indNetwork in range(0, nNetworks):
    
    # good cells
    cells = avg_units_allClusters[indNetwork].copy()

    for indParam in range(0, nArousal_vals):
        
        sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, indParam) 
        
        filename = ('%s%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f_spont_cvISI_windL%0.3f.mat' % \
                   (data_path, simID, net_type, sweep_param_str, stim_shape, stim_rel_amp, windL) )
            
        data = loadmat(filename, simplify_cells=True)

        # unpack data
            
        cvISI_eachTrial = data['avg_cvISI_eachTrial'].copy()
        cvISI_trialAggregate = data['cvISI_trialAggregate'].copy()


        singleCell_cvISI_singleTrial[indNetwork, indParam] = cvISI_eachTrial[cells, indNetwork].copy()
        singleCell_cvISI_trialAggregate[indNetwork, indParam] = cvISI_trialAggregate[cells, indNetwork].copy()
        

        # cell avg
        cellAvg_cvISI_singleTrial[indNetwork, indParam] = np.nanmean(singleCell_cvISI_singleTrial[indNetwork, indParam], 0)
        cellSem_cvISI_singleTrial[indNetwork, indParam] = scipy.stats.sem(singleCell_cvISI_singleTrial[indNetwork, indParam], axis=0, nan_policy='omit') 
        
        cellAvg_cvISI_trialAggregate[indNetwork, indParam] = np.nanmean(singleCell_cvISI_trialAggregate[indNetwork, indParam], 0)
        cellSem_cvISI_trialAggregate[indNetwork, indParam] = scipy.stats.sem(singleCell_cvISI_trialAggregate[indNetwork, indParam], axis=0, nan_policy='omit') 


#%% average and sd across networks
netAvg_cvISI_singleTrial = np.mean(cellAvg_cvISI_singleTrial, 0)
netStd_cvISI_singleTrial = np.std(cellAvg_cvISI_singleTrial, 0)

netAvg_cvISI_trialAggregate = np.mean(cellAvg_cvISI_trialAggregate, 0)
netStd_cvISI_trialAggregate = np.std(cellAvg_cvISI_trialAggregate, 0)


#%% trial aggregate

plt.rcParams.update({'font.size': 7})
fig = plt.figure(figsize=(1.,1.))  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 

x = arousal_level.copy()
y = netAvg_cvISI_trialAggregate.copy()
yerr = netStd_cvISI_trialAggregate.copy()

ax.errorbar(x, y, yerr=yerr, xerr=None, color='k', linewidth=0.75, fmt='none')
ax.plot(x, y, '-',  color='k', markersize=1, linewidth=0.5)
ax.plot(x, y, 'o',  color=np.array([175,54,60])/256, markersize=1)

ax.set_xticks([0,100])
ax.set_xlim([-2,102])
ax.set_yticks([0.6, 2.6])

ax.set_xlabel('arousal level [%]')
ax.set_ylabel('$\\langle cvISI_{spont} \\rangle$')
    
plt.savefig('%s%s.pdf' % (fig_path, figID), bbox_inches='tight', pad_inches=0, transparent=True)





