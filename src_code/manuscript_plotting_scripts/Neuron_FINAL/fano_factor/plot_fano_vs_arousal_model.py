
'''
This script generates different figure panels depending on the values of certain settings

    figPlot = cluster_mainArousal
    windL = 100e-3
        Fig8A-C
        
    figPlot = cluster_mainArousal
    windL = 50e-3, 100e-3, 200e-3
        FigS7H-J
        
    figPlot = cluster_suppArousal
    windL = 100e-3
        FigS3H    

'''


#%% basic imports
import sys
import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
import importlib
import os

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

#%% load my functions
func_path0 = global_settings.path_to_src_code + 'run_simulations/'
func_path1 = global_settings.path_to_src_code + 'functions/'
func_path2 = global_settings.path_to_src_code + 'simulations_analysis/fano_factor/'

sys.path.append(func_path0)
sys.path.append(func_path1)
sys.path.append(func_path2)

from fcn_simulation_loading import fcn_set_sweepParam_string
from fcn_simulation_setup import fcn_define_arousalSweep
from fcn_simulation_setup import fcn_compute_cluster_assignments
from fcn_compute_firing_stats import Dict2Class
import fcn_plot_fanofactor

#%% settings

# paths
sim_params_path = global_settings.path_to_src_code + 'run_simulations/'
sim_path = global_settings.path_to_sim_output
data_path = global_settings.path_to_sim_output + 'fanofactor/'

windL = 100e-3
rate_thresh = 1.   
burnTime = 0.2
evoked_window_length = 1.
t_eval_FFevoked = 'min_allStim'
param_name_plot = 'arousal level [%]'
param_name_plot_short = 'arousal level [%]'
fano_filename = 'FanofactorRaw_timecourse'

figPlot = 'cluster_altArousal'


if figPlot == 'cluster_mainArousal':
    simParams_fname = 'simParams_051325_clu'
    sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'
    nNetworks = 10
    net_type = 'baseEIclu'
    figureSize = (1., 1.)
    fontSize = 7
    linewidth = 0.5
    markersize = 1
    fig_path = global_settings.path_to_manuscript_figs_final + 'fanofactor_vs_arousal_model/cluster_mainArousal/'
    savename = 'fanofactor_model_cluster_windL%0.3fs_' % windL
    
    figID1M = 'Fig8A'
    figID2 = 'Fig8B'
    figID3 = 'Fig8C'    
    
    if windL == 50e-3:
        figID1S = 'FigS7H'
    if windL == 100e-3:
        figID1S = 'FigS7I'
    if windL == 200e-3:
        figID1S = 'FigS7J'
    
    
elif figPlot == 'cluster_altArousal':
    simParams_fname = 'simParams_050925_clu'
    sweep_param_name = 'zeroMean_sd_nu_ext_ee'
    nNetworks = 5
    net_type = 'baseEIclu'   
    figureSize = (1.35, 1.35)
    fontSize = 8
    linewidth = 1
    markersize = 2
    fig_path = global_settings.path_to_manuscript_figs_final + 'fanofactor_vs_arousal_model/cluster_altArousal/'
    figID = 'FigS3H'
    savename = 'fanofactor_model_cluster_altArousal_windL%0.3fs_' % windL
    
else:
    sys.exit('invalid figPlot')
    


#%% make output directory
if os.path.isdir(fig_path) == False:
    os.makedirs(fig_path)
    
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

#%% set filenames

figname = ('%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f_windL%0.3f_rateThresh%0.1fHz' % \
          (simID, net_type, sweep_param_name, stim_shape, stim_rel_amp, windL, rate_thresh))
    
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


#%% load one example fano for initialization

# fano data
filename = ('%s%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f_%s_windL%0.3f.mat' % (data_path, simID, net_type, sweep_param_str, stim_shape, stim_rel_amp, fano_filename, windL) )
fano_data = loadmat(filename, simplify_cells=True)

# time window
t_window = fano_data['t_window']

# baseline tInds
base_inds = np.nonzero( (t_window >= burnTime + windL) & (t_window <= stimOn) )[0]

# evoked tInds
evoked_inds = np.nonzero( (t_window > stimOn) & (t_window <= stimOn + evoked_window_length) )[0]


#%% get cells in stimulated clusters and cells that pass baseline rate cut

avg_units_stimClusters_E = np.zeros((nNetworks, nStim), dtype='object')
avg_units_stimClusters_EI = np.zeros((nNetworks, nStim), dtype='object')
avg_units_allClusters = np.zeros((nNetworks, nStim), dtype='object')

for indNetwork in range(0, nNetworks):
    
    # cells with good baseline rate
    baseRate_allBaseMod = np.zeros((N, nArousal_vals))
                
            
    for indParam in range(0, nArousal_vals):
        
        sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, indParam) 

        filename = ('%s%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f_%s_windL%0.3f.mat' % \
                    (data_path, simID, net_type, sweep_param_str, stim_shape, stim_rel_amp, fano_filename, windL) )

        fano_data = loadmat(filename, simplify_cells=True)
            
                
        if ( (nStim == 1) and (nNetworks == 1) ):
            
            # spike counts
            spkRate = fano_data['avg_spkCount']/windL 
            
            # average baseline rate
            baseRate_allBaseMod[:, indParam] = np.mean(spkRate[:, base_inds], axis=(1))

            
        elif ( (nStim == 1) and (nNetworks !=1 ) ):
            
            # spike counts
            spkRate = fano_data['avg_spkCount'][:, indNetwork, :]/windL
            
            # avg baseline rate
            baseRate_allBaseMod[:, indParam] = np.mean(spkRate[:, base_inds], 1)
            
        elif  ( (nNetworks == 1) and (nStim != 1) ):
            
            # spike counts
            spkRate = fano_data['avg_spkCount']/windL
            
            # avg baseline rate
            baseRate_allBaseMod[:, indParam] = np.mean(spkRate[:, :, base_inds], axis=(1,2))

        else:
            
            # spike counts
            spkRate = fano_data['avg_spkCount'][:, :, indNetwork, :]/windL
            
            # avg baseline rate
            baseRate_allBaseMod[:, indParam] = np.mean(spkRate[:, :, base_inds], axis=(1,2))
            

    good_rate_cells = np.nonzero(np.all( baseRate_allBaseMod >= rate_thresh, 1 ))[0]
    
    
    # cells in stimulated clsuters
    for indStim in range(0, nStim):
    
        # filename
        sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, 0) 

        sim_filename = ( sim_path + '%s_%s_sweep_%s_network%d_IC%d_stim%d_stimType_%s_stim_rel_amp%0.3f_simulationData.mat' % \
                       ( simID, net_type, sweep_param_str, indNetwork, 0, indStim, stim_shape, stim_rel_amp) )
             
        # load data
        sim_data = loadmat(sim_filename, simplify_cells=True)                
        s_params = Dict2Class(sim_data['sim_params'])
        popSize_E = sim_data['popSize_E'].copy()
        popSize_I = sim_data['popSize_I'].copy()

        
        # cluster assignment of every cell
        Ecluster_ids, Icluster_ids = fcn_compute_cluster_assignments(popSize_E, popSize_I)

        # all cluster cells
        allCluster_cells_E = np.nonzero(Ecluster_ids < nClu)[0]
        allCluster_cells_I = np.nonzero(Icluster_ids < nClu)[0] + Ne
        allCluster_cells = np.append(allCluster_cells_E, allCluster_cells_I)
        
        
        # stimulated E clusters
        stimClusters_E = s_params.selectiveClusters.copy()

        # cells in stimulated clusters
        stim_cluster_cells_E = np.array([])
        stim_cluster_cells_I = np.array([])
        
        for iClu, clu in enumerate(stimClusters_E):
            
            stim_cluster_cells_E = np.append(stim_cluster_cells_E, np.nonzero(Ecluster_ids == clu)[0])
            stim_cluster_cells_I = np.append(stim_cluster_cells_I, (Ne + np.nonzero(Icluster_ids == clu)[0]))
            
        stim_cluster_cells_EI = np.append(stim_cluster_cells_E, stim_cluster_cells_I)
    
        # cells to average over
        avg_units_stimClusters_E[indNetwork, indStim] = np.intersect1d(good_rate_cells, stim_cluster_cells_E).astype(int)
        avg_units_stimClusters_EI[indNetwork, indStim] = np.intersect1d(good_rate_cells, stim_cluster_cells_EI).astype(int)
        avg_units_allClusters[indNetwork, indStim] = np.intersect1d(good_rate_cells, allCluster_cells).astype(int)

print('computed units to average over')

#%% initialize ff quantities

stimAvg_spontFF_stimE = np.nan*np.ones((nNetworks, nArousal_vals), dtype='object')
cellAvg_stimAvg_spontFF_stimE = np.nan*np.ones((nNetworks, nArousal_vals))
stimAvg_evokedFF_stimE = np.nan*np.ones((nNetworks, nArousal_vals), dtype='object')
cellAvg_stimAvg_evokedFF_stimE = np.nan*np.ones((nNetworks, nArousal_vals))
stimAvg_diffFF_stimE = np.nan*np.ones((nNetworks, nArousal_vals), dtype='object')
cellAvg_stimAvg_diffFF_stimE = np.nan*np.ones((nNetworks, nArousal_vals))


stimAvg_spontFF_stimEI = np.nan*np.ones((nNetworks, nArousal_vals), dtype='object')
cellAvg_stimAvg_spontFF_stimEI = np.nan*np.ones((nNetworks, nArousal_vals))
stimAvg_evokedFF_stimEI = np.nan*np.ones((nNetworks, nArousal_vals), dtype='object')
cellAvg_stimAvg_evokedFF_stimEI = np.nan*np.ones((nNetworks, nArousal_vals))
stimAvg_diffFF_stimEI = np.nan*np.ones((nNetworks, nArousal_vals), dtype='object')
cellAvg_stimAvg_diffFF_stimEI = np.nan*np.ones((nNetworks, nArousal_vals))


stimAvg_spontFF_allCluster = np.nan*np.ones((nNetworks, nArousal_vals), dtype='object')
cellAvg_stimAvg_spontFF_allCluster = np.nan*np.ones((nNetworks, nArousal_vals))
stimAvg_evokedFF_allCluster = np.nan*np.ones((nNetworks, nArousal_vals), dtype='object')
cellAvg_stimAvg_evokedFF_allCluster = np.nan*np.ones((nNetworks, nArousal_vals))
stimAvg_diffFF_allCluster = np.nan*np.ones((nNetworks, nArousal_vals), dtype='object')
cellAvg_stimAvg_diffFF_allCluster = np.nan*np.ones((nNetworks, nArousal_vals))


for indNetwork in range(0, nNetworks):
    

    for indParam in range(0, nArousal_vals):
    
        sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, indParam) 

        filename = ('%s%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f_%s_windL%0.3f.mat' % \
                        (data_path, simID, net_type, sweep_param_str, stim_shape, stim_rel_amp, fano_filename, windL) )

        data = loadmat(filename, simplify_cells=True)
        

        if ( (nStim == 1) ):
                        
            if nNetworks == 1:
            
                fano = data['fanofactor'].copy()
                
            else:
                    
                fano = data['fanofactor'][:, indNetwork, :].copy()            
                
            # fano times
            t_fano = data['t_window'].copy()
            t_fano_evoked = t_fano[evoked_inds].copy()
            
            
            # spont fano [window before stimulus onset]
            spont_fano = fano[:, base_inds[-1]].copy()

            # evoked fano timecourse
            evoked_fano = fano[:, evoked_inds].copy()
            
            # diff fano timecourse
            diff_fano = np.ones((N, len(t_window)))*np.nan
            for i in range(0, N):
                diff_fano[i, :] =  spont_fano[i] - fano[i,:]
            diff_fano = diff_fano[:, evoked_inds].copy()  

                
            # compute ff quantities for different cell subsets
            
            # stim E clusters
            stimAvg_spontFF_stimE[indNetwork, indParam], \
            stimAvg_evokedFF_stimE[indNetwork, indParam], \
            stimAvg_diffFF_stimE[indNetwork, indParam] = \
                fcn_plot_fanofactor.fcn_compute_fano_cellSubset_singleStim(spont_fano, evoked_fano, diff_fano, avg_units_stimClusters_E[indNetwork, 0],tavg = t_eval_FFevoked)

            cellAvg_stimAvg_spontFF_stimE[indNetwork, indParam] = np.nanmean(stimAvg_spontFF_stimE[indNetwork, indParam])    
            cellAvg_stimAvg_evokedFF_stimE[indNetwork, indParam] = np.nanmean(stimAvg_evokedFF_stimE[indNetwork, indParam])    
            cellAvg_stimAvg_diffFF_stimE[indNetwork, indParam] = np.nanmean(stimAvg_diffFF_stimE[indNetwork, indParam])    
        
            # stim E I clusters
            stimAvg_spontFF_stimEI[indNetwork, indParam], \
            stimAvg_evokedFF_stimEI[indNetwork, indParam], \
            stimAvg_diffFF_stimEI[indNetwork, indParam] = \
                fcn_plot_fanofactor.fcn_compute_fano_cellSubset_singleStim(spont_fano, evoked_fano, diff_fano, avg_units_stimClusters_EI[indNetwork, 0],tavg = t_eval_FFevoked)

            cellAvg_stimAvg_spontFF_stimEI[indNetwork, indParam] = np.nanmean(stimAvg_spontFF_stimEI[indNetwork, indParam])    
            cellAvg_stimAvg_evokedFF_stimEI[indNetwork, indParam] = np.nanmean(stimAvg_evokedFF_stimEI[indNetwork, indParam])    
            cellAvg_stimAvg_diffFF_stimEI[indNetwork, indParam] = np.nanmean(stimAvg_diffFF_stimEI[indNetwork, indParam])    
        
        
            # all cluster cells
            stimAvg_spontFF_allCluster[indNetwork, indParam], \
            stimAvg_evokedFF_allCluster[indNetwork, indParam], \
            stimAvg_diffFF_allCluster[indNetwork, indParam] = \
                fcn_plot_fanofactor.fcn_compute_fano_cellSubset_singleStim(spont_fano, evoked_fano, diff_fano, avg_units_allClusters[indNetwork, 0],tavg = t_eval_FFevoked)

            cellAvg_stimAvg_spontFF_allCluster[indNetwork, indParam] = np.nanmean(stimAvg_spontFF_allCluster[indNetwork, indParam])    
            cellAvg_stimAvg_evokedFF_allCluster[indNetwork, indParam] = np.nanmean(stimAvg_evokedFF_allCluster[indNetwork, indParam])    
            cellAvg_stimAvg_diffFF_allCluster[indNetwork, indParam] = np.nanmean(stimAvg_diffFF_allCluster[indNetwork, indParam])    
        

        else:
            
            if nNetworks == 1:
                fano = data['fanofactor'].copy()
                
            else:
                fano = data['fanofactor'][:, :, indNetwork, :].copy()
                
            # fano times
            t_fano = data['t_window'].copy()
            t_fano_evoked = t_fano[evoked_inds].copy()
            
            # spont fano
            spont_fano = fano[:, :, base_inds[-1]].copy()

            # evoked fano timecourse
            evoked_fano = fano[:, :, evoked_inds].copy()
        
            # diff fano timecourse
            diff_fano = np.ones((N, nStim, len(evoked_inds)))*np.nan        
            for indStim in range(0, nStim):                
                for i in range(0, N):            
                    diff_fano[i, indStim, :] =  spont_fano[i, indStim] - fano[i, indStim, evoked_inds]

                

            # compute ff quantities for different cell subsets
            
            # stim E clusters
            stimAvg_spontFF_stimE[indNetwork, indParam], \
            stimAvg_evokedFF_stimE[indNetwork, indParam], \
            stimAvg_diffFF_stimE[indNetwork, indParam] = \
                fcn_plot_fanofactor.fcn_compute_fano_cellSubset_multipleStim(spont_fano, evoked_fano, diff_fano, avg_units_stimClusters_E[indNetwork, :], tavg = t_eval_FFevoked)

            cellAvg_stimAvg_spontFF_stimE[indNetwork, indParam] = np.nanmean(stimAvg_spontFF_stimE[indNetwork, indParam])    
            cellAvg_stimAvg_evokedFF_stimE[indNetwork, indParam] = np.nanmean(stimAvg_evokedFF_stimE[indNetwork, indParam])    
            cellAvg_stimAvg_diffFF_stimE[indNetwork, indParam] = np.nanmean(stimAvg_diffFF_stimE[indNetwork, indParam])    
        
        
            # stim E I clusters
            stimAvg_spontFF_stimEI[indNetwork, indParam], \
            stimAvg_evokedFF_stimEI[indNetwork, indParam], \
            stimAvg_diffFF_stimEI[indNetwork, indParam] = \
                fcn_plot_fanofactor.fcn_compute_fano_cellSubset_multipleStim(spont_fano, evoked_fano, diff_fano, avg_units_stimClusters_EI[indNetwork, :], tavg = t_eval_FFevoked)

            cellAvg_stimAvg_spontFF_stimEI[indNetwork, indParam] = np.nanmean(stimAvg_spontFF_stimEI[indNetwork, indParam])    
            cellAvg_stimAvg_evokedFF_stimEI[indNetwork, indParam] = np.nanmean(stimAvg_evokedFF_stimEI[indNetwork, indParam])    
            cellAvg_stimAvg_diffFF_stimEI[indNetwork, indParam] = np.nanmean(stimAvg_diffFF_stimEI[indNetwork, indParam])    
        
        
            # all cluster cells
            stimAvg_spontFF_allCluster[indNetwork, indParam], \
            stimAvg_evokedFF_allCluster[indNetwork, indParam], \
            stimAvg_diffFF_allCluster[indNetwork, indParam] = \
                fcn_plot_fanofactor.fcn_compute_fano_cellSubset_multipleStim(spont_fano, evoked_fano, diff_fano, avg_units_allClusters[indNetwork, :], tavg = t_eval_FFevoked)

            cellAvg_stimAvg_spontFF_allCluster[indNetwork, indParam] = np.nanmean(stimAvg_spontFF_allCluster[indNetwork, indParam])    
            cellAvg_stimAvg_evokedFF_allCluster[indNetwork, indParam] = np.nanmean(stimAvg_evokedFF_allCluster[indNetwork, indParam])    
            cellAvg_stimAvg_diffFF_allCluster[indNetwork, indParam] = np.nanmean(stimAvg_diffFF_allCluster[indNetwork, indParam])    


#%% average across networks

netAvg_cellAvg_spontFF_stimE = np.nanmean(cellAvg_stimAvg_spontFF_stimE, axis=0)
netStd_cellAvg_spontFF_stimE = np.nanstd(cellAvg_stimAvg_spontFF_stimE, axis=0)
netAvg_cellAvg_evokedFF_stimE  = np.nanmean(cellAvg_stimAvg_evokedFF_stimE , axis=0)
netStd_cellAvg_evokedFF_stimE  = np.nanstd(cellAvg_stimAvg_evokedFF_stimE, axis=0)
netAvg_cellAvg_diffFF_stimE  = np.nanmean(cellAvg_stimAvg_diffFF_stimE , axis=0)
netStd_cellAvg_diffFF_stimE  = np.nanstd(cellAvg_stimAvg_diffFF_stimE , axis=0)


netAvg_cellAvg_spontFF_stimEI = np.nanmean(cellAvg_stimAvg_spontFF_stimEI, axis=0)
netStd_cellAvg_spontFF_stimEI = np.nanstd(cellAvg_stimAvg_spontFF_stimEI, axis=0)
netAvg_cellAvg_evokedFF_stimEI  = np.nanmean(cellAvg_stimAvg_evokedFF_stimEI , axis=0)
netStd_cellAvg_evokedFF_stimEI  = np.nanstd(cellAvg_stimAvg_evokedFF_stimEI, axis=0)
netAvg_cellAvg_diffFF_stimEI  = np.nanmean(cellAvg_stimAvg_diffFF_stimEI, axis=0)
netStd_cellAvg_diffFF_stimEI  = np.nanstd(cellAvg_stimAvg_diffFF_stimEI , axis=0)


netAvg_cellAvg_spontFF_allCluster = np.nanmean(cellAvg_stimAvg_spontFF_allCluster, axis=0)
netStd_cellAvg_spontFF_allCluster = np.nanstd(cellAvg_stimAvg_spontFF_allCluster, axis=0)
netAvg_cellAvg_evokedFF_allCluster  = np.nanmean(cellAvg_stimAvg_evokedFF_allCluster , axis=0)
netStd_cellAvg_evokedFF_allCluster  = np.nanstd(cellAvg_stimAvg_evokedFF_allCluster, axis=0)
netAvg_cellAvg_diffFF_allCluster  = np.nanmean(cellAvg_stimAvg_diffFF_allCluster, axis=0)
netStd_cellAvg_diffFF_allCluster  = np.nanstd(cellAvg_stimAvg_diffFF_allCluster , axis=0)



#%% save the results

params = {}
results = {}

params['fano_filename'] = fano_filename
params['windL'] = windL
params['burnTime'] = burnTime
params['evoked_window_length'] = evoked_window_length
params['rate_thresh'] = rate_thresh
params['nNetworks'] = nNetworks
params['nStim'] = nStim
params['stim_rel_amp'] = stim_rel_amp
params['stim_shape'] = stim_shape
params['nTrials'] = nTrials
params['net_type'] = net_type
params['simID'] = simID
params['data_path'] = data_path
params['sim_params_path'] = sim_params_path
params['sim_path'] = sim_path
params['fig_path'] = fig_path
params['sweep_param_name'] = sweep_param_name
params['swept_params_dict'] = swept_params_dict
params['t_eval_FFevoked'] = t_eval_FFevoked
params['simParams_fname'] = simParams_fname
results['params'] = params

save_filename = (fig_path + savename + 'params.mat')      
savemat(save_filename, results) 


#%% FANO FACTOR VS PERTURBATION STRENGTH


if ( (figPlot == 'cluster_altArousal') and (windL == 100e-3) ):
    

    plt.rcParams.update({'font.size': fontSize})
    fig = plt.figure(figsize=figureSize)  
    
    ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 
    x = arousal_level
    y = netAvg_cellAvg_spontFF_stimEI
    yerr = netStd_cellAvg_spontFF_stimEI
    ax.errorbar(x, y, yerr=yerr, xerr=None, color='k', linewidth=linewidth, fmt='none')
    ax.plot(x, y, '-',  color='k', linewidth=linewidth)
    ax.plot(x, y, 'o',  color=np.array([175,54,60])/256, markersize=markersize)   
    ax.set_xticks([0,100])
    ax.set_xlim([-2,102])
    
    if windL == 100e-3:
        ax.set_yticks([0.5, 1.75, 3.0])
    
    ax.set_xlabel('arousal level [%]')
    ax.set_ylabel('$\\langle FF_{spont} \\rangle$')
    plt.savefig('%s%s.pdf' % (fig_path, figID), bbox_inches='tight', pad_inches=0, transparent=True)
    


if figPlot == 'cluster_mainArousal':

    plt.rcParams.update({'font.size': fontSize})
    fig = plt.figure(figsize=figureSize)  
    
    ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 
    x = arousal_level
    y = netAvg_cellAvg_spontFF_stimEI
    yerr = netStd_cellAvg_spontFF_stimEI
    ax.errorbar(x, y, yerr=yerr, xerr=None, color='k', linewidth=linewidth, fmt='none')
    ax.plot(x, y, '-',  color='k', linewidth=linewidth)
    ax.plot(x, y, 'o',  color=np.array([175,54,60])/256, markersize=markersize)   
    ax.set_xticks([0,100])
    ax.set_xlim([-2,102])
    
    if windL == 50e-3:
        ax.set_yticks([0.5, 1.8])
    elif windL == 100e-3:
        ax.set_yticks([0.5, 3.0])
    elif windL == 200e-3:
        ax.set_yticks([0.5, 5])
        
    ax.set_xlabel('arousal level [%]')
    ax.set_ylabel('$\\langle FF_{spont} \\rangle$')
    plt.savefig('%s%s.pdf' % (fig_path, figID1M), bbox_inches='tight', pad_inches=0, transparent=True)
    plt.savefig('%s%s.pdf' % (fig_path, figID1S), bbox_inches='tight', pad_inches=0, transparent=True)
    
    
    if windL == 100e-3:
    
        plt.rcParams.update({'font.size': fontSize})
        fig = plt.figure(figsize=figureSize)  
        ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 
        x = arousal_level
        y = netAvg_cellAvg_evokedFF_stimEI
        yerr = netStd_cellAvg_evokedFF_stimEI
        ax.errorbar(x, y, yerr=yerr, xerr=None, color='k', linewidth=linewidth, fmt='none')
        ax.plot(x, y, '-',  color='k', linewidth=linewidth)
        ax.plot(x, y, 'o',  color=np.array([175,54,60])/256, markersize=markersize)
        ax.set_xticks([0,100])
        ax.set_xlim([-2,102])
        ax.set_yticks([0.5, 3.])
        ax.set_xlabel('arousal level [%]')
        ax.set_ylabel('$\\langle FF_{evoked} \\rangle$')
        plt.savefig('%s%s.pdf' % (fig_path, figID2), bbox_inches='tight', pad_inches=0, transparent=True)
        
        
        plt.rcParams.update({'font.size': fontSize})
        fig = plt.figure(figsize=figureSize)  
        ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 
        x = arousal_level
        y = netAvg_cellAvg_diffFF_stimEI
        yerr = netStd_cellAvg_diffFF_stimEI
        ax.errorbar(x, y, yerr=yerr, xerr=None, color='k', linewidth=linewidth, fmt='none')
        ax.plot(x, y, '-',  color='k', linewidth=linewidth)
        ax.plot(x, y, 'o',  color=np.array([175,54,60])/256, markersize=markersize)
        ax.set_xticks([0,100])
        ax.set_xlim([-2,102])
        ax.set_yticks([0, 0.4])
        ax.set_xlabel('arousal level [%]')
        ax.set_ylabel('$\\langle \\Delta FF \\rangle$' + ' $_{(spont-evoked)}$')
        plt.savefig('%s%s.pdf' % (fig_path, figID3), bbox_inches='tight', pad_inches=0, transparent=True)
    

