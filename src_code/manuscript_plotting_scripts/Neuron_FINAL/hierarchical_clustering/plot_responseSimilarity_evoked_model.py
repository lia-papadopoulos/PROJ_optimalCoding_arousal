
'''
before running this script, run plot_clusters_evoked_model.py for  
    figPlot = 'cluster'

this script will then generate
    Fig 4C, D
'''


#%%
# basic imports
import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
import sys
import os
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


#%% data paths
func_path0 = global_settings.path_to_src_code + 'run_simulations/'
func_path1 = global_settings.path_to_src_code + 'functions/'
func_path2 = global_settings.path_to_src_code + 'data_analysis/'
sim_params_path = global_settings.path_to_src_code + 'run_simulations/'
sim_path = global_settings.path_to_sim_output
psth_path = global_settings.path_to_sim_output + 'psth/'
cluster_path = global_settings.path_to_manuscript_figs_final + 'hierarchical_clustering_model/'
outpath = global_settings.path_to_manuscript_figs_final + 'hierarchical_clustering_model/'

#%% analysis functions
from fcn_plot_corrMatrix_withClusters import fcn_plot_corrMatrix_withClusters
sys.path.append(func_path2)
from fcn_SuData_analysis import fcn_cosineSim_respVectors
sys.path.append(func_path0)
sys.path.append(func_path1)
from fcn_analyze_corr import fcn_compare_sigCorr_within_vs_between_true_shuffled_clustering_alt
from fcn_analyze_corr import fcn_sorted_corrMatrix
import fcn_hierarchical_clustering
import fcn_compute_firing_stats
from fcn_simulation_loading import fcn_set_sweepParam_string
from fcn_simulation_setup import fcn_define_arousalSweep


#%% loading parameters


simParams_fname = 'simParams_051325_clu'    
net_type = 'baseEIclu'
sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'
nNetworks = 10
nDraws = 10
windSize = 100e-3
sigThresh = 0.05
nPerms = 1000
withinClu_avgType = 'v1'
include_selfCorr = False
responseSim_type = 'pearson'
model_type = 'cluster'

### fig IDs
fig1ID = 'Fig4C'
fig2ID = 'Fig4D'
savename = 'responseSim_model_clusterNetwork'

#%% filenames

fname_sim = ( '%s%s_%s_sweep_%s_network%d_IC%d_stim%d_stimType_%s_stim_rel_amp%0.3f_simulationData.mat' )
fname_psth = ( '%s%s_%s_sweep_%s_network%d_stim%d_stimType_%s_stim_rel_amp%0.3f_psth_windSize%0.3fs.mat')
fname_clustering = ('%sclustering_model_%sNetwork_net%d_results.mat')

# make output directory
if os.path.isdir(outpath) == False:
    os.makedirs(outpath)


#%% load sim parameters
sys.path.append(sim_params_path)
params = importlib.import_module(simParams_fname) 
s_params = params.sim_params

#%% arousal sweep
s_params = fcn_define_arousalSweep(s_params)

#%% unpack simulation parameters
simID = s_params['simID']
n_sweepParams = s_params['nParams_sweep']
swept_params_dict = s_params['swept_params_dict']
simID = s_params['simID']
nStim = s_params['nStim']
stim_shape = s_params['stim_shape']
stim_rel_amp = s_params['stim_rel_amp']

del params
del s_params


#%% loop over networks and draws

for indNet in range(0, nNetworks):
                    
    ### load in one simulation to get cluster assignments
    sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, 0) 
    params_tuple = (sim_path, simID, net_type, sweep_param_str, indNet, 0, 0, stim_shape, stim_rel_amp)
    filename = ( (fname_sim) % (params_tuple) )
    sim_data = loadmat(filename, simplify_cells=True) 

    ### initialize
    responseMatrix_unsorted = [None]*nDraws
    responseSimMatrix_sorted = [None]*nDraws
    responseSim_cluster_true = [None]*nDraws
    responseSim_cluster_perm = [None]*nDraws
    pVal_responseSim_cluster_perm = [None]*nDraws
    
    ### load clustering info
    clustering_tuple = (cluster_path, model_type, indNet)
    clustering_filename = ( (fname_clustering) % (clustering_tuple) )
    clusterInfo = loadmat(clustering_filename, simplify_cells=True) 
    
    ### unpack clustering info
    subsampledCells_allDraws = clusterInfo['subsampledCells_allDraws']
    clusterIDs_allDraws = clusterInfo['clusterIDs']
    sigClusters_allDraws = clusterInfo['sigClusters_allDraws']
    psth_windSize_allDraws = clusterInfo['params']['psth_windSize']
    corr_windSize_allDraws = clusterInfo['params']['corr_windSize']
    sigLevel_psth_allDraws = clusterInfo['params']['sigLevel_psth']
    
    ### loop over draws
    for indDraw in range(0, nDraws):

        ### get data for this draw
        if nDraws == 1:
            psth_windSize = psth_windSize_allDraws
            sigLevel_psth = sigLevel_psth_allDraws 
            subsampledCells = subsampledCells_allDraws
            clusterID = clusterIDs_allDraws
            goodClusters = sigClusters_allDraws          
        else:
            psth_windSize = psth_windSize_allDraws[indDraw]
            sigLevel_psth = sigLevel_psth_allDraws[indDraw]   
            subsampledCells = subsampledCells_allDraws[indDraw]
            clusterID = clusterIDs_allDraws[indDraw]
            goodClusters = sigClusters_allDraws[indDraw]
        
        ### initialize response vector for each stimulus
        resp_eachStim = np.zeros((nStim), dtype='object')
        
        ### loop over stimuli and get responses
        for indStim in range(0, nStim):
            
            # filename
            params_tuple = (psth_path, simID, net_type, sweep_param_name, indNet, indStim, stim_shape, stim_rel_amp, psth_windSize)        
            fname_psth_full = ( (fname_psth) % ( params_tuple ))
            psth_data = loadmat(fname_psth_full, simplify_cells=True)
            
            # significant cells and responses
            _, resp_eachStim[indStim] = fcn_compute_firing_stats.fcn_compute_sigCells_respAmp(psth_data, sigLevel_psth)

        ### response of every cell to each stim
        resp_allStim = fcn_compute_firing_stats.fcn_responseAmp_allStim(resp_eachStim)

        ### only keep cells that we want
        resp_allStim = resp_allStim[subsampledCells, :].copy()

        ### response similarity 
        if responseSim_type == 'pearson':
            responseSim = np.corrcoef(resp_allStim)
        else:
            responseSim = fcn_cosineSim_respVectors(resp_allStim)
        
        ### sort response similarity by spontaneous clusters
        responseSim_sorted = fcn_sorted_corrMatrix(responseSim, clusterID)
        np.fill_diagonal(responseSim_sorted,0)


        ### compute statistical significance of cluster response similarity
        if np.size(goodClusters) >= 2:
            
            within_minus_between_true, within_minus_between_shuf = \
                fcn_compare_sigCorr_within_vs_between_true_shuffled_clustering_alt(clusterID, responseSim, goodClusters, \
                                                                                   withinClu_avgType, include_selfCorr, nPerms)
            
            pval = fcn_hierarchical_clustering.fcn_oneSided_pval(within_minus_between_true, within_minus_between_shuf, 'greater')  
    
        else:
    
            within_minus_between_true = 0.
            within_minus_between_shuf = np.ones(nPerms)*np.nan
            pval = np.nan
        
        ### store data
        responseMatrix_unsorted[indDraw] = responseSim
        responseSimMatrix_sorted[indDraw] = responseSim_sorted
        responseSim_cluster_true[indDraw] = within_minus_between_true
        responseSim_cluster_perm[indDraw] = within_minus_between_shuf
        pVal_responseSim_cluster_perm[indDraw] = pval
    
        print(indDraw)
    
        #%% save results
    
        results = dict()
        params = dict()
        
        params['func_path0'] = func_path0
        params['func_path1'] = func_path1
        params['func_path2'] = func_path2
        params['simParams_fname'] = simParams_fname
        params['sim_params_path'] = sim_params_path
        params['psth_path'] = sweep_param_name
        params['sim_path'] = sim_path
        params['net_type'] = net_type
        params['sweep_param_name'] = sweep_param_name
        params['cluster_path'] = cluster_path
        params['cluster_path'] = cluster_path
        params['nDraws'] = nDraws
        params['sigThresh'] = sigThresh
        params['withinClu_avgType'] = withinClu_avgType
        params['include_selfCorr'] = include_selfCorr
        params['nPerms'] = nPerms
        params['windSize'] = windSize
        params['responseSim_type'] = responseSim_type
        results['params'] = params
        savemat(('%s%s_results.mat' % (outpath, savename)), results)
    

    ### PLOTTING
    if indNet == 0:
    
        ### sorted response similarity for example draw
        drawPlot = 0        
        cmap = 'RdBu_r'
        cbar_label = ''
        cbar_lim = 0.9
    
        plt.rcParams.update({'font.size': 6})
        fig = plt.figure(figsize=(0.95,0.95))  
        ax = fig.add_axes([0.1, 0.1, 0.9, 0.9]) 
        
        if nDraws == 1:
            
            fcn_plot_corrMatrix_withClusters(ax, responseMatrix_unsorted[drawPlot], clusterIDs_allDraws, sigClusters_allDraws, \
                                         'k', cbar_label, cmap, cbar_lim, linewidth=1., noTicks=False)        
        else:
        
            fcn_plot_corrMatrix_withClusters(ax, responseMatrix_unsorted[drawPlot], clusterIDs_allDraws[drawPlot], sigClusters_allDraws[drawPlot], \
                                         'k', cbar_label, cmap, cbar_lim, linewidth=1., noTicks=False)
        if net_type == 'baseEIclu':
            ax.set_title('tuning sim.\nclustered network', fontsize=6)   
        else:
            ax.set_title('tuning sim.\nuniform network', fontsize=6)           
        plt.savefig((outpath + fig1ID + '.pdf'), bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()
        


        ### shuffled and true response similarity for an example draw
        drawPlot = 0
        colorPlot = colorPlot = 'dimgrey'
    
        plt.rcParams.update({'font.size': 6})
        fig = plt.figure(figsize=(0.85,0.95))  
        ax = fig.add_axes([0.2, 0.2, 0.75, 0.72]) 
        ax.tick_params(axis='both', width=0.5)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        yTrue = responseSim_cluster_true[drawPlot]
        yPerm = responseSim_cluster_perm[drawPlot]
        ax.hist(yPerm, color=colorPlot, label='permuted')
        ax.plot([yTrue, yTrue], [0, 220], color='lightseagreen', label='observed')
        ax.set_xlabel('cluster-based tuning sim.')
        ax.set_ylabel('no. permutations')
        if net_type == 'baseEIclu':
            ax.set_title('test statistic\nclustered network', fontsize=6)
        else:
            ax.set_title('test statistic\nuniform network', fontsize=6)
            
        plt.savefig((outpath + fig2ID + '.pdf'), bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()    
        
        
        results = dict()
        params = dict()
        
        params['func_path0'] = func_path0
        params['func_path1'] = func_path1
        params['func_path2'] = func_path2
        params['simParams_fname'] = simParams_fname
        params['sim_params_path'] = sim_params_path
        params['psth_path'] = sweep_param_name
        params['sim_path'] = sim_path
        params['net_type'] = net_type
        params['sweep_param_name'] = sweep_param_name
        params['cluster_path'] = cluster_path
        params['cluster_path'] = cluster_path
        params['nDraws'] = nDraws
        params['sigThresh'] = sigThresh
        params['withinClu_avgType'] = withinClu_avgType
        params['include_selfCorr'] = include_selfCorr
        params['nPerms'] = nPerms
        params['windSize'] = windSize
        params['responseSim_type'] = responseSim_type
        results['params'] = params
        savemat(('%s_figurePanel_results.mat' % (outpath)), results)
