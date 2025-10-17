'''
figPlot = 'cluster'
    Fig 4B
    
'''

#%%

# basic imports
import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
import sys
import os
from sklearn import metrics
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

#%% plot settings

# data paths
func_path0 = global_settings.path_to_src_code + 'run_simulations/'
func_path1 = global_settings.path_to_src_code + 'functions/'
sim_params_path = global_settings.path_to_src_code + 'run_simulations/'
sim_path = global_settings.path_to_sim_output
cluster_path = global_settings.path_to_sim_output + 'evoked_corr/hClustering/'
outpath = global_settings.path_to_manuscript_figs_final + 'hierarchical_clustering_model/'


# loading parameters
figPlot = 'cluster'

if figPlot == 'cluster':
    simParams_fname = 'simParams_051325_clu'    
    net_type = 'baseEIclu'
elif figPlot == 'hom':
    simParams_fname = 'simParams_051325_hom'
    net_type = 'baseHOM'
else:
    sys.exit('unknown figPlot')

sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'
nNetworks = 10
nDraws = 10
clusterSize_cutoff = 2
method = 'contrast'
avg_type = 'v1'
include_selfCorr = False
nPerms = 5000
sig_level_null = 0.05
fcluster_criterion = 'maxclust'
nullType = 'shuffle' # shuffle or permute

### beginning of filename
fname_begin = ( '%s%s_%s_sweep_%s_network%d_IC%d_stim%d_stimType_%s_stim_rel_amp%0.3f_simulationData.mat')

### figure ids
if figPlot == 'cluster':
    fig1ID = 'Fig4B'

savename = 'clustering_model_%sNetwork' % figPlot

#%% import functions

from fcn_plot_corrMatrix_withClusters import fcn_plot_corrMatrix_withClusters
sys.path.append(func_path0)
sys.path.append(func_path1)
from fcn_analyze_corr import fcn_sorted_corrMatrix
import fcn_hierarchical_clustering
from fcn_make_network_cluster import fcn_compute_cluster_assignments
from fcn_simulation_loading import fcn_set_sweepParam_string
from fcn_simulation_setup import fcn_define_arousalSweep


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


#%% output directory

# make output directory
if os.path.isdir(outpath) == False:
    os.makedirs(outpath)


#%% summary statistics

avgAccuracy_allNets = np.ones(nNetworks)*np.nan
frac_goodSizeClusters_sigClusters_allNets = np.ones(nNetworks)*np.nan


#%% loop over networks and draws

for indNet in range(0, nNetworks):
    
    # load in one simulation to get cluster assignments
    sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, 0) 
    params_tuple = (sim_path, simID, net_type, sweep_param_str, indNet, 0, 0, stim_shape, stim_rel_amp)
    filename = ( (fname_begin) % (params_tuple) )
    sim_data = loadmat(filename, simplify_cells=True) 
    popSize_E = sim_data['popSize_E']
    popSize_I = sim_data['popSize_I']
    clusterID_true, _ = fcn_compute_cluster_assignments(popSize_E, popSize_I)
    
    ### initialize
    nClusters_inPartition = [None]*nDraws
    qualityNull_vs_nClusters = [None]*nDraws
    quality_vs_nClusters = [None]*nDraws
    opt_nClusters = [None]*nDraws
    clustered_corr = [None]*nDraws
    unsorted_corr = [None]*nDraws
    clusterIDs = [None]*nDraws
    num_sigClusters = np.ones(nDraws)*np.nan
    sigClusters_allDraws = [None]*nDraws
    fracCells_sigClusters = np.ones(nDraws)*np.nan
    frac_goodSizeClusters_sigClusters = np.ones(nDraws)*np.nan
    size_sigClusters = np.array([])
    removedCells_allDraws = [None]*nDraws
    subsampledCells_allDraws = [None]*nDraws
    clusterAccuracy_allDraws = np.ones(nDraws)*np.nan
    psth_windSize = np.ones(nDraws)*np.nan
    corr_windSize = np.ones(nDraws)*np.nan
    sigLevel_psth = np.ones(nDraws)*np.nan
    Qtrue_opt_allDraws = np.ones(nDraws)*np.nan
    Qnull_opt_allDraws = np.ones((nDraws),dtype='object')*np.nan
    pvalue_Qtrue_Qshuff = np.ones(nDraws)*np.nan
    Qcluster_true_allDraws = np.ones((nDraws), dtype='object')*np.nan
    Qcluster_shuffle_allDraws = np.ones((nDraws), dtype='object')*np.nan
        

    for indDraw in range(0, nDraws):


        ### load clustering data
        fname = ( ('%s%s_%s_sweep_%s_network%d_stimType_%s_stim_rel_amp%0.3f.mat') % \
                  (cluster_path, simID, net_type, sweep_param_name, indNet, stim_shape, stim_rel_amp) )    


            
        # clustering info
        clusterInfo = loadmat(fname, simplify_cells=True)
        cells_subsample = clusterInfo['keep_cells_subsample'][indDraw].astype(int)
        corr = clusterInfo['corr'][indDraw].copy()
        Z = clusterInfo['linkageMatrix'][indDraw].copy()

        # parameters for later
        psth_windSize[indDraw] = clusterInfo['params']['psth_windSize']
        corr_windSize[indDraw] = clusterInfo['params']['corr_windSize']
        sigLevel_psth[indDraw] = clusterInfo['params']['sig_level']

        # true cluster IDs for this draw
        clusterID_true_sample = clusterID_true[cells_subsample].copy()
        
        # number of cells
        nCells = np.size(corr,1)

        # compute contrast vs # clusters in partition
        nClusters, Q_true = fcn_hierarchical_clustering.fcn_contrast_vs_nClusters(corr, Z, include_selfCorr)
    
        # optimal observed Q
        Q_true_opt = np.max(Q_true)
        
        # maximum determines optimal partition
        opt_nClu = nClusters[np.nanargmax(Q_true)]

        ### clustered correlation matrix
        clusterID_hCluster = fcn_hierarchical_clustering.fcn_threshold_hierarchical_clustering(Z, opt_nClu, fcluster_criterion)
        corr_sorted = fcn_sorted_corrMatrix(corr, clusterID_hCluster)

        ### size of all clusters
        sizeClusters = fcn_hierarchical_clustering.fcn_sizeClusters(clusterID_hCluster)
        goodSize_clusters = np.nonzero(sizeClusters >= clusterSize_cutoff)[0]

        
        # compute contrast vs # clusters for shuffle
        if nullType == 'shuffle':
            
            # shuffle data
            corr_shuffle = clusterInfo['corr_shuffle'][indDraw]
            Zshuffle = clusterInfo['linkageMatrix_shuffle'][indDraw,:]
            
            # shuffle clusters
            nClusters_shuffle, Q_shuffle, clusterID_hCluster_shuffle = fcn_hierarchical_clustering.fcn_optimalPartition_contrast_shuffle(corr_shuffle, Zshuffle, include_selfCorr)
            qualityNull_vs_nClusters[indDraw] = Q_shuffle

            ### optimal shuffle Q
            Q_shuffle_opt = np.max(Q_shuffle,0)
            Qnull_opt_allDraws[indDraw] = Q_shuffle_opt

            ### pvalue comparing optimal shuffle Q to optimal observed Q
            pvalue_Qtrue_Qshuff[indDraw] = fcn_hierarchical_clustering.fcn_oneSided_pval(Q_true_opt, Q_shuffle_opt, 'greater')

            ### for each cluster, compare to shuffled distribution
            sigClusters, trueQClu, nullQClu, sig_cutoff = fcn_hierarchical_clustering.fcn_compute_sigClusters_againstShuffle(corr, clusterID_hCluster, corr_shuffle, clusterID_hCluster_shuffle, method, include_selfCorr, clusterSize_cutoff, sig_level=sig_level_null)
            Qcluster_shuffle_allDraws[indDraw] = nullQClu
            Qcluster_true_allDraws[indDraw] = trueQClu[goodSize_clusters]
            frac_goodSizeClusters_sigClusters[indDraw] = np.size(sigClusters)/np.size(goodSize_clusters)
            
        elif nullType == 'permute':

            ### significant clusters
            sigClusters, trueQ, nullQ_permute = fcn_hierarchical_clustering.fcn_compute_sigClusters_againstPermutation(corr, clusterID_hCluster, nPerms, method, include_selfCorr, clusterSize_cutoff, sig_level=sig_level_null)
    
        else:
            sys.exit('unknown null type')
    
    
        ### fraction of cells in significant clusters
        fCells_sigClusters = np.sum(sizeClusters[sigClusters])/nCells
        
        ### clustering accuracy
        if net_type == 'baseEIclu':
            cluster_accuracy = metrics.adjusted_rand_score(clusterID_true_sample, clusterID_hCluster)
        else:
            cluster_accuracy = np.nan

    
        ### save info
        quality_vs_nClusters[indDraw] = Q_true
        nClusters_inPartition[indDraw] = nClusters
        opt_nClusters[indDraw] = opt_nClu
        clusterIDs[indDraw] = clusterID_hCluster
        clustered_corr[indDraw] = corr_sorted
        unsorted_corr[indDraw] = corr
        sigClusters_allDraws[indDraw] = sigClusters
        num_sigClusters[indDraw] = len(sigClusters)
        fracCells_sigClusters[indDraw] = fCells_sigClusters
        size_sigClusters = np.append(size_sigClusters, sizeClusters[sigClusters])   
        clusterAccuracy_allDraws[indDraw] = cluster_accuracy
        subsampledCells_allDraws[indDraw] = cells_subsample
        Qtrue_opt_allDraws[indDraw] = Q_true_opt

        print(indDraw)

    # summary stats
    avgAccuracy_allNets[indNet] = np.mean(clusterAccuracy_allDraws)
    frac_goodSizeClusters_sigClusters_allNets[indNet] = np.mean(frac_goodSizeClusters_sigClusters)

    print(clusterAccuracy_allDraws)
    print(num_sigClusters)
    print(pvalue_Qtrue_Qshuff)
    print(frac_goodSizeClusters_sigClusters)
    
    ### save results
    results = dict()
    params = dict()
    
    params['nNetworks'] = nNetworks
    params['nDraws'] = nDraws
    params['net_type'] = net_type
    params['sweep_param_name'] = sweep_param_name
    params['simParams_fname'] = simParams_fname
    params['sim_params_path'] = sim_params_path
    params['sim_path'] = sim_path
    params['clusterSize_cutoff'] = clusterSize_cutoff
    params['cluster_path'] = cluster_path
    params['sig_level_null'] = sig_level_null
    params['nullType'] = nullType
    params['fcluster_criterion'] = fcluster_criterion
    params['method_name'] = method
    params['avg_type'] = avg_type
    params['include_selfCorr'] = include_selfCorr
    params['nPerms'] = nPerms
    params['corr_windSize'] = corr_windSize
    params['psth_windSize'] = psth_windSize
    params['sigLevel_psth'] = sigLevel_psth
    
    results['nClusters_inPartition'] = nClusters_inPartition
    results['quality_vs_nClusters'] = quality_vs_nClusters
    results['qualityNull_vs_nClusters'] =  qualityNull_vs_nClusters
    results['Qcluster_shuffle_allDraws'] = Qcluster_shuffle_allDraws
    results['Qcluster_true_allDraws'] = Qcluster_true_allDraws
    results['opt_nClusters'] = opt_nClusters
    results['subsampledCells_allDraws'] = subsampledCells_allDraws
    results['clusterIDs'] = clusterIDs
    results['sigClusters_allDraws'] = sigClusters_allDraws
    results['clusterAccuracy_allDraws'] = clusterAccuracy_allDraws
    results['frac_goodSizeClusters_sigClusters'] = frac_goodSizeClusters_sigClusters
    results['avgAccuracy_allNets'] = avgAccuracy_allNets
    results['frac_goodSizeClusters_sigClusters_allNets'] = frac_goodSizeClusters_sigClusters_allNets

    results['params'] = params
    
    savemat(('%s%s_net%d_results.mat' % (outpath, savename, indNet)), results)
    
    
    #%% sorted correlation matrix for example draw
    
    if figPlot == 'cluster':

        if indNet == 0:
    
            drawPlot = 0        
            cbar_lim = 0.5
            cmap = 'RdBu_r'
            cbar_label = ''
            
            plt.rcParams.update({'font.size': 6})
            fig = plt.figure(figsize=(0.95,0.95))  
            ax = fig.add_axes([0.1, 0.1, 0.9, 0.9]) 
            fcn_plot_corrMatrix_withClusters(ax, unsorted_corr[drawPlot], clusterIDs[drawPlot], sigClusters_allDraws[drawPlot], 'k', cbar_label, cmap, cbar_lim, noTicks=False)
            if net_type == 'baseEIclu':
                ax.set_title('noise corr. (sorted)\nclustered network', fontsize=6)    
            else:
                ax.set_title('noise corr. (sorted)\n uniform network', fontsize=6)                
            plt.savefig((outpath + fig1ID + '.pdf'), bbox_inches='tight', pad_inches=0, transparent=True)
            plt.close()
    
            ### save results
            results = dict()
            params = dict()
            
            params['nNetworks'] = nNetworks
            params['nDraws'] = nDraws
            params['net_type'] = net_type
            params['sweep_param_name'] = sweep_param_name
            params['simParams_fname'] = simParams_fname
            params['sim_params_path'] = sim_params_path
            params['sim_path'] = sim_path
            params['clusterSize_cutoff'] = clusterSize_cutoff
            params['cluster_path'] = cluster_path
            params['sig_level_null'] = sig_level_null
            params['nullType'] = nullType
            params['fcluster_criterion'] = fcluster_criterion
            params['method_name'] = method
            params['avg_type'] = avg_type
            params['include_selfCorr'] = include_selfCorr
            params['nPerms'] = nPerms
            params['corr_windSize'] = corr_windSize
            params['psth_windSize'] = psth_windSize
            params['sigLevel_psth'] = sigLevel_psth
            
            results['nClusters_inPartition'] = nClusters_inPartition
            results['quality_vs_nClusters'] = quality_vs_nClusters
            results['qualityNull_vs_nClusters'] =  qualityNull_vs_nClusters
            results['Qcluster_shuffle_allDraws'] = Qcluster_shuffle_allDraws
            results['Qcluster_true_allDraws'] = Qcluster_true_allDraws
            results['opt_nClusters'] = opt_nClusters
            results['subsampledCells_allDraws'] = subsampledCells_allDraws
            results['clusterIDs'] = clusterIDs
            results['sigClusters_allDraws'] = sigClusters_allDraws
            results['clusterAccuracy_allDraws'] = clusterAccuracy_allDraws
            results['frac_goodSizeClusters_sigClusters'] = frac_goodSizeClusters_sigClusters
            results['avgAccuracy_allNets'] = avgAccuracy_allNets
            results['frac_goodSizeClusters_sigClusters_allNets'] = frac_goodSizeClusters_sigClusters_allNets
        
            results['params'] = params
            
            savemat(('%s%s_%d_results.mat' % (outpath, savename, fig1ID)), results)

#%% SUMMARY STATISTICS

print('net avg accuracy: %0.5f' % np.mean(avgAccuracy_allNets))
print('net avg frac good clusters that are significant: %0.5f' % np.mean(frac_goodSizeClusters_sigClusters_allNets))

    


