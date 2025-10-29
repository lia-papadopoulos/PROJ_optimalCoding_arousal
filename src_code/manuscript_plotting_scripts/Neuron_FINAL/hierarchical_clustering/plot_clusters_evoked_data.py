

'''
cellSelection = ''
    Fig4 E,F,I,J

cellSelection = '_spkTemplate_soundResp_cellSelection1'
    FigS8E
'''


#%%

# basic imports
import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
import sys
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

#%% load functions

# analysis functions
from fcn_plot_corrMatrix_withClusters import fcn_plot_corrMatrix_withClusters

sys.path.append(global_settings.path_to_src_code + 'data_analysis/')
sys.path.append(global_settings.path_to_src_code + 'functions/')
from fcn_analyze_corr import fcn_sorted_corrMatrix
import fcn_hierarchical_clustering

#%% plott settings

# dataset to run
cellSelection = ''

# sessions to run
sessions_to_run = np.array(['LA3_session3', \
                       'LA8_session1', \
                       'LA8_session2', \
                       'LA9_session1', \
                       'LA9_session3', 'LA9_session4', 'LA9_session5', \
                       'LA11_session1', 'LA11_session2', 'LA11_session3', 'LA11_session4', \
                       'LA12_session1', 'LA12_session2', 'LA12_session3', 'LA12_session4'
                      ])
    
sessions_plot = ['LA9_session1', 'LA11_session2']


# parameters
cells_toKeep = 'allSigCells' 
wind_length = 100e-3
rate_thresh = -np.inf
restOnly = False
clusterSize_cutoff = 2
nClusters_cutoff = 2
method = 'contrast'
avg_type = 'v1'
include_selfCorr = False
nPerms = 5000
plot_results = False
sig_level_null = 0.05
fcluster_criterion = 'maxclust'
nullType = 'shuffle' 


# data paths
cluster_path = global_settings.path_to_data_analysis_output + 'spont_evoked_correlations_pupil/evoked_hClustering_pupilPercentile_combinedBlocks/'
outpath = global_settings.path_to_manuscript_figs_final + 'hierarchical_clustering_data/'


# update figure path
if cellSelection == '':
    outpath = outpath + 'original_cellSelection/'
elif cellSelection == '_spkTemplate_soundResp_cellSelection1':
    outpath = outpath + 'spkTemplate_soundResp_cellSelection1/'
    

# figure ids

if cellSelection == '':
    fig1ID = 'Fig4E'
    fig2ID = 'Fig4F'
    fig3ID = 'Fig4I'
    fig4ID = 'Fig4J'
    savename = 'clustering_data_default'
    
if cellSelection == '_spkTemplate_soundResp_cellSelection1':
    fig3ID = 'FigS8E'
    savename = 'clustering_data_alt_cellSelection'


#%% initialize

### data name
data_name = '' + cellSelection

### make output directory
if os.path.isdir(outpath) == False:
    os.makedirs(outpath)

nSessions = len(sessions_to_run)
nClusters_inPartition = [None]*nSessions
quality_vs_nClusters = [None]*nSessions
qualityNull_vs_nClusters = [None]*nSessions
opt_nClusters = [None]*nSessions
clustered_corr = [None]*nSessions
unsorted_corr = [None]*nSessions
clusterIDs = [None]*nSessions
num_goodClusters = np.ones(nSessions)*np.nan
goodClusters_allSessions = [None]*nSessions
fracCells_goodClusters = np.ones(nSessions)*np.nan
size_goodClusters = np.array([])
removedCells_allSessions = [None]*nSessions
Qtrue_opt_allSessions = np.ones(nSessions)*np.nan
Qnull_opt_allSessions = np.ones((nSessions),dtype='object')*np.nan
pvalue_Qtrue_Qshuff_allSessions = np.ones(nSessions)*np.nan
Qcluster_true_allSessions = np.ones((nSessions), dtype='object')*np.nan
Qcluster_shuffle_allSessions = np.ones((nSessions), dtype='object')*np.nan
sig_cutoff_allSessions = np.ones((nSessions))*np.nan


#%% loop over sessions

for indSession, session in enumerate(sessions_to_run):

    print(session)

    
    ### load data
    clusterInfo = loadmat(('%s%s_responsiveOnly_windLength%0.3fs_rateThresh%0.3fHz_hClustering_%s%s.mat' % (cluster_path, session, wind_length, rate_thresh, cells_toKeep, data_name)), simplify_cells=True)
    
        
    ### correlation info

    # cells that were removed from analysis
    removed_cells = clusterInfo['remove_cells']
    
    # significance level used to determine significant cells
    sigLevel_responsiveCells = clusterInfo['params']['sig_level']
    
    # true correlation and clustering results
    corr = clusterInfo['corr_allPupil']
    Z = clusterInfo['linkageMatrix_allPupil']

    # cells
    nCells = np.size(corr,1)

    # compute contrast vs # clusters in partition
    nClusters, Q_true = fcn_hierarchical_clustering.fcn_contrast_vs_nClusters(corr, Z, include_selfCorr)
    Q_true_opt = np.max(Q_true)
    Qtrue_opt_allSessions[indSession] = np.max(Q_true)
    
    # maximum determines optimal partition
    opt_nClu = nClusters[np.nanargmax(Q_true)]

    ### clustered correlation matrix
    clusterID_hCluster = fcn_hierarchical_clustering.fcn_threshold_hierarchical_clustering(Z, opt_nClu, fcluster_criterion)
    corr_sorted = fcn_sorted_corrMatrix(corr, clusterID_hCluster)

    ### size of all clusters
    sizeClusters = fcn_hierarchical_clustering.fcn_sizeClusters(clusterID_hCluster)
    goodSize_clusters = np.nonzero(sizeClusters >= clusterSize_cutoff)[0]


    # shuffle clustering    
    if nullType == 'shuffle':
        
        # shuffle correlation and clustering results
        corr_shuffle = clusterInfo['corr_allPupil_shuffle']
        Z_shuffle = clusterInfo['linkageMatrix_allPupil_shuffle'] 
        
        # shuffle clusters
        nClusters_shuffle, Q_shuffle, clusterID_hCluster_shuffle = fcn_hierarchical_clustering.fcn_optimalPartition_contrast_shuffle(corr_shuffle, Z_shuffle, include_selfCorr)
        Q_shuffle_opt = np.max(Q_shuffle, 0)
        
        qualityNull_vs_nClusters[indSession] = Q_shuffle
        Qnull_opt_allSessions[indSession] = Q_shuffle_opt
        pvalue_Qtrue_Qshuff_allSessions[indSession] = fcn_hierarchical_clustering.fcn_oneSided_pval(Q_true_opt, Q_shuffle_opt, 'greater')
        
        # for each cluster, compare to shuffled distribution
        goodClusters, trueQ, nullQ_permute, sig_cutoff = fcn_hierarchical_clustering.fcn_compute_sigClusters_againstShuffle(corr, clusterID_hCluster, corr_shuffle, clusterID_hCluster_shuffle, method, include_selfCorr, clusterSize_cutoff, sig_level=sig_level_null)
        Qcluster_true_allSessions[indSession] = trueQ[goodSize_clusters]
        Qcluster_shuffle_allSessions[indSession] = nullQ_permute
        sig_cutoff_allSessions[indSession] = sig_cutoff
    
    elif nullType == 'permute':

        ### significant clusters
        goodClusters, trueQ, nullQ_permute = fcn_hierarchical_clustering.fcn_compute_sigClusters_againstPermutation(corr, clusterID_hCluster, nPerms, method, include_selfCorr, clusterSize_cutoff, sig_level=sig_level_null)

    else:
        
        sys.exit('unknown null type')


    ### fraction of cells in significant clusters
    if np.size(goodClusters) >= nClusters_cutoff:
        fCells_goodClusters = np.sum(sizeClusters[goodClusters])/nCells
        goodClusters_allSessions[indSession] = goodClusters
        num_goodClusters[indSession] = len(goodClusters)
        fracCells_goodClusters[indSession] = fCells_goodClusters
        size_goodClusters = np.append(size_goodClusters, sizeClusters[goodClusters])   
    else:
        fCells_goodClusters = 0.
        goodClusters_allSessions[indSession] = np.array([])
        num_goodClusters[indSession] = 0
        fracCells_goodClusters[indSession] = fCells_goodClusters
        print('no good clusters in session %s' % session)
    

    ### save info
    removedCells_allSessions[indSession] = removed_cells
    quality_vs_nClusters[indSession] = Q_true
    nClusters_inPartition[indSession] = nClusters
    opt_nClusters[indSession] = opt_nClu
    clusterIDs[indSession] = clusterID_hCluster
    clustered_corr[indSession] = corr_sorted
    unsorted_corr[indSession] = corr
 
      

### print pvalue
print(pvalue_Qtrue_Qshuff_allSessions)

### save results
results = dict()
params = dict()

params['sessions'] = sessions_to_run
params['cells_toKeep'] = cells_toKeep
params['wind_length'] = wind_length
params['rate_thresh'] = rate_thresh
params['restOnly'] = restOnly
params['clusterSize_cutoff'] = clusterSize_cutoff
params['cluster_path'] = cluster_path
params['sig_level_null'] = sig_level_null
params['nullType'] = nullType
params['fcluster_criterion'] = fcluster_criterion
params['method_name'] = method
params['avg_type'] = avg_type
params['include_selfDist'] = include_selfCorr
params['nPerms'] = nPerms
params['removedCells_allSessions'] = removedCells_allSessions
params['sigLevel_responsiveCells'] = sigLevel_responsiveCells

results['nClusters_inPartition'] = nClusters_inPartition
results['quality_vs_nClusters'] = quality_vs_nClusters
results['clustered_corr'] = clustered_corr
results['unsorted_corr'] = unsorted_corr
results['clusterIDs'] = clusterIDs
results['num_goodClusters'] = num_goodClusters
results['size_goodClusters'] = size_goodClusters
results['opt_nClusters'] = opt_nClusters
results['goodClusters_allSessions'] = goodClusters_allSessions
results['fracCells_goodClusters'] = fracCells_goodClusters
results['pvalue_Qtrue_Qshuff_allSessions'] = pvalue_Qtrue_Qshuff_allSessions
results['params'] = params

savemat(('%s%s_results.mat' % (outpath, savename)), results)


#%% PLOTTING

figname = data_name + '_'


#%% compare true and shuffled Q for each cluster at optimal partition for example sessions

if cellSelection == '':

    count = 0
    
    for _, session in enumerate(sessions_plot):
        
        count+=1
        sessionInd = np.nonzero(sessions_to_run == session)[0][0]
                
        plt.rcParams.update({'font.size': 6})
        fig = plt.figure(figsize=(0.85,0.95))  
        ax = fig.add_axes([0.2, 0.2, 0.75, 0.72]) 
        ax.tick_params(axis='both', width=0.5)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        sig_thresh = sig_cutoff_allSessions[sessionInd]
        data1 = Qcluster_shuffle_allSessions[sessionInd]
        data2 = Qcluster_true_allSessions[sessionInd]
        
        vPlot = ax.violinplot(data1, positions=np.array([1]), vert=True, widths=0.6, showextrema=True, showmedians=True, quantiles=None, points=100)
        
        
        for pc in vPlot['bodies']:
            pc.set_facecolor('dimgrey')
            pc.set_edgecolor('dimgrey')
            pc.set_alpha(1)
            pc.set_label('cluster obs.')
        vPlot['cmaxes'].set_color('black')
        vPlot['cmaxes'].set_linewidth(0.5)
        vPlot['cmins'].set_color('black')
        vPlot['cmins'].set_linewidth(0.5)
        vPlot['cmedians'].set_color('black')
        vPlot['cmedians'].set_linewidth(0.5)
        vPlot['cbars'].set_color('black')
        vPlot['cbars'].set_linewidth(0.5)
        
        vPlot = ax.violinplot(data2, positions=np.array([2]), vert=True, widths=0.6, showextrema=True, showmedians=True, quantiles=None, points=100)
        
        
        for pc in vPlot['bodies']:
            pc.set_facecolor('lightseagreen')
            pc.set_edgecolor('lightseagreen')
            pc.set_alpha(1)
            pc.set_label('cluster obs.')
    
        vPlot['cmaxes'].set_color('black')
        vPlot['cmaxes'].set_linewidth(0.5)
        vPlot['cmins'].set_color('black')
        vPlot['cmins'].set_linewidth(0.5)
        vPlot['cmedians'].set_color('black')
        vPlot['cmedians'].set_linewidth(0.5)
        vPlot['cbars'].set_color('black')
        vPlot['cbars'].set_linewidth(0.5)
        
        ax.plot([0.5,2.5], [sig_thresh, sig_thresh], '--', linewidth=1, markersize=0.3, color='darkorchid')
    
        ax.set_ylabel('cluster quality')
        ax.set_xticks([1,2])
        ax.set_xticklabels(['shuf', 'obs'])
    
        plt.savefig((outpath + fig1ID + '_example%d.pdf' % count), bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()


#%% sorted correlation matrix for example sessions

if cellSelection == '':

    cbar_lim = 0.5
    cmap = 'RdBu_r'
    cbar_label = ''
    
    count = 0
    
    for _, session in enumerate(sessions_plot):
        
        count+=1
        sessionInd = np.nonzero(sessions_to_run == session)[0][0]
        
        plt.rcParams.update({'font.size': 6})
        fig = plt.figure(figsize=(0.95,0.95))  
        ax = fig.add_axes([0.1, 0.1, 0.9, 0.9]) 
        fcn_plot_corrMatrix_withClusters(ax, unsorted_corr[sessionInd], clusterIDs[sessionInd], goodClusters_allSessions[sessionInd], 'k', cbar_label, cmap, cbar_lim, linewidth=0.5, noTicks=False)
        ax.set_title('noise corr. (sorted)', fontsize=6)    
        plt.savefig((outpath + fig2ID + '_example%d.pdf' % count), bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()
        


#%% histogram of fraction of clustered cells in a session

colorPlot = colorPlot = 'dimgrey'
colorPlot2='cornflowerblue'
bin_width = 0.1

hist_data = fracCells_goodClusters.flatten()
hist_data = hist_data[~np.isnan(hist_data)]
data_extreme = np.nanmax(np.abs(hist_data))
bins = np.arange( -bin_width/2, np.round(data_extreme + 2*bin_width, 1), bin_width )
counts, bin_edges = np.histogram(hist_data, bins)
bin_centers = bin_edges[:-1] + bin_width/2

plt.rcParams.update({'font.size': 7})
fig = plt.figure(figsize=(1.,1.))  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 
ax.tick_params(axis='both', width=0.5)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.bar(bin_centers, counts, width=np.diff(bin_edges), color=colorPlot)
ax.set_xlim([bin_edges[0],bin_edges[-1]])
ax.set_ylabel('no. sessions')
ax.set_xlabel('fraction cells\n in sig. clusters')
plt.savefig((outpath + fig3ID + '.pdf'), bbox_inches='tight', pad_inches=0, transparent=True)
plt.close()



#%% histogram of size of good clusters in a session

if cellSelection == '':

    colorPlot = colorPlot = 'dimgrey'
    colorPlot2='cornflowerblue'
    bin_width = 1.
    
    hist_data = size_goodClusters.copy()
    hist_data = hist_data[~np.isnan(hist_data)]
    data_extreme = np.nanmax(np.abs(hist_data))
    bins = np.arange( clusterSize_cutoff-bin_width/2, np.round(data_extreme + 2*bin_width, 1), bin_width )
    counts, bin_edges = np.histogram(hist_data, bins)
    bin_centers = bin_edges[:-1] + bin_width/2
    
    plt.rcParams.update({'font.size': 7})
    fig = plt.figure(figsize=(1.,1.))  
    ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 
    ax.tick_params(axis='both', width=0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.bar(bin_centers, counts, width=np.diff(bin_edges), color=colorPlot)
    ax.set_xlim([bin_edges[0],bin_edges[-1]])
    ax.set_xticks(np.arange(bin_centers[0],bin_centers[-1],10))
    ax.set_ylabel('no. sig. clusters')
    ax.set_xlabel('cluster size')
    plt.savefig((outpath + fig4ID + '.pdf'), bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

