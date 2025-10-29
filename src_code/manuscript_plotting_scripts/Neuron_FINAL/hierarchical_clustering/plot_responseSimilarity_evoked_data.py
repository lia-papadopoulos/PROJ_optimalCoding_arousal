
'''
before running this script, run plot_clusters_evoked_data.py

using cellSelection = '', this script will generate
    Fig4 G, H, K

using cellSelection = '_spkTemplate_soundResp_cellSelection1', this script will generate
    FigS8F
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


#%% analysis functions
from fcn_plot_corrMatrix_withClusters import fcn_plot_corrMatrix_withClusters
sys.path.append(global_settings.path_to_src_code + 'data_analysis/')
from fcn_SuData_analysis import fcn_cosineSim_respVectors
from fcn_SuData_analysis import fcn_significant_preStim_vs_postStim
sys.path.append(global_settings.path_to_src_code + 'functions/')
from fcn_analyze_corr import fcn_compare_sigCorr_within_vs_between_true_shuffled_clustering_alt
from fcn_analyze_corr import fcn_sorted_corrMatrix
import fcn_hierarchical_clustering

#%% plot settings

# path to data
psth_path = global_settings.path_to_data_analysis_output + 'psth_allTrials/'
cluster_path = global_settings.path_to_manuscript_figs_final + 'hierarchical_clustering_data/'
outpath = global_settings.path_to_manuscript_figs_final + 'hierarchical_clustering_data/'


# data set to run
cellSelection = ''


# sessions to run
sessions_to_run = np.array(['LA3_session3', \
                       'LA8_session1', 'LA8_session2', \
                       'LA9_session1', \
                       'LA9_session3', 'LA9_session4', 'LA9_session5', \
                       'LA11_session1', 'LA11_session2', 'LA11_session3', 'LA11_session4', \
                       'LA12_session1', 'LA12_session2', 'LA12_session3', 'LA12_session4'
                      ])

    
sessions_plot = ['LA9_session1', 'LA11_session2']
    

# analysis parameters
sigThresh = 0.05
nPerms = 1000
withinClu_avgType = 'v1'
include_selfCorr = False
responseSim_type = 'pearson'

# update figure path
if cellSelection == '':
    outpath = outpath + 'original_cellSelection/'  
    cluster_path = cluster_path + 'original_cellSelection/'
elif cellSelection == '_spkTemplate_soundResp_cellSelection1':
    outpath = outpath + 'spkTemplate_soundResp_cellSelection1/'
    cluster_path = cluster_path + 'spkTemplate_soundResp_cellSelection1/'
    
    
# figure ids

if cellSelection == '':
    fig1ID = 'Fig4G'
    fig2ID = 'Fig4H'
    fig3ID = 'Fig4K'
    loadname = 'clustering_data_default_results'
    savename = 'responseSim_data_default'
    
if cellSelection == '_spkTemplate_soundResp_cellSelection1':
    fig3ID = 'FigS8F'
    loadname = 'clustering_data_alt_cellSelection_results'
    savename = 'responseSim_data_alt_cellSelection'


#%% initialize

### make output directory ###
if os.path.isdir(outpath) == False:
    os.makedirs(outpath)

### data name
data_name = '' + cellSelection

nSessions = len(sessions_to_run)
responseMatrix_unsorted = [None]*nSessions
responseSimMatrix_sorted = [None]*nSessions
responseSim_cluster_true = [None]*nSessions
responseSim_cluster_perm = [None]*nSessions
pVal_responseSim_cluster_perm = [None]*nSessions

#%% load clustering info

clusterInfo = loadmat(('%s%s.mat' % (cluster_path, loadname)), simplify_cells=True)
removedCells_allSessions = clusterInfo['params']['removedCells_allSessions']
sigLevel_responsiveCells = clusterInfo['params']['sigLevel_responsiveCells']
clusterIDs_allSessions = clusterInfo['clusterIDs']
goodClusters_allSessions = clusterInfo['goodClusters_allSessions']


#%% loop over sessions

for indSession, session in enumerate(sessions_to_run):
    
    ### load data
    psth_data = loadmat(('%spsth_allTrials_%s_windLength0.100s%s.mat' % (psth_path, session, data_name)), simplify_cells=True)
    
    
    ### clustering
    clusterID = clusterIDs_allSessions[indSession]
    if np.size(removedCells_allSessions) == 0:
        removed_cells = np.array([])
    else:
        removed_cells = removedCells_allSessions[indSession]
    goodClusters = goodClusters_allSessions[indSession]
    

    ### psth
    t_psth = psth_data['t_window'].copy()
    trialAvg_gain_alt = psth_data['trialAvg_gain_alt'].copy()
    
    ### compute response similarity
    resp_pre_vs_post = fcn_significant_preStim_vs_postStim(psth_data, sigLevel_responsiveCells)
    response_vec = resp_pre_vs_post['resp_vector']
    if np.size(removed_cells) > 0:
        response_vec = np.delete(response_vec, removed_cells, 0)      
    if responseSim_type == 'pearson':
        responseSim = np.corrcoef(response_vec)
    else:
        responseSim = fcn_cosineSim_respVectors(response_vec)   
    
    ### sort response similarity by clusters
    responseSim_sorted = fcn_sorted_corrMatrix(responseSim, clusterID)
    np.fill_diagonal(responseSim_sorted,0)

    ### compute statistical significance of cluster response similarity
    if np.size(goodClusters) >= 1:
        
        within_minus_between_true, within_minus_between_shuf = \
            fcn_compare_sigCorr_within_vs_between_true_shuffled_clustering_alt(clusterID, responseSim, goodClusters, \
                                                                           withinClu_avgType, include_selfCorr, nPerms)
        
        pval = fcn_hierarchical_clustering.fcn_oneSided_pval(within_minus_between_true, within_minus_between_shuf, 'greater')  

    else:

        within_minus_between_true = 0.
        within_minus_between_shuf = np.ones(nPerms)*np.nan
        pval = np.nan
    
    ### store data
    responseMatrix_unsorted[indSession] = responseSim
    responseSimMatrix_sorted[indSession] = responseSim_sorted
    responseSim_cluster_true[indSession] = within_minus_between_true
    responseSim_cluster_perm[indSession] = within_minus_between_shuf
    pVal_responseSim_cluster_perm[indSession] = pval
    
    print(session, pval)
    

### save results
results = dict()
params = dict()

params['sessions'] = sessions_to_run
params['cluster_path'] = cluster_path
params['psth_path'] = psth_path
params['sigThresh'] = sigThresh
params['withinClu_avgType'] = withinClu_avgType
params['include_selfCorr'] = include_selfCorr
params['nPerms'] = nPerms
params['responseSim_type'] = responseSim_type
params['cellSelection'] = cellSelection
results['params'] = params
savemat(('%s%s_results.mat' % (outpath,savename)), results)

    
#%% PLOT RESULTS

figname = data_name + '_'


#%% sorted response similarity for example sessions

if cellSelection == '':

    count = 0
    for _, session in enumerate(sessions_plot):
        
        count+=1
        sessionInd = np.nonzero(sessions_to_run == session)[0][0]
        
        cmap = 'RdBu_r'
        cbar_label = ''
        cbar_lim = 0.9
    
        plt.rcParams.update({'font.size': 6})
        fig = plt.figure(figsize=(0.95,0.95))  
        ax = fig.add_axes([0.1, 0.1, 0.9, 0.9]) 
        fcn_plot_corrMatrix_withClusters(ax, responseMatrix_unsorted[sessionInd], clusterIDs_allSessions[sessionInd], goodClusters_allSessions[sessionInd], \
                                         'k', cbar_label, cmap, cbar_lim, linewidth=1, noTicks=False)
            
        ax.set_title('tuning sim.', fontsize=6)    
        
        plt.savefig((outpath + fig1ID + '_example%d.pdf' % count), bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()
    
    
#%% shuffled and true response similarity for example sessions

if cellSelection == '':

    count = 0
    for _, session in enumerate(sessions_plot):
        
        count+=1
        sessionInd = np.nonzero(sessions_to_run == session)[0][0]
        
        colorPlot = colorPlot = 'dimgrey'
    
        plt.rcParams.update({'font.size': 6})
        fig = plt.figure(figsize=(0.85,0.95))  
        ax = fig.add_axes([0.2, 0.2, 0.75, 0.72]) 
        ax.tick_params(axis='both', width=0.5)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        yTrue = responseSim_cluster_true[sessionInd]
        yPerm = responseSim_cluster_perm[sessionInd]
        
        if np.any(np.isnan(yPerm)):
            continue
        
        ax.hist(yPerm, color=colorPlot, label='permuted')
        ax.plot([yTrue, yTrue], [0, 220], color='lightseagreen', label='observed')
        ax.set_xlabel('cluster-based tuning sim.')
        ax.set_ylabel('no. permutations')
        ax.set_title('test statistic', fontsize=6)
    
        plt.savefig((outpath + fig2ID + '_example%d.pdf' % count), bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()

#%% shuffled and true response similarity for all sessions


plt.rcParams.update({'font.size': 7})
fig = plt.figure(figsize=(2.5,1))  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 
ax.tick_params(axis='both', width=0.5)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
    
for count, session in enumerate(sessions_to_run):

    x = np.array([count]) + 1
    yTrue = responseSim_cluster_true[count]
    yPerm = responseSim_cluster_perm[count]
    pVal = pVal_responseSim_cluster_perm[count]
    
    if pVal < sigThresh:
        ax.plot(x, yTrue, 'D', markersize=2, color='lightseagreen')
    else:
        ax.plot(x, yTrue, 'D', markersize=2, color='magenta')

    vPlot = ax.violinplot(yPerm, positions=x, vert=True, widths=0.6, showextrema=True, showmedians=True, quantiles=None, points=100)
    for pc in vPlot['bodies']:
        pc.set_facecolor('dimgrey')
        pc.set_edgecolor('black')
        pc.set_alpha(0.5)
    vPlot['cmaxes'].set_color('black')
    vPlot['cmaxes'].set_linewidth(0.5)
    vPlot['cmins'].set_color('black')
    vPlot['cmins'].set_linewidth(0.5)
    vPlot['cmedians'].set_color('black')
    vPlot['cbars'].set_color('black')
    vPlot['cbars'].set_linewidth(0.5)

ax.set_ylim([-0.25, 0.65])
ax.set_xticks([])
ax.set_ylabel('cluster-based\ntuning. sim.')
ax.set_xlabel('sessions')
ax.set_title('test statistic', fontsize=7)
plt.savefig((outpath + fig3ID + '.pdf'), bbox_inches='tight', pad_inches=0, transparent=True)
plt.close()
    
