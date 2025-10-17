
'''
before running this script, run plot_clusters_evoked_model.py for both 
    figPlot = 'cluster'
    figPlot = 'hom'

this script will then generate
    Fig4A
'''


#%% basic imports
import numpy as np
from scipy.io import loadmat
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

#%% plot settings

# data paths
func_path0 = global_settings.path_to_src_code + 'run_simulations/'
func_path1 = global_settings.path_to_src_code + 'functions/'
sim_params_path = global_settings.path_to_src_code + 'run_simulations/'
data_path = global_settings.path_to_manuscript_figs_final + 'hierarchical_clustering_model/'
outpath = global_settings.path_to_manuscript_figs_final + 'hierarchical_clustering_model/'

# parameters
indNet = 0
nDraws = 1

### filenames for loading clustering results
loadname1 = ('clustering_model_clusterNetwork_net%d_results.mat' % indNet)
loadname2 = ('clustering_model_homNetwork_net%d_results.mat' % indNet)

### figure ID
fig1ID = 'Fig4A_cluster'
fig2ID = 'Fig4A_hom'



#%% output directory

# make output directory
if os.path.isdir(outpath) == False:
    os.makedirs(outpath)


#%% load data for clustered and uniform networks

clustered_data = loadmat(('%s%s' % (data_path, loadname1)), simplify_cells=True)
uniform_data = loadmat(('%s%s' % (data_path, loadname2)), simplify_cells=True)


#%% unpack items that we need

Qcluster_shuffle_allDraws_clu = clustered_data['Qcluster_shuffle_allDraws']
Qcluster_true_allDraws_clu = clustered_data['Qcluster_true_allDraws']

Qcluster_shuffle_allDraws_hom = uniform_data['Qcluster_shuffle_allDraws']
Qcluster_true_allDraws_hom = uniform_data['Qcluster_true_allDraws']

#%% plot cluster quality: shuffled and observed

for count in range(nDraws):
    
    # clustered

    plt.rcParams.update({'font.size': 6})
    fig = plt.figure(figsize=(0.55,0.95))  
    ax = fig.add_axes([0.2, 0.2, 0.75, 0.72]) 
    ax.tick_params(axis='both', width=0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    data1_clu = Qcluster_shuffle_allDraws_clu[count]
    data2_clu = Qcluster_true_allDraws_clu[count]

    vPlot = ax.violinplot(data1_clu, positions=np.array([1]), vert=True, widths=0.6, showextrema=True, showmedians=True, quantiles=None, points=100)
    
    
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
    

    vPlot = ax.violinplot(data2_clu, positions=np.array([2]), vert=True, widths=0.6, showextrema=True, showmedians=True, quantiles=None, points=100)
    
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

    ax.set_xticks([1,2])
    ax.set_xticklabels(['shuf', 'obs'])

    ax.set_ylabel('cluster quality')
    ax.set_ylim([-0.02, 0.55]) 
    
    ax.set_title('clustered\nnetwork',fontsize=6)
    

    plt.savefig((outpath + fig1ID + '.pdf'), bbox_inches='tight', pad_inches=0, transparent=True)

    
    # uniform
    plt.rcParams.update({'font.size': 6})
    fig = plt.figure(figsize=(0.6,0.95))  
    ax = fig.add_axes([0.2, 0.2, 0.75, 0.72]) 
    ax.tick_params(axis='both', width=0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    data1 = Qcluster_shuffle_allDraws_hom[count]
    data2 = Qcluster_true_allDraws_hom[count]
    
    vPlot = ax.violinplot(data1, positions=np.array([1]), vert=True, widths=0.6, showextrema=True, showmedians=True, quantiles=None, points=100)
    
    
    for pc in vPlot['bodies']:
        pc.set_facecolor('lightsteelblue')
        pc.set_edgecolor('lightsteelblue')
        pc.set_alpha(1)
        pc.set_label('uniform shuf.')
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
        pc.set_facecolor('magenta')
        pc.set_edgecolor('magenta')
        pc.set_alpha(0.75)
        pc.set_label('uniform obs.')
    vPlot['cmaxes'].set_color('black')
    vPlot['cmaxes'].set_linewidth(0.5)
    vPlot['cmins'].set_color('black')
    vPlot['cmins'].set_linewidth(0.5)
    vPlot['cmedians'].set_color('black')
    vPlot['cmedians'].set_linewidth(0.5)
    vPlot['cbars'].set_color('black')
    vPlot['cbars'].set_linewidth(0.5)
    
    ax.set_xticks([1,2])
    ax.set_xticklabels(['shuf', 'obs'])

    ax.set_ylim([-0.02, 0.55])    
    ax.set_yticklabels([])
    ax.set_title('uniform\nnetwork',fontsize=6)

    plt.savefig((outpath + fig2ID + '.pdf'), bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()