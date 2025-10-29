
'''
This script generates various figure panels depending on the figPlot setting:

    figPlot = 'cluster_mainArousal'
        Fig4E
    
    figPlot = 'cluster_mainArousal_supp'
        FigS4A
    
    figPlot = 'hom_mainArousal'
        Fig4F
    
    figPlot = 'hom_mainArousal_supp'
        FigS4B
    
    figPlot = 'cluster_altArousal'
        FigS3E
'''


#%% basic imports
import sys
import numpy as np
import os
from scipy.io import loadmat
import importlib

#%% import global settings file
sys.path.append('../../../')
import global_settings

#%% plotting
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import font_manager
font_path = '/home/liap/fonts/Arial.ttf'  # Your font path goes here
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()
plt.rcParams["mathtext.default"]="regular"
plt.rcParams['axes.linewidth'] = 0.5

#%% load my functions

func_path0 = global_settings.path_to_src_code + 'run_simulations/'
func_path1 = global_settings.path_to_src_code + 'functions/'

sys.path.append(func_path0)
sys.path.append(func_path1)
from fcn_decoding import fcn_maxAccuracy_duringStim
from fcn_simulation_loading import fcn_set_sweepParam_string
from fcn_simulation_setup import fcn_define_arousalSweep
from fcn_statistics import fcn_pctChange_max

#%% settings

fontSize = 8

sim_params_path = global_settings.path_to_src_code + 'run_simulations/'
load_path = global_settings.path_to_sim_output + 'decoding_analysis/'

windL = 100e-3
classifier = 'LinearSVC'
rate_thresh = 0.
figPlot = 'cluster_mainArousal'
    

if figPlot == 'cluster_mainArousal':
    simParams_fname = 'simParams_051325_clu'
    sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'
    nNetworks = 10
    net_type = 'baseEIclu'
    ensembleSize_plot = np.array([160])
    figureSize = (1.4, 1.4)
    fig_path = global_settings.path_to_manuscript_figs_final + 'decoding_model/cluster_mainArousal/'
    figID = 'Fig4E'
    savename = 'decoding_model_cluster_main_'


elif figPlot == 'cluster_mainArousal_supp':
    simParams_fname = 'simParams_051325_clu'
    sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'
    nNetworks = 10
    net_type = 'baseEIclu'
    ensembleSize_plot = np.array([1, 2, 4, 8, 16, 32])*19
    figureSize = (1.4, 1.4)
    fig_path = global_settings.path_to_manuscript_figs_final + 'decoding_model/cluster_mainArousal_supp/'
    figID = 'FigS4A'
    figID_legend = 'FigS4_legend'
    savename = 'decoding_model_cluster_supp_'
    


elif figPlot == 'hom_mainArousal':
    simParams_fname = 'simParams_051325_hom'
    sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'
    nNetworks = 10
    net_type = 'baseHOM'
    ensembleSize_plot = np.array([160])
    figureSize = (1.4, 1.4)
    fig_path = global_settings.path_to_manuscript_figs_final + 'decoding_model/hom_mainArousal/'
    figID = 'Fig4F'
    savename = 'decoding_model_hom_main_'


elif figPlot == 'hom_mainArousal_supp':
    simParams_fname = 'simParams_051325_hom'
    sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'
    nNetworks = 10
    net_type = 'baseHOM'
    ensembleSize_plot = np.array([1, 2, 4, 8, 16, 32])*19
    figureSize = (1.4, 1.4)
    fig_path = global_settings.path_to_manuscript_figs_final + 'decoding_model/hom_mainArousal_supp/'
    figID = 'FigS4B'
    figID_legend = 'FigS4_legend'
    savename = 'decoding_model_hom_supp_'

elif figPlot == 'cluster_altArousal':
    simParams_fname = 'simParams_050925_clu'
    sweep_param_name = 'zeroMean_sd_nu_ext_ee'
    nNetworks = 5
    net_type = 'baseEIclu'   
    ensembleSize_plot = np.array([160])
    figureSize = (1.35, 1.35)
    fig_path = global_settings.path_to_manuscript_figs_final + 'decoding_model/cluster_altArousal/'
    figID = 'FigS3E'
    savename = 'decoding_model_cluster_altArousal_supp_'

else:
    sys.exit('invalid figPlot')




#%% load sim parameters
sys.path.append(sim_params_path)
params = importlib.import_module(simParams_fname) 
s_params = params.sim_params

#%% arousal sweep
s_params = fcn_define_arousalSweep(s_params)

#%% unpack sim params
simID = s_params['simID']
nTrials = s_params['n_ICs']
stim_shape = s_params['stim_shape']
stim_rel_amp = s_params['stim_rel_amp']
Ne = s_params['N']*s_params['ne']
n_sweepParams = s_params['nParams_sweep']
swept_params_dict = s_params['swept_params_dict']
arousal_level = s_params['arousal_levels']*100

del s_params
del params

# arousal level
n_arousalLevels = np.size(arousal_level)
n_ensembleSizes = np.size(ensembleSize_plot)

#%% set filenames
drawNeurons_str = ('_rateThresh%0.2fHz' % rate_thresh)


#%% make output directory ###

if os.path.isdir(fig_path) == False:
    os.makedirs(fig_path)

#%% LOAD ONE DATA FILE TO GET ARRAY SIZES + PARAMETERS

# swept param string
sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, 0)     

filename = ( ( '%s%s_sweep_%s_network%d_windL%dms_ensembleSize%d%s_%s.mat') % \
                (load_path, simID, sweep_param_str, 0, windL*1000, ensembleSize_plot[0], drawNeurons_str, classifier) )


# load data
data = loadmat(filename, simplify_cells=True)
parameters = data['parameters']

# unpack data
t_windows = data['t_window']
stimOn = parameters['stimOn']

# number of windows
nWindows = np.size(t_windows)


#%% INITIALIZE ARRAYS

maxAccuracy_all = np.zeros((nNetworks, n_arousalLevels, n_ensembleSizes))
maxAccuracy_norm_all = np.zeros((nNetworks, n_arousalLevels, n_ensembleSizes))

#%% LOAD IN/PROCESS DATA

for paramInd in range(0, n_arousalLevels, 1):

    sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, paramInd)     

    for netInd in range(0, nNetworks, 1):
                 
        for popSizeInd, popSize in enumerate(ensembleSize_plot):
        
            # filename
            filename = ( ( '%s%s_sweep_%s_network%d_windL%dms_ensembleSize%d%s_%s.mat') % \
                        (load_path, simID, sweep_param_str, netInd, windL*1000, popSize, drawNeurons_str, classifier) )

            # load data
            data = loadmat(filename, simplify_cells=True)       
            t_window = data['t_window']
            accuracy = data['accuracy']
          
            # max accuracy
            maxAccuracy_all[netInd, paramInd, popSizeInd] = fcn_maxAccuracy_duringStim(t_window, accuracy, stimOn, t_window[-1])
            

#%% COMPUTE NORMALIZED ACCURACY

for netInd in range(0, nNetworks, 1):
             
    for popSizeInd, popSize in enumerate(ensembleSize_plot):

        maxAccuracy_norm_all[netInd, :, popSizeInd] = fcn_pctChange_max(maxAccuracy_all[netInd, :, popSizeInd])
        
              
#%% COMPUTE AVERAGES OVER NETWORKS

netAvg_maxAccuracy = np.mean(maxAccuracy_all, 0)
netStd_maxAccuracy = np.std(maxAccuracy_all, 0)

netAvg_maxAccuracy_norm = np.mean(maxAccuracy_norm_all, 0)
netStd_maxAccuracy_norm = np.std(maxAccuracy_norm_all, 0)


#%% PLOTTING


#%% plot all ensemble sizes on the same plot

if ( (figPlot == 'cluster_mainArousal_supp') or (figPlot == 'hom_mainArousal_supp') ):

        
    cmap = cm.get_cmap('plasma', len(ensembleSize_plot))
    cmap = cmap(range(len(ensembleSize_plot)))
    
    plt.rcParams.update({'font.size': fontSize})
    fig = plt.figure(figsize=figureSize)  
    ax = fig.add_axes([0.2, 0.2, 0.8, 0.8])
    
    for popSizeInd, popSize in enumerate(ensembleSize_plot):
    
        x = arousal_level
        y = netAvg_maxAccuracy[:, popSizeInd]
        yerr = netStd_maxAccuracy[:, popSizeInd]
        ax.errorbar(x, y, yerr=yerr, xerr=None, color=cmap[popSizeInd,:], linewidth=1, fmt='none')
        ax.plot(x, y, '-o',  color=cmap[popSizeInd,:], linewidth=1, markersize=2)
        
    ax.set_xlim([-2, 102])
    ax.set_xlabel('arousal level [%]')
    ax.set_ylabel('accuracy')
    plt.savefig(('%s%s.pdf' % (fig_path, figID)), bbox_inches='tight', pad_inches=0, transparent=True)


    # just plot legend
    plt.rcParams.update({'font.size': fontSize})
    fig = plt.figure(figsize=figureSize)  
    ax = fig.add_axes([0.2, 0.2, 0.8, 0.8])
    
    for popSizeInd, popSize in enumerate(ensembleSize_plot):
    
        cells_per_clu = popSize/19
        pct_sampled = (popSize/Ne)*100
        
        legend_label = ('pct. population sampled = %0.1f' % pct_sampled)
        
        ax.plot([0,0],[0,0], '-o', color=cmap[popSizeInd,:], linewidth=1, markersize=2, label= legend_label)
        
    ax.legend(fontsize=8, frameon=True)
    plt.savefig(('%s%s.pdf' % (fig_path, figID_legend)), bbox_inches='tight', pad_inches=0, transparent=True)


#%% plot ensemble sizes on different plots (norm accuracy)

if ( (figPlot == 'cluster_mainArousal') or (figPlot == 'hom_mainArousal') or (figPlot == 'cluster_altArousal') ):

    for popSizeInd, popSize in enumerate(ensembleSize_plot):

        plt.rcParams.update({'font.size': fontSize})
        fig = plt.figure(figsize=figureSize)  
        ax = fig.add_axes([0.2, 0.2, 0.8, 0.8])
        
        x = arousal_level
        y = netAvg_maxAccuracy_norm[:, popSizeInd]
        yerr = netStd_maxAccuracy_norm[:, popSizeInd]
        
        
        ax.errorbar(x, y, yerr=yerr, xerr=None, color='k', linewidth=1, fmt='none')
        ax.plot(x, y, '-',  color='k', linewidth=1)
        ax.plot(x, y, 'o',  color=np.array([175,54,60])/256, markersize=2)
        
        ax.set_xlim([-2, 102])
        ax.set_xlabel('arousal level [%]')
        ax.set_ylabel('% change in accuracy\n(relative to max)', multialignment='center')
        plt.savefig(('%s%s.pdf' % (fig_path, figID)), bbox_inches='tight', pad_inches=0, transparent=True)
    
