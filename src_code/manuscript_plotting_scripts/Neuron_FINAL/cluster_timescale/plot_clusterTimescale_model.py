
'''
Using figPlot = 'cluster_mainArousal', this script generates
    Fig6G
    
Using figPlot = 'cluster_altArousal', this script generates
    FigS3G
    
'''


#%% basic imports
import numpy as np
from scipy.io import loadmat
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

#%% settings

func_path = global_settings.path_to_src_code + 'functions/'
func_path0 = global_settings.path_to_src_code + 'run_simulations/'
loadSIM_path = global_settings.path_to_sim_output
sim_params_path = global_settings.path_to_src_code + 'run_simulations/'
loadANALYSIS_path = global_settings.path_to_sim_output + 'clusterTimescale/' 
rateThresh = 0
gain_based = True

figPlot = 'cluster_altArousal'

if figPlot == 'cluster_mainArousal':
    simParams_fname = 'simParams_051325_clu_spontLong'
    net_type = 'baseEIclu'
    nNetworks = 10
    sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'
    fig_path = global_settings.path_to_manuscript_figs_final + 'clusterTimescale_model/cluster_mainArousal/'
    figureSize = (1.4, 1.4)
    fontSize = 8
    figID = 'Fig6G'
    savename = 'clusterTimescale_mainArousal'

    
elif figPlot == 'cluster_altArousal':

    simParams_fname = 'simParams_050925_clu'
    net_type = 'baseEIclu'
    nNetworks = 5
    sweep_param_name = 'zeroMean_sd_nu_ext_ee'
    fig_path = global_settings.path_to_manuscript_figs_final + 'clusterTimescale_model/cluster_altArousal/'
    figureSize = (1.35, 1.35)
    fontSize = 8
    figID = 'FigS3G'
    savename = 'clusterTimescale_altArousal'
    
else:
    
    sys.exit('unknown figPlot')
    


#%% load custom functions
sys.path.append(func_path)
sys.path.append(func_path0)
from fcn_simulation_loading import fcn_set_sweepParam_string
from fcn_simulation_setup import fcn_define_arousalSweep

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
stim_type = s_params['stim_type']
stim_rel_amp = s_params['stim_rel_amp']
n_sweepParams = s_params['nParams_sweep']
swept_params_dict = s_params['swept_params_dict']
arousal_level = s_params['arousal_levels']*100

del s_params
del params

#%% number of arousal levels
nArousal_vals = np.size(arousal_level)

#%% set figure filenames

# make figure path
if os.path.isdir(fig_path) == False:
    os.makedirs(fig_path)

if gain_based:     
    fig_filename = ( '%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f_rateThresh%0.1fHz_gainBased_' % \
                   ( simID, net_type, sweep_param_name, stim_shape, stim_rel_amp, rateThresh ) )    
else:
    fig_filename = ( '%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f_rateThresh%0.1fHz_' % \
                   ( simID, net_type, sweep_param_name, stim_shape, stim_rel_amp, rateThresh ) )    

#%% load example files for setup

# swept parameters string
sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, 0) 
    
# simulation filename
sim_filename = ( '%s_%s_sweep_%s_network%d_IC%d_stim%d_stimType_%s_stim_rel_amp%0.3f_simulationData.mat' % \
               (simID, net_type, sweep_param_str, 0, 0, 0, stim_shape, stim_rel_amp ) )

# analysis filename
if gain_based:
    analysis_filename = ( '%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f__clusterTimescale_gainBased.mat' % \
                        ( simID, net_type, sweep_param_str, stim_shape, stim_rel_amp) )
else:
    analysis_filename = ( '%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f__clusterTimescale.mat' % \
                        ( simID, net_type, sweep_param_str, stim_shape, stim_rel_amp) )  

# example simulation data
SIM_data = loadmat(loadSIM_path + sim_filename, simplify_cells=True)
sim_params = SIM_data['sim_params']
nClu = sim_params['p']
Jee_plus_sims = sim_params['JplusEE']

# example analysis data
ANALYSIS_data = loadmat(loadANALYSIS_path + analysis_filename, simplify_cells=True)
analysis_params = ANALYSIS_data['parameters']
rate_thresh_array = analysis_params['rate_thresh']

# rate threshold to plot
indThresh_plot = np.nonzero(rate_thresh_array==rateThresh)[0][0]

#%% get quantities for plotting

# initialize arrays for quantities that we want to plot
clusterTimescale_E = np.zeros((nArousal_vals))
clusterTimescale_E_error = np.zeros((nArousal_vals))

clusterIAI_E = np.zeros((nArousal_vals))
clusterIAI_E_error = np.zeros((nArousal_vals))

# loop over perturbation
for ind_sweep_param in range(0, nArousal_vals):
    
    # name of swept parameters
    sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, ind_sweep_param) 

    # analysis filename
    if gain_based:
        analysis_filename = ( '%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f__clusterTimescale_gainBased.mat' % \
                        ( simID, net_type, sweep_param_str, stim_shape, stim_rel_amp ) )   
    else:
        analysis_filename = ( '%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f__clusterTimescale.mat' % \
                        ( simID, net_type, sweep_param_str, stim_shape, stim_rel_amp ) )               
    # load the data
    ANALYSIS_data = loadmat(loadANALYSIS_path + analysis_filename, simplify_cells=True)

    # fill up the arrays
    clusterTimescale_E[ind_sweep_param] = ANALYSIS_data['netAvg_clusterTimescale_E'][indThresh_plot].copy()
    clusterTimescale_E_error[ind_sweep_param] = ANALYSIS_data['netStd_clusterTimescale_E'][indThresh_plot].copy()

    clusterIAI_E[ind_sweep_param] = ANALYSIS_data['netAvg_clusterIAI_E'][indThresh_plot].copy()
    clusterIAI_E_error[ind_sweep_param] = ANALYSIS_data['netStd_clusterIAI_E'][indThresh_plot].copy()


#%% plot on the same figure


if figPlot == 'cluster_mainArousal':

    plt.rcParams.update({'font.size': fontSize})
    fig = plt.figure(figsize=figureSize)  
    ax2 = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 
    
    x = arousal_level.copy() 
    y = clusterIAI_E.copy()
    yerr = clusterIAI_E_error.copy()
    ax2.errorbar(x, y, yerr=yerr, xerr=None, color='darkviolet', linewidth=1, fmt='none', alpha=0.7)
    ax2.plot(x, y, '-o',  color='darkviolet', linewidth=1, markersize=2, alpha=0.7)
    ax2.set_xlim([-2,102])
    ax2.set_yticks([0.2, 0.7, 1.2, 1.7])
    ax2.set_ylim([0.15, 1.75])
    ax2.tick_params(axis='y', colors='darkviolet')
    ax2.set_xlabel('arousal level [%]')
    ax2.set_ylabel('cluster IAI [s]', color='darkviolet')
    
    ax3 = ax2.twinx()
    x = arousal_level.copy() 
    y = clusterTimescale_E.copy()
    yerr = clusterTimescale_E_error.copy()
    ax3.errorbar(x, y, yerr=yerr, xerr=None, color='lightseagreen', linewidth=1, fmt='none', alpha=0.7)
    ax3.plot(x, y, '-o',  color='lightseagreen', linewidth=1, markersize=2, alpha=0.7)
    ax3.set_yticks([0.1, 0.2, 0.3, 0.4])
    ax3.tick_params(axis='y', colors='lightseagreen')
    ax3.set_ylabel('cluster AT [s]',color='lightseagreen')
    
    plt.savefig('%s%s.pdf' % (fig_path, figID), bbox_inches='tight', pad_inches=0, transparent=True)
    

if figPlot == 'cluster_altArousal':
    
    plt.rcParams.update({'font.size': fontSize})
    fig = plt.figure(figsize=figureSize)  
    ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 
    
    x = arousal_level.copy() 
    y = clusterTimescale_E.copy()
    yerr = clusterTimescale_E_error.copy()
    
    ax.errorbar(x, y, yerr=yerr, xerr=None, color='k', linewidth=1, fmt='none')
    ax.plot(x, y, '-',  color='k', markersize=1, linewidth=1)
    ax.plot(x, y, 'o',  color=np.array([175,54,60])/256, markersize=2)
    
    
    ax.set_xlim([-2,102])
    ax.set_ylim([0.08, 0.42])
    ax.set_xlabel('arousal level [%]')
    ax.set_ylabel('cluster activation\ntimescale [s]')
    
    plt.savefig('%s%s.pdf' % (fig_path, figID), bbox_inches='tight', pad_inches=0, transparent=True)
