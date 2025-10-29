"""
using figPlot = 'cluster_mainArousal', this script generates
    FigS2F

using figPlot = 'hom_mainArousal', this script generates
    FigS2G
    
using figPlot = 'cluster_altArousal', this script
    FigS3C
"""

#%%
import sys
import numpy as np
from scipy.io import loadmat
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

#%% settings

# which figure to plot
figPlot = 'cluster_altArousal'

# path to simulation parameters            
simParams_path = global_settings.path_to_src_code + 'run_simulations/'
                
# path to data
load_path = global_settings.path_to_sim_output + 'singleCell_tuning_to_perturbation/'


# set sim params based on which figure to plot
if figPlot == 'cluster_mainArousal':
    simParams_fname = 'simParams_051325_clu'
    net_type = 'baseEIclu' 
    nNets = 10
    sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'
    fig_path = global_settings.path_to_manuscript_figs_final + 'arousal_rate_correlations_model/cluster_mainArousal/'
    figSize = (1.1,1.75)
    figID = 'FigS2F'

elif figPlot == 'hom_mainArousal':
    simParams_fname = 'simParams_051325_hom'
    net_type = 'baseHOM' 
    nNets = 10
    sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'    
    fig_path = global_settings.path_to_manuscript_figs_final + 'arousal_rate_correlations_model/hom_mainArousal/'
    figSize = (1.1,1.75)
    figID = 'FigS2G'

elif figPlot == 'cluster_altArousal':
    simParams_fname = 'simParams_050925_clu'
    net_type = 'baseEIclu' 
    nNets = 5
    sweep_param_name = 'zeroMean_sd_nu_ext_ee' 
    fig_path = global_settings.path_to_manuscript_figs_final + 'arousal_rate_correlations_model/cluster_altArousal/'
    figSize = (1.1,1.35)
    figID = 'FigS3C'

else:
    sys.exit('invalid figPlot')    


# analysis parameters
sig_level = 0.05

#%% setup

### make output directory ###
if os.path.isdir(fig_path) == False:
    os.makedirs(fig_path)

#%% load parameters

sys.path.append(simParams_path)
s_params_data = importlib.import_module(simParams_fname) 
s_params = s_params_data.sim_params
del s_params_data


#%% unpack

simID = s_params['simID']
stim_shape = s_params['stim_shape']
stim_rel_amp = s_params['stim_rel_amp']
arousal_level = s_params['arousal_levels']


#%% set filenames

# beginning of filename
fname_begin = ( '%s_%s_sweep_%s' % (simID, net_type, sweep_param_name) )

# end of filename
fname_end = ( '_stimType_%s_stim_rel_amp%0.3f_' % (stim_shape, stim_rel_amp) )


#%% load the data

data = loadmat('%s%s%ssingleCell_tuning_to_perturbation.mat' % (load_path, fname_begin, fname_end), simplify_cells=True)


#%% unpack the data

params = data['params']
trialAvg_stimAvg_firingRate_Ecells_base = data['trialAvg_stimAvg_firingRate_cells_base']
corr_pert_rate_base = data['corr_pert_rate_base']
pval_pert_rate_base = data['pval_pert_rate_base']


#%% get quantities of interest

# number of cells
N = np.shape(corr_pert_rate_base)[0]


#%% fraction positively and negatively correlated

frac_posCorr_pert_rate = np.zeros(nNets)
frac_negCorr_pert_rate = np.zeros(nNets)

if nNets == 1:
    frac_posCorr_pert_rate[0] = np.size(np.nonzero( (pval_pert_rate_base < sig_level) & (corr_pert_rate_base > 0)  )[0])/N
    frac_negCorr_pert_rate[0] = np.size(np.nonzero( (pval_pert_rate_base < sig_level) & (corr_pert_rate_base < 0)  )[0])/N

else:
    for indNet in range(0, nNets):
        frac_posCorr_pert_rate[indNet] = np.size(np.nonzero( (pval_pert_rate_base[:, indNet] < sig_level) & (corr_pert_rate_base[:, indNet] > 0)  )[0])/N
        frac_negCorr_pert_rate[indNet] = np.size(np.nonzero( (pval_pert_rate_base[:, indNet] < sig_level) & (corr_pert_rate_base[:, indNet] < 0)  )[0])/N



#%% PLOT FRACTION OF POSITIVELY AND NEGATIVELY CORRELATED CELLS

plt.rcParams.update({'font.size': 8})
fig = plt.figure(figsize=figSize)  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 

# pos
y = np.mean(frac_posCorr_pert_rate*100)
yerr = np.std(frac_posCorr_pert_rate*100)
ax.bar(0, y, yerr=yerr, color='lightseagreen')

# neg
y = np.mean(frac_negCorr_pert_rate*100)
yerr = np.std(frac_negCorr_pert_rate*100)
ax.bar(1, y, yerr=yerr, color='darkviolet')

#labels
ax.set_xlim([-0.5, 1.5])
ax.set_xticks([0,1])
ax.set_xticklabels(['pos. corr.', 'neg. corr.'], rotation=45)
ax.set_yticks([0, 25, 50])
ax.set_ylabel('percent of cells')
plt.savefig( (('%s%s.pdf') % (fig_path, figID)), bbox_inches='tight', pad_inches=0, transparent=True)






