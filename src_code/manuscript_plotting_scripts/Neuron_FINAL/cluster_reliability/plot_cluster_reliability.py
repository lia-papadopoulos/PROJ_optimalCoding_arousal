
'''
This script generates
    Fig7C
'''

#%% basic imports
import numpy as np
import sys
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
from matplotlib import font_manager
font_path = global_settings.path_to_plotting_font
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()
plt.rcParams["mathtext.default"]="regular"
plt.rcParams['axes.linewidth'] = 0.5

figureSize = (2.2, 1.9)
figureSize2 = (1.75, 1.5)
fontSize = 8

#%% settings

fig_path = global_settings.path_to_manuscript_figs_final + 'cluster_reliability/'
func_path0 = global_settings.path_to_src_code + 'functions/'
func_path1 = global_settings.path_to_src_code + 'run_simulations/'
load_path = global_settings.path_to_sim_output + 'num_active_targeted_nontargeted_clusters/'
sim_params_path =  global_settings.path_to_src_code + 'run_simulations/'
simParams_fname = 'simParams_051325_clu'
sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'
net_type = 'baseEIclu'
nNets = 10
gain_based = True
zscore = False
gainThresh_plot = 0.

figID = 'Fig7C'

#%% import custom functions
sys.path.append(func_path0) 
sys.path.append(func_path1) 
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
stim_shape = s_params['stim_shape']
stim_rel_amp = s_params['stim_rel_amp']
n_sweepParams = s_params['nParams_sweep']
swept_params_dict = s_params['swept_params_dict']

n_arousalLevels = np.size(swept_params_dict['param_vals1'])
arousal_level = s_params['arousal_levels']*100

del s_params
del params

#%% figures

# make output directory
if os.path.isdir(fig_path) == False:
    os.makedirs(fig_path)

# figure name
if gain_based == True:
    fig_filename = ( '%s_%s_sweep_%s_gainThresh%0.2f' %  (simID, net_type, sweep_param_name, gainThresh_plot) )
else:
    fig_filename = ( '%s_%s_sweep_%s_rateThresh%0.2f' %  (simID, net_type, sweep_param_name, gainThresh_plot) )

#%% initialize

netAvg_f_targeted_nontargeted_active_avgGain = np.zeros((n_arousalLevels))
netStd_f_targeted_nontargeted_active_avgGain = np.zeros((n_arousalLevels))

#%% loop over arousal
for ind_sweep_param in range(0, n_arousalLevels):
    
    # swept parameter name + value as a string
    sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, ind_sweep_param)

    # analysis filename
    if gain_based:
        
        if zscore:
            analysis_filename = ( '%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f__numActive_targeted_vs_nontargeted_clusters_gainBased_zscore.mat' % \
                                ( simID, net_type, sweep_param_str, stim_shape, stim_rel_amp) )
        else:
            analysis_filename = ( '%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f__numActive_targeted_vs_nontargeted_clusters_gainBased.mat' % \
                                ( simID, net_type, sweep_param_str, stim_shape, stim_rel_amp) )
                
    else:
        analysis_filename = ( '%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f__numActive_targeted_vs_nontargeted_clusters.mat' % \
                            ( simID, net_type, sweep_param_str, stim_shape, stim_rel_amp ) )          
        
    # load the data
    ANALYSIS_data = loadmat(load_path + analysis_filename, simplify_cells=True)
    ANALYSIS_params = ANALYSIS_data['parameters']
    
    # gain threshold array
    gain_thresh = ANALYSIS_params['gain_thresh']
    
    # find value of gain thresh that we want
    ind_gainThresh = np.argmin(np.abs(gain_thresh - gainThresh_plot))
    
    # cluster reliability
    netAvg_f_targeted_nontargeted_active_avgGain[ind_sweep_param] = ANALYSIS_data['netAvg_f_targeted_nontargeted_active_avgGain'][ind_gainThresh]
    netStd_f_targeted_nontargeted_active_avgGain[ind_sweep_param] = ANALYSIS_data['netStd_f_targeted_nontargeted_active_avgGain'][ind_gainThresh]


#%% PLOTTING


#%% cluster reliability

plt.rcParams.update({'font.size': fontSize})
fig = plt.figure(figsize=figureSize)  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 
color = [0.2, 0.2, 0.2]

x = arousal_level
y = netAvg_f_targeted_nontargeted_active_avgGain
yerr = netStd_f_targeted_nontargeted_active_avgGain
ax.errorbar(x, y, yerr=yerr, xerr=None, color=color, linewidth=1, fmt='none')
ax.plot(x, y, '-o',  color=color, linewidth=1, markersize=2)
ax.set_xlim([-2,102])
ax.set_xlabel('arousal level [%]')
ax.set_ylabel('cluster reliability $C_{r}$')
plt.savefig( ( ('%s%s.pdf') % (fig_path, figID) ), bbox_inches='tight', pad_inches=0, transparent=True) 
