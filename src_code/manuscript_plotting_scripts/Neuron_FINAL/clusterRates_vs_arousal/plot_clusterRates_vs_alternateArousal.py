'''
This script generates
    FigS3F
'''

#%% basic imports
import sys
import os
import numpy as np
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

#%% load my functions
func_path0 = global_settings.path_to_src_code + 'run_simulations/'
func_path1 = global_settings.path_to_src_code + 'functions/'

sys.path.append(func_path0)
sys.path.append(func_path1)

from fcn_simulation_loading import fcn_set_sweepParam_string
from fcn_simulation_setup import fcn_define_arousalSweep

#%% settings

# paths
sim_params_path = global_settings.path_to_src_code + 'run_simulations/'
loadSIM_path = global_settings.path_to_sim_output
fig_path = global_settings.path_to_manuscript_figs_final + 'clusterRates_vs_arousal/cluster_altArousal/'
loadMFT_path = global_settings.path_to_sim_output + 'MFT_sweep_JeePlus_arousalSweep/'
loadANALYSIS_path = global_settings.path_to_sim_output + 'clusterRates_numActiveClusters/'

# plotting
figureSize = (1.35, 1.35)
fontSize = 8

# simulation ID
simParams_fname = 'simParams_050925_clu'

# netowrk name
net_type = 'baseEIclu'

# sweep param name
sweep_param_name = 'zeroMean_sd_nu_ext_ee'

# window length
windLength = 25e-3

# rate threshold
rateThresh = 0.

# gain based
gain_based = True

# figure id
figID = 'FigS3F'

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
stim_shape = s_params['stim_shape']
stim_rel_amp = s_params['stim_rel_amp']
n_sweepParams = s_params['nParams_sweep']
swept_params_dict = s_params['swept_params_dict']

n_arousalLevels = np.size(swept_params_dict['param_vals1'])
all_arousal_levels = s_params['arousal_levels']*100

del s_params
del params


#%% LOAD EXAMPLE SIMULATION DATA

sweep_param_str_val = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, 0)

sim_fname =  ( ('%s_%s_sweep_%s_network%d_IC%d_stim%d_stimType_%s_stim_rel_amp%0.3f') % \
               ( simID, net_type, sweep_param_str_val, 0, 0, 0, stim_shape, stim_rel_amp) )
    
# simulation filename
sim_filename = sim_fname + '_simulationData.mat'
    
SIM_data = loadmat(loadSIM_path + sim_filename, simplify_cells=True)
sim_params = SIM_data['sim_params']
nClu = sim_params['p']
Jee_plus_sims = sim_params['JplusEE']
arousal_level = sim_params['arousal_levels']*100


#%% LOAD EXAMPLE ANALYSIS DATA

sweep_param_str_val = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, 0)

analysis_fname =  ( ('%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f') % \
                  ( simID, net_type, sweep_param_str_val, stim_shape, stim_rel_amp) )    
    
 
if gain_based:
    analysis_filename = analysis_fname + '__clusterRates_numActiveClusters_gainBased.mat'
else:
    analysis_filename = analysis_fname + '__clusterRates_numActiveClusters.mat'


ANALYSIS_data = loadmat(loadANALYSIS_path + analysis_filename, simplify_cells=True)
analysis_params = ANALYSIS_data['parameters']
nNets = analysis_params['nNets']
rate_thresh_array = analysis_params['rate_thresh']

    
#%% WINDOW LENGTH AND RATE THRESHOLD TO PLOT

indThresh_plot = np.nonzero(rate_thresh_array==rateThresh)[0][0]

rateThresh = rate_thresh_array[indThresh_plot]


#%% FIGURE FILENAMES

fig_fname =  ( ('%s_%s') % ( simID, net_type) ) 
fig_filename_sim = ((fig_fname + '_rateThresh%0.1fHz_windowStd%0.3fs_') % (rateThresh, windLength))

#%% SIMULATIONS

n_paramVals_sweep = np.size(swept_params_dict['param_vals1'])

# initialize arrays for quantities that we want to plot
activeRate_XActiveClusters_E = np.zeros((nClu+1, n_paramVals_sweep))
activeRate_XActiveClusters_E_error = np.zeros((nClu+1, n_paramVals_sweep))

# initialize arrays for quantities that we want to plot
inactiveRate_XActiveClusters_E = np.zeros((nClu+1, n_paramVals_sweep))
inactiveRate_XActiveClusters_E_error = np.zeros((nClu+1, n_paramVals_sweep))


# initialize arrays for quantities that we want to plot
activeRate_XActiveClusters_E_eachNet = np.zeros((nNets, nClu+1, n_paramVals_sweep))

# initialize arrays for quantities that we want to plot
inactiveRate_XActiveClusters_E_eachNet = np.zeros((nNets, nClu+1, n_paramVals_sweep))

# probability of n active clusters
prob_nActive_clusters_E_eachNet = np.zeros((nNets, nClu+1, n_paramVals_sweep))

prob_nActive_clusters_E = np.zeros((nClu+1, n_paramVals_sweep))
prob_nActive_clusters_E_error = np.zeros((nClu+1, n_paramVals_sweep))

popAvg_rate_E = np.zeros((n_paramVals_sweep))
popAvg_rate_E_error = np.zeros((n_paramVals_sweep))

popAvg_activeCluster_rate_e = np.zeros((n_paramVals_sweep))
popAvg_activeCluster_rate_e_error = np.zeros((n_paramVals_sweep))

popAvg_inactiveCluster_rate_e = np.zeros((n_paramVals_sweep))
popAvg_inactiveCluster_rate_e_error = np.zeros((n_paramVals_sweep))


# loop over perturbation
for ind_sweep_param in range(0, n_paramVals_sweep):
    

    sweep_param_str_val = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, ind_sweep_param)
      
        
    # analysis filename
    if gain_based:
        analysis_filename =  ( ('%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f__clusterRates_numActiveClusters_gainBased.mat') % \
                        ( simID, net_type, sweep_param_str_val, stim_shape, stim_rel_amp) )    
    else:
        analysis_filename =  ( ('%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f__clusterRates_numActiveClusters.mat') % \
                        ( simID, net_type, sweep_param_str_val, stim_shape, stim_rel_amp) )    

        
    # load the data
    ANALYSIS_data = loadmat(loadANALYSIS_path + analysis_filename, simplify_cells=True)

    # fill up the arrays
    
    activeRate_XActiveClusters_E[:, ind_sweep_param] = ANALYSIS_data['netAvg_avgRate_active_XActiveClusters_E'][:, indThresh_plot].copy()
    activeRate_XActiveClusters_E_error[:, ind_sweep_param] = ANALYSIS_data['netStd_avgRate_active_XActiveClusters_E'][:, indThresh_plot].copy()
    
    inactiveRate_XActiveClusters_E[:, ind_sweep_param] = ANALYSIS_data['netAvg_avgRate_inactive_XActiveClusters_E'][:, indThresh_plot].copy()
    inactiveRate_XActiveClusters_E_error[:, ind_sweep_param] = ANALYSIS_data['netStd_avgRate_inactive_XActiveClusters_E'][:, indThresh_plot].copy()

    prob_nActive_clusters_E[:, ind_sweep_param] = ANALYSIS_data['netAvg_prob_nActive_clusters_E'][:, indThresh_plot].copy()
    prob_nActive_clusters_E_error[:, ind_sweep_param] = ANALYSIS_data['netStd_prob_nActive_clusters_E'][:, indThresh_plot].copy()

    popAvg_rate_E[ind_sweep_param] = ANALYSIS_data['netAvg_popAvg_rate_E']
    popAvg_rate_E_error[ind_sweep_param] = ANALYSIS_data['netStd_popAvg_rate_E']
    

    popAvg_activeCluster_rate_e[ind_sweep_param]  = ANALYSIS_data['netAvg_popAvg_activeCluster_rate_e'][indThresh_plot].copy()
    popAvg_activeCluster_rate_e_error[ind_sweep_param]  = ANALYSIS_data['netStd_popAvg_activeCluster_rate_e'][indThresh_plot].copy()
    
    popAvg_inactiveCluster_rate_e[ind_sweep_param]  = ANALYSIS_data['netAvg_popAvg_inactiveCluster_rate_e'][indThresh_plot].copy()
    popAvg_inactiveCluster_rate_e_error[ind_sweep_param]  = ANALYSIS_data['netStd_popAvg_inactiveCluster_rate_e'][indThresh_plot].copy()    
    
    
    # loop over networks
    for indNet in range(0, nNets):
        
        if nNets == 1:
            activeRate_XActiveClusters_E_eachNet[indNet, :, ind_sweep_param] = ANALYSIS_data['trialAvg_avgRate_active_XActiveClusters_E'][:, indThresh_plot].copy()
            inactiveRate_XActiveClusters_E_eachNet[indNet, :, ind_sweep_param] = ANALYSIS_data['trialAvg_avgRate_inactive_XActiveClusters_E'][:, indThresh_plot].copy()
            prob_nActive_clusters_E_eachNet[indNet, :, ind_sweep_param] = ANALYSIS_data['trialAvg_prob_nActive_clusters_E'][:, indThresh_plot].copy()
            
        else:
            activeRate_XActiveClusters_E_eachNet[indNet, :, ind_sweep_param] = ANALYSIS_data['trialAvg_avgRate_active_XActiveClusters_E'][indNet, :, indThresh_plot].copy()
            inactiveRate_XActiveClusters_E_eachNet[indNet, :, ind_sweep_param] = ANALYSIS_data['trialAvg_avgRate_inactive_XActiveClusters_E'][indNet, :, indThresh_plot].copy()
            prob_nActive_clusters_E_eachNet[indNet, :, ind_sweep_param] = ANALYSIS_data['trialAvg_prob_nActive_clusters_E'][indNet, :, indThresh_plot].copy()


# most likely # active clusters at each perturbation
mostLikely_nActive_clusters_E = np.argmax(prob_nActive_clusters_E,0)


#%% PLOTTING


#%% plot in/active cluster rate for fixed Jee+ simulations; most likely # active clusters

plt.rcParams.update({'font.size': fontSize})
fig = plt.figure(figsize=figureSize)  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 
       

x = arousal_level
y = np.zeros(len(x))
yerr = np.zeros(len(x))

for i in range(0,len(y)):
    
    y[i] = activeRate_XActiveClusters_E[mostLikely_nActive_clusters_E[i], i]
    yerr[i] = activeRate_XActiveClusters_E_error[mostLikely_nActive_clusters_E[i], i]
    

ax.errorbar(x, y, yerr=yerr, xerr=None, color='lightseagreen', linewidth=1, fmt='none')
ax.plot(x, y, '-o',  color='lightseagreen', linewidth=1, markersize=2, label='active')


x = arousal_level
y = np.zeros(len(x))
yerr = np.zeros(len(x))

for i in range(0,len(y)):
    
    y[i] = inactiveRate_XActiveClusters_E[mostLikely_nActive_clusters_E[i], i]
    yerr[i] = inactiveRate_XActiveClusters_E_error[mostLikely_nActive_clusters_E[i], i]

ax.errorbar(x, y, yerr=yerr, xerr=None, color='darkviolet', linewidth=1, fmt='none')
ax.plot(x, y, '-o',  color='darkviolet', linewidth=1, markersize=2, label='inactive')
ax.set_yticks([0, 25, 50])
ax.set_xlim([-2,102])    
ax.set_xlabel('arousal level [%]')
ax.set_ylabel('E cluster rate [sp/s]')
ax.legend(fontsize=7, loc='upper right', frameon=False)
plt.savefig( ( (fig_path + figID + '.pdf') ), bbox_inches='tight', pad_inches=0, transparent=True)


