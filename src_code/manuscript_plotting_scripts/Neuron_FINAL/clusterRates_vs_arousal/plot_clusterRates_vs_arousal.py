
'''
This script generates
    Fig6A
    Fig6B
    FigS5C
    FigS5C_legend
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
from matplotlib import cm
from matplotlib import font_manager
font_path = global_settings.path_to_plotting_font
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()
plt.rcParams["mathtext.default"]="regular"
plt.rcParams['axes.linewidth'] = 0.5

#%% load my functions

func_path = global_settings.path_to_src_code + 'functions/'
sys.path.append(func_path)
from fcn_simulation_loading import fcn_set_sweepParam_string

#%% settings

### paths 
simParams_mft_path = global_settings.path_to_sim_output + 'simParams_mft/'
loadMFT_path = global_settings.path_to_sim_output + 'MFT_sweep_JeePlus_arousalSweep/'
loadANALYSIS_path = global_settings.path_to_sim_output + 'clusterRates_numActiveClusters/'
loadSIM_path = global_settings.path_to_sim_output
fig_path = global_settings.path_to_manuscript_figs_final + 'clusterRates_vs_arousal/cluster_mainArousal/'


### simulation params

# simulation ID
simID = '051300002025_clu'  

# netowrk name
net_type = 'baseEIclu'

# stim shape
stim_shape = 'diff2exp'

# relative stimulation amplitude
stim_rel_amp = 0.05

# sweep param name
sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'

# number of swept parameters
n_sweepParams = 3
    
# window length
windLength = 25e-3

# rate threshold
rateThresh = 0.

# gain based
gain_based = True


### MFT params

mftID = '051300002025_clu' 

# mft reduced
mft_reduced = True

# plot uniform
plot_uniform = False


### plotting params
figureSize = (1.9, 1.55)
fontSize = 8
fig1ID = 'Fig6B'
fig2ID = 'Fig6A'
fig3ID = 'FigS5C'
fig4ID = 'FigS5C_legend'


#%% make output directory
if os.path.isdir(fig_path) == False:
    os.makedirs(fig_path)



#%% LOAD EXAMPLE MFT DATA

mft_fname =  ( ('%s_%s_%s_%0.3fpct_stimType_%s_stim_rel_amp%0.3f') % \
               ( simID, net_type, sweep_param_name, 0., stim_shape, stim_rel_amp) ) 
    
# mft filename
if mft_reduced == True:
    mft_filename = mft_fname + '_reducedMFT_noDisorder_sweepJeePlus.mat' 
        
else:
    sys.exit('have only run reduced MFT')
    
MFT_data = loadmat(loadMFT_path + mft_filename, simplify_cells=True)

n_paramVals_mft = MFT_data['sim_params']['n_paramVals_mft']
swept_params_dict_sims = MFT_data['sim_params']['swept_params_dict_sims']

JeePlus_backSweep_results = MFT_data['JeePlus_backSweep_results']
JeePlus_forSweep_results = MFT_data['JeePlus_forSweep_results']

mft_params = MFT_data['mft_params']
n_activeClusters_sweep = mft_params['n_active_clusters_sweep']
JplusEE_back = JeePlus_backSweep_results['JplusEE_back']
JplusEE_for = JeePlus_forSweep_results['JplusEE_for']

n_nActive_clusters = np.size(n_activeClusters_sweep)
n_JplusEE_back = np.size(JplusEE_back)
n_JplusEE_for = np.size(JplusEE_for)

ind_param_plotMFT = np.array([0, int(n_paramVals_mft/2), -1])

arousal_level_mft = MFT_data['sim_params']['arousal_levels']*100


#%% LOAD EXAMPLE SIMULATION DATA

sweep_param_str_val = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict_sims, 0)

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

sweep_param_str_val = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict_sims, 0)

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
fig_filename_mft = (( fig_fname + '_rateThresh%0.1fHz_windowStd%0.3fs_') % (rateThresh, windLength))

#%% SIMULATIONS

n_paramVals_sweep = np.size(swept_params_dict_sims['param_vals1'])

activeRate_XActiveClusters_E = np.zeros((nClu+1, n_paramVals_sweep))
activeRate_XActiveClusters_E_error = np.zeros((nClu+1, n_paramVals_sweep))

inactiveRate_XActiveClusters_E = np.zeros((nClu+1, n_paramVals_sweep))
inactiveRate_XActiveClusters_E_error = np.zeros((nClu+1, n_paramVals_sweep))

prob_nActive_clusters_E = np.zeros((nClu+1, n_paramVals_sweep))
prob_nActive_clusters_E_error = np.zeros((nClu+1, n_paramVals_sweep))


# loop over perturbation
for ind_sweep_param in range(0, n_paramVals_sweep):
    

    sweep_param_str_val = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict_sims, ind_sweep_param)
      
        
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


# most likely # active clusters at each perturbation
mostLikely_nActive_clusters_E = np.argmax(prob_nActive_clusters_E,0)


#%% MFT

# initialize
activeRate_XActiveClusters_E_mftBack = np.zeros((n_JplusEE_back, n_nActive_clusters, n_paramVals_mft))
activeRate_XActiveClusters_E_mftFor = np.zeros((n_JplusEE_for, n_nActive_clusters, n_paramVals_mft))

inactiveRate_XActiveClusters_E_mftBack = np.zeros((n_JplusEE_back, n_nActive_clusters, n_paramVals_mft))
inactiveRate_XActiveClusters_E_mftFor = np.zeros((n_JplusEE_for, n_nActive_clusters, n_paramVals_mft))

# loop over perturbation
for ind_sweep_param in range(0, n_paramVals_mft):
         
    # arousal level
    arousal_level_mft_ind = arousal_level_mft[ind_sweep_param]

    # analysis filename
    mft_filename =  ( ('%s_%s_%s_%0.3fpct_stimType_%s_stim_rel_amp%0.3f_reducedMFT_noDisorder_sweepJeePlus.mat') % \
                   ( simID, net_type, sweep_param_name, arousal_level_mft_ind, stim_shape, stim_rel_amp) )    

    # load the data
    MFT_data = loadmat(loadMFT_path + mft_filename, simplify_cells=True)

    # backwards and forwards sweep
    JeePlus_backSweep_results = MFT_data['JeePlus_backSweep_results']
    JeePlus_forSweep_results = MFT_data['JeePlus_forSweep_results']
    
    # unpack 
    nu_bar_e_back = JeePlus_backSweep_results['nu_e_backSweep']
    nu_bar_e_for = JeePlus_forSweep_results['nu_e_forSweep']

    # backwards
    activeRate_XActiveClusters_E_mftBack[:, :, ind_sweep_param] = nu_bar_e_back[0, :, :]
    inactiveRate_XActiveClusters_E_mftBack[:, :, ind_sweep_param] = nu_bar_e_back[1, :, :]
    
    # forwards
    activeRate_XActiveClusters_E_mftFor[:, :, ind_sweep_param] = nu_bar_e_for[0, :, :]
    inactiveRate_XActiveClusters_E_mftFor[:, :, ind_sweep_param] = nu_bar_e_for[1, :, :]


#%% FIND VALUE OF J AT WHICH BASELINE SIMULATIONS AND MFT BEST MATCH

# most likely # active clusters baseline
mostLikely_nActive_base = int(np.argmax(prob_nActive_clusters_E[:, 0]))
indMFT_mostLikely_nActive_base = np.nonzero(n_activeClusters_sweep == mostLikely_nActive_base)[0]
activeRate_mostLikely_nActive_base = activeRate_XActiveClusters_E[mostLikely_nActive_base, 0]

# mft J+ with closest match
activeRate_mft_base = activeRate_XActiveClusters_E_mftBack[:, indMFT_mostLikely_nActive_base, 0].copy()
mft_indJeePlus_back_simMatch = np.argmin(np.abs(activeRate_mostLikely_nActive_base - activeRate_mft_base))
mft_JeePlus_back_simMatch = JplusEE_back[mft_indJeePlus_back_simMatch]
print(mft_JeePlus_back_simMatch)

JeePlus_mft_plot = mft_JeePlus_back_simMatch


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
plt.savefig( ( (fig_path + fig1ID + '.pdf') ), bbox_inches='tight', pad_inches=0, transparent=True)


#%% plot in/active cluster rate vs perturbation for fixed Jee+ mft; most likely n active from simulations


ind_JeePlus_plot = np.argmin( np.abs(JplusEE_back - JeePlus_mft_plot) )

ind_nActivePlot = np.zeros( len(arousal_level) )
for i in range(0, len(arousal_level)):
    ind_nActivePlot[i] = np.nonzero(n_activeClusters_sweep == mostLikely_nActive_clusters_E[i])[0][0]

ind_nActivePlot = ind_nActivePlot.astype(int)

plt.rcParams.update({'font.size': fontSize})
fig = plt.figure(figsize=figureSize)  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 
    
### active
x = arousal_level
y = np.zeros(len(x))
for i in range(0,len(x)):
    indMFT = np.argmin(np.abs(arousal_level_mft - arousal_level[i]))
    y[i] = activeRate_XActiveClusters_E_mftBack[ind_JeePlus_plot, ind_nActivePlot[i], indMFT]
    
ax.plot( x, y, '-o', linewidth=1, markersize=2, color='lightseagreen', label='active')       


### inactive
x = arousal_level
y = np.zeros(len(x))
for i in range(0,len(x)):
    indMFT = np.argmin(np.abs(arousal_level_mft - arousal_level[i]))
    y[i] = inactiveRate_XActiveClusters_E_mftBack[ind_JeePlus_plot, ind_nActivePlot[i], indMFT]
    
ax.plot( x, y, '-o', linewidth=1, markersize=2, color='darkviolet', label='inactive')   


### uniform
if plot_uniform == True:
    yuni_active = np.zeros(len(x))
    yuni_inactive = np.zeros(len(x))
    for i in range(0,len(x)):
        indMFT = np.argmin(np.abs(arousal_level_mft - arousal_level[i]))
        yuni_active[i] = np.flipud(activeRate_XActiveClusters_E_mftFor[:, ind_nActivePlot[i], indMFT])[ind_JeePlus_plot]
        yuni_inactive[i] = np.flipud(inactiveRate_XActiveClusters_E_mftFor[:, ind_nActivePlot[i], indMFT])[ind_JeePlus_plot]
        
    ax.plot( x, yuni_active, '-o', linewidth=1, markersize=2, color='gray', label='uniform')       
    ax.plot( x, yuni_inactive, '-o', linewidth=1, markersize=2, color='gray')       

ax.set_yticks([0, 25, 50])
ax.set_xlim([-2,102])    
ax.set_xlabel('arousal level [%]')
ax.set_ylabel('E cluster rate [sp/s]')
ax.legend(fontsize=7, loc='upper right', frameon=False)
plt.savefig( ( (fig_path + fig2ID + '.pdf') ), bbox_inches='tight', pad_inches=0, transparent=True)


#%% probability of x active clusters vs perturbation


plt.rcParams.update({'font.size': fontSize})
fig = plt.figure(figsize=figureSize)  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8])  

arousalPlot = np.array([0, 20, 30, 40, 50, 60, 80, 100])

cmap = cm.get_cmap('Spectral', len(arousalPlot))
cmap = cmap(range(len(arousalPlot)))
       
for count, val in enumerate(arousalPlot):
    
    indArousal = np.argmin(np.abs(arousal_level - val))
    
    x = np.arange(0, nClu+1)
    y = prob_nActive_clusters_E[:, indArousal]
    yerr = prob_nActive_clusters_E_error[:, indArousal]
    ax.errorbar(x, y, yerr=yerr, xerr=None, linewidth=1, color=cmap[count,:], fmt='none')
    ax.plot(x, y, '-o', color=cmap[count,:], linewidth=1, markersize=2, label=('arousal = %d' % (arousal_level[indArousal])))
    
ax.set_xticks(x[::4])
ax.set_yticks([0, 0.25, 0.5])
ax.set_xlabel('number active clusters $(n_{A})$')
ax.set_ylabel('probability')
plt.savefig( ( (fig_path + fig3ID + '.pdf') ), bbox_inches='tight', pad_inches=0, transparent=True)


# plot legend labels
plt.rcParams.update({'font.size': 6})
fig = plt.figure(figsize=figureSize)  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8])  

arousalPlot = np.array([0, 20, 30, 40, 50, 60, 80, 100])

cmap = cm.get_cmap('Spectral', len(arousalPlot))
cmap = cmap(range(len(arousalPlot)))
       
for count, val in enumerate(arousalPlot):
    
    indArousal = np.argmin(np.abs(arousal_level - val))
    
    ax.plot(0, 0, 'o', color=cmap[count,:], linewidth=1, markersize=2, label=('arousal = %d %%' % (arousal_level[indArousal])))

ax.set_xticks([])
ax.set_yticks([])
ax.legend(fontsize=6, loc='upper right', frameon=True)
plt.savefig( ( (fig_path + fig4ID + '.pdf') ), bbox_inches='tight', pad_inches=0, transparent=True)

