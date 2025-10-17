
'''
This script generates
    FigS5A
    FigS5B
'''

#%% basic imports
import numpy as np
from scipy.io import loadmat
import sys
import importlib
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

func_path0 = global_settings.path_to_src_code + 'functions/'
func_path1 = global_settings.path_to_src_code + 'run_simulations/'

sys.path.append(func_path0)
sys.path.append(func_path1)
from fcn_simulation_loading import fcn_set_sweepParam_string
from fcn_simulation_setup import fcn_define_arousalSweep


#%% settings

### paths 
sim_params_path = global_settings.path_to_src_code + 'run_simulations/'
loadMFT_path = global_settings.path_to_sim_output + 'MFT_sweep_JeePlus_baseline/'
loadANALYSIS_path = global_settings.path_to_sim_output + 'clusterRates_numActiveClusters_sweepJeePlus/'
loadSIM_path = global_settings.path_to_sim_output
fig_path = global_settings.path_to_manuscript_figs_final + 'clusterRates_vs_JeePlus/'

### plotting
figureSize = (1.9, 1.55)
fontSize = 8

### simulation parameters

# sim params fname
simParams_fname = 'simParams_041725_clu_varyJEEplus'

# simulation ID
simID = '041700002025_clu'  

# netowrk name
net_type = 'baseEIclu'

# stim shape
stim_shape = 'diff2exp'

# relative stimulation amplitude
stim_rel_amp = 0.0

# sweep param name
sweep_param_name = 'JplusEE_sweep'

# number of swept parameters
n_sweepParams = 1
    
# window length
windLength = 25e-3

# rate threshold
rateThresh = 0.

# gain based
gain_based = True

# nActive array
nActive_array_plot_sims = np.arange(1,10)

# min probability active
min_probActive = 0.2

### mft parameters

# mft reduced
mft_reduced = True

# nActive array
nActive_array_plot_mft = np.arange(1,10)

# figure IDs
fig1ID = 'figS5B'
fig2ID = 'figS5B_legend'
fig3ID = 'figS5A'


#%% make output directory 

if os.path.isdir(fig_path) == False:
    os.makedirs(fig_path)


#%% LOAD MFT DATA

mft_fname =  ( ('%s_%s') % ( simID, net_type) ) 

  
# mft filename
if mft_reduced == True:
    mft_filename = mft_fname + '_reducedMFT_sweepJeePlus_baseline.mat' 
        
else:
    sys.exit('have only run reduced MFT')
    
MFT_data = loadmat(loadMFT_path + mft_filename, simplify_cells=True)


#%% LOAD EXAMPLE SIMULATION DATA

sys.path.append(sim_params_path)
params = importlib.import_module(simParams_fname) 
default_params = params.sim_params
default_params = fcn_define_arousalSweep(default_params)
swept_params_dict_sims = default_params['swept_params_dict']
JeePlus_special_sims = default_params['JplusEE']

del params
del default_params

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
sweep_param_array_sim = swept_params_dict_sims['param_vals1']

minJ = np.min(sweep_param_array_sim)
maxJ = np.max(sweep_param_array_sim)

activeRate_XActiveClusters_E = np.zeros((nClu+1, n_paramVals_sweep))
activeRate_XActiveClusters_E_error = np.zeros((nClu+1, n_paramVals_sweep))

inactiveRate_XActiveClusters_E = np.zeros((nClu+1, n_paramVals_sweep))
inactiveRate_XActiveClusters_E_error = np.zeros((nClu+1, n_paramVals_sweep))

prob_nActive_clusters_E = np.zeros((nClu+1, n_paramVals_sweep))
prob_nActive_clusters_E_error = np.zeros((nClu+1, n_paramVals_sweep))

popAvg_rate_E = np.zeros((n_paramVals_sweep))
popAvg_rate_E_error = np.zeros((n_paramVals_sweep))


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

    popAvg_rate_E[ind_sweep_param] = ANALYSIS_data['netAvg_popAvg_rate_E']
    popAvg_rate_E_error[ind_sweep_param] = ANALYSIS_data['netStd_popAvg_rate_E']
    

    
# most likely # active clusters at each perturbation
mostLikely_nActive_clusters_E = np.argmax(prob_nActive_clusters_E,0)


#%% UNPACK MFT DATA

mft_params = MFT_data['mft_params']
JeePlus_critical = MFT_data['JeePlus_critical'].copy()
JeePlus_backSweep_results = MFT_data['JeePlus_backSweep_results']
JeePlus_forSweep_results = MFT_data['JeePlus_forSweep_results']

n_activeClusters_sweep = mft_params['n_active_clusters_sweep'].copy()

JplusEE_back = JeePlus_backSweep_results['JplusEE_back'].copy()
nu_e_backSweep = JeePlus_backSweep_results['nu_e_backSweep'].copy()
n_activeClustersE_back = JeePlus_backSweep_results['n_activeClustersE_back'].copy()

JplusEE_for = JeePlus_forSweep_results['JplusEE_for'].copy()
nu_e_forSweep = JeePlus_forSweep_results['nu_e_forSweep'].copy()
n_activeClustersE_for = JeePlus_forSweep_results['n_activeClustersE_for'].copy()


#%% FIND VALUE OF J AT WHICH BASELINE SIMULATIONS AND MFT BEST MATCH

ind_JeePlus_special_sims = np.nonzero(sweep_param_array_sim == JeePlus_special_sims)[0]
mostLikely_nActive_JeePlus_special_sims = int(np.argmax(prob_nActive_clusters_E[:, ind_JeePlus_special_sims]))
activeRate_mostLikely_nActive_JeePlus_special_sims = activeRate_XActiveClusters_E[mostLikely_nActive_JeePlus_special_sims, ind_JeePlus_special_sims]

# mft J+ with closest match
indMFT_mostLikely_nActive_base = np.nonzero(n_activeClusters_sweep == mostLikely_nActive_JeePlus_special_sims)[0]
activeRate_mft = nu_e_backSweep[0, :, indMFT_mostLikely_nActive_base].copy()
mft_indJeePlus_back_simMatch = np.argmin(np.abs(activeRate_mft - activeRate_mostLikely_nActive_JeePlus_special_sims))
mft_JeePlus_back_simMatch = JplusEE_back[mft_indJeePlus_back_simMatch]
print(mft_JeePlus_back_simMatch)

JeePlus_special_mft = mft_JeePlus_back_simMatch


#%% PLOTTING


#%% plot in/active cluster rate vs JeePlus

plt.rcParams.update({'font.size': fontSize})
fig = plt.figure(figsize=figureSize)  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 

legend_vals = nActive_array_plot_sims.copy()      

cmap = cm.get_cmap('magma', np.size(nActive_array_plot_sims)+2)
cmap = cmap(range(np.size(nActive_array_plot_sims)+2))
cmap = cmap[1:,:]

for ind, ind_nActivePlot in enumerate(nActive_array_plot_sims):

    prob_nActive = prob_nActive_clusters_E[ind_nActivePlot, :].copy()
    prob_nActive[prob_nActive < min_probActive] = np.nan
    prob_nActive[prob_nActive >= min_probActive] = 1

    x = sweep_param_array_sim.copy()
    x[x < minJ] = np.nan
    x[x > maxJ] = np.nan
    
    y = activeRate_XActiveClusters_E[ind_nActivePlot, :]*prob_nActive
    yerr = activeRate_XActiveClusters_E_error[ind_nActivePlot, :]*prob_nActive
     
    ax.fill_between(x, y-yerr, y+yerr, color=cmap[ind_nActivePlot, :], alpha=0.3)
    ax.plot( x, y, '-o', color=cmap[ind, :], linewidth=0.75, markersize=2.75)

y = popAvg_rate_E
yerr = popAvg_rate_E_error

ax.fill_between(x, y-yerr, y+yerr, color='gray', linewidth=0.75)
ax.plot( x, y, '-', color='gray', linewidth=1)
    
ax.plot([JeePlus_special_sims, JeePlus_special_sims], [0,100], '--', color='black', label='sim $J^{EE}_{+}$')
ax.plot([JeePlus_special_mft, JeePlus_special_mft], [0,100], color='black', label='mft $J^{EE}_{+}$')
ax.legend(fontsize=6, frameon=False)

ax.set_xlim([minJ-0.25, maxJ+0.25])
ax.set_xticks([12, 14.5, 17, 19.5])
ax.set_ylim([0, 95])
ax.set_xlabel(r'E-to-E intracluster weight factor $J^{EE}_{+}$')
ax.set_ylabel('active E cluster rate [sp/s]')
plt.savefig( ( fig_path + fig1ID + '.pdf' ) , bbox_inches='tight', pad_inches=0, transparent=True)


#%% legend

plt.rcParams.update({'font.size': fontSize})
fig = plt.figure(figsize=figureSize)  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 

legend_vals = nActive_array_plot_sims.copy()      

cmap = cm.get_cmap('magma', np.size(nActive_array_plot_sims)+2)
cmap = cmap(range(np.size(nActive_array_plot_sims)+2))
cmap = cmap[1:,:]

for ind, ind_nActivePlot in enumerate(nActive_array_plot_sims):

    x = sweep_param_array_sim[0]    
    y = activeRate_XActiveClusters_E[ind_nActivePlot, 0]
             
    ax.plot( x, y, '-o', color=cmap[ind, :], linewidth=0.75, markersize=2.75, label=(r'$n_A = %d$' % ind_nActivePlot) )
    
y = popAvg_rate_E[0]
ax.plot( x, y, '-', color='gray', linewidth=0.75, label = 'uniform/pop. avg.')
ax.legend(fontsize=6, loc='upper left')
ax.set_yticks([])
ax.set_xticks([])
plt.savefig( ( fig_path + fig2ID + '.pdf' ) , bbox_inches='tight', pad_inches=0, transparent=True)


#%% plot rate of active E cluster vs Jplus without stability

plt.rcParams.update({'font.size': fontSize})
fig = plt.figure(figsize=figureSize)  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 

legend_vals = nActive_array_plot_mft.copy()      

cmap = cm.get_cmap('magma', np.size(nActive_array_plot_mft)+2)
cmap = cmap(range(np.size(nActive_array_plot_mft)+2))
cmap = cmap[1:,:]

for ind, nActive in enumerate(nActive_array_plot_mft):
    
    ind_nActivePlot = np.argmin( np.abs(n_activeClusters_sweep - nActive))
    
    for indPop in range(0, 1):
        
        x = JplusEE_back.copy()
        x[x <= JeePlus_critical[ind_nActivePlot] - 1e-6] = np.nan
        x[x < minJ] = np.nan
        x[x > maxJ] = np.nan
        
        y = nu_e_backSweep[indPop,:,ind_nActivePlot].copy()
        ax.plot(x, y, '-o', markersize=0.75, color=cmap[ind,:])
        
x = JplusEE_for.copy()
x[x < minJ] = np.nan
x[x > maxJ] = np.nan      
y = nu_e_forSweep[indPop,:,ind_nActivePlot].copy()
ax.plot(x, y, '-',  linewidth=0.75, color='gray')        

ax.plot([JeePlus_special_sims, JeePlus_special_sims], [0,100], '--', color='black', label='sim $J^{EE}_{+}$')
ax.plot([JeePlus_special_mft, JeePlus_special_mft], [0,100], color='black', label='mft $J^{EE}_{+}$')

ax.legend(fontsize=6, frameon=False)
ax.set_xlim([minJ-0.25, maxJ+0.25])
ax.set_xticks([12, 14.5, 17, 19.5])
ax.set_ylim([0, 95])
ax.set_xlabel(r'E-to-E intracluster weight factor $J^{EE}_{+}$')
ax.set_ylabel('active E cluster rate [sp/s]')
plt.savefig( ( fig_path + fig3ID + '.pdf' ) , bbox_inches='tight', pad_inches=0, transparent=True)


