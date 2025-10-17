
'''
This script generates 
    FigS6B
    FigS6C
'''


#%% basic imports
import sys
import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
import importlib
import scipy
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

#%% load my functions
func_path0 = global_settings.path_to_src_code + 'run_simulations/'
func_path = global_settings.path_to_src_code + 'functions/'

sys.path.append(func_path)
sys.path.append(func_path0)

from fcn_simulation_loading import fcn_set_sweepParam_string
from fcn_simulation_setup import fcn_define_arousalSweep
from fcn_simulation_setup import fcn_compute_cluster_assignments
from fcn_compute_firing_stats import Dict2Class

#%% settings

# paths
sim_params_path = global_settings.path_to_src_code + 'run_simulations/'
fig_path = global_settings.path_to_manuscript_figs_final + 'spectra_model/'
sim_path = global_settings.path_to_sim_output
data_path = global_settings.path_to_sim_output + 'spont_spikeSpectra_vsPerturbation/'

# simulation parameters
simParams_fname = 'simParams_051325_clu_spont'
base_mod_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'
net_type = 'baseEIclu'
nNetworks = 2

# analysis parameters
windL = 2500e-3
df_array = np.array([0.8, 1.6, 4])
df_plot = 1.6
rate_thresh = 1.
estimation_plot = 'mt'
dcSubtract_type_plot = 0
lowFreq_band = np.array([1,4])

figID1 = 'figS6B'
figID2 = 'figS6C'

#%% functions

def fcn_avgPower(power, freq):
    meanPower = np.mean(power)
    return meanPower

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
nTrials = s_params['n_ICs']
nStim = s_params['nStim']
stim_shape = s_params['stim_shape']
stim_rel_amp = s_params['stim_rel_amp']
n_sweepParams = s_params['nParams_sweep']
swept_params_dict = s_params['swept_params_dict']
arousal_level = s_params['arousal_levels']*100


del s_params
del params

    
#%% number of arousal levels
nArousal_vals = np.size(arousal_level)

#%% set filenames

figname = ('%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f_windL%0.3f_rateThresh%0.1fHz_%s_df%d' % \
          (simID, net_type, base_mod_name, stim_shape, stim_rel_amp, windL, rate_thresh, estimation_plot, df_plot))

#%% check number of stimuli

if nStim > 1:
    sys.exit('code only works for one stimulus')
    
    
#%% load example simulation for initialization

# filename
sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, base_mod_name, swept_params_dict, 0) 

sim_filename = ( sim_path + '%s_%s_sweep_%s_network%d_IC%d_stim%d_stimType_%s_stim_rel_amp%0.3f_simulationData.mat' % \
               ( simID, net_type, sweep_param_str, 0, 0, 0, stim_shape, stim_rel_amp) )
     
# load data
sim_data = loadmat(sim_filename, simplify_cells=True)                
s_params = Dict2Class(sim_data['sim_params'])
N = s_params.N_e + s_params.N_i
stimOn = s_params.stim_onset    
nClu = s_params.p
Ne = s_params.N_e


#%% get cells in stimulated clusters and cells that pass baseline rate cut

avg_units_allClusters = np.zeros((nNetworks), dtype='object')


for indNetwork in range(0, nNetworks):
    
    # cells with good baseline rate
    baseRate_allBaseMod = np.zeros((N, nArousal_vals))
            
    for indParam in range(0, nArousal_vals):
        
        sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, base_mod_name, swept_params_dict, indParam) 
        
        
        filename = ('%s%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f_spont_spikeSpectra_windL%0.3f.mat' % \
                   (data_path, simID, net_type, sweep_param_str, stim_shape, stim_rel_amp, windL) )

        data = loadmat(filename, simplify_cells=True)

                            
        # spike counts
        spkRate = data['avg_spkCount'][:, indNetwork]/windL
            
        # avg baseline rate
        baseRate_allBaseMod[:, indParam] = spkRate.copy()
            
            
    # cells with good firing rate
    good_rate_cells = np.nonzero(np.all( baseRate_allBaseMod >= rate_thresh, 1 ))[0]


    # get all clustered cells
    
    sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, base_mod_name, swept_params_dict, 0) 

    sim_filename = ( sim_path + '%s_%s_sweep_%s_network%d_IC%d_stim%d_stimType_%s_stim_rel_amp%0.3f_simulationData.mat' % \
                   ( simID, net_type, sweep_param_str, indNetwork, 0, 0, stim_shape, stim_rel_amp) )
         
    # load data
    sim_data = loadmat(sim_filename, simplify_cells=True)                
    s_params = Dict2Class(sim_data['sim_params'])
    if 'popSize_E' in sim_data.keys():
        popSize_E = sim_data['popSize_E'].copy()
        popSize_I = sim_data['popSize_I'].copy()
    else:
        popSize_E = sim_data['clust_sizeE'].copy()
        popSize_I = sim_data['clust_sizeI'].copy()
        
        
    # cluster assignment of every cell
    Ecluster_ids, Icluster_ids = fcn_compute_cluster_assignments(popSize_E, popSize_I)

    # all cluster cells
    allCluster_cells_E = np.nonzero(Ecluster_ids < nClu)[0]
    allCluster_cells_I = np.nonzero(Icluster_ids < nClu)[0] + Ne
    allCluster_cells = np.append(allCluster_cells_E, allCluster_cells_I)
        
    avg_units_allClusters[indNetwork] = np.intersect1d(good_rate_cells, allCluster_cells).astype(int)


#%% initialize quantities

# spectra
singleCell_norm_spectra  = np.ones((nNetworks,nArousal_vals), dtype='object')*np.nan
cellAvg_norm_spectra = np.ones((nNetworks, nArousal_vals), dtype='object')*np.nan
cellSem_norm_spectra = np.ones((nNetworks, nArousal_vals), dtype='object')*np.nan
cellAvg_lowFreqPower = np.ones((nNetworks, nArousal_vals))*np.nan

# frequency
frequency_spectra = np.ones((nNetworks), dtype='object')*np.nan


for indNetwork in range(0, nNetworks):
    
    # good cells
    cells = avg_units_allClusters[indNetwork].copy()

    for indParam in range(0, nArousal_vals):
        
        sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, base_mod_name, swept_params_dict, indParam) 
        
        filename = ('%s%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f_spont_spikeSpectra_windL%0.3f.mat' % \
                   (data_path, simID, net_type, sweep_param_str, stim_shape, stim_rel_amp, windL) )
            
        data = loadmat(filename, simplify_cells=True)

        # unpack data
        if estimation_plot == 'raw':
            
            norm_power_spectra = data['norm_power_spectra_raw'].copy()
            frequency = data['frequency_spectra_raw'].copy()
            singleCell_norm_spectra[indNetwork, indParam] = norm_power_spectra[cells, indNetwork, :, dcSubtract_type_plot]
        
        elif estimation_plot == 'mt':
            
            norm_power_spectra = data['norm_power_spectra'].copy()
            frequency = data['frequency_spectra'].copy()     
            df_array = data['parameters']['df_array']
            
            # value of bandwidth parameter
            if len(df_array) > 1:
                ind_df = np.nonzero(df_array == df_plot)[0][0]
                singleCell_norm_spectra[indNetwork, indParam] = norm_power_spectra[cells, indNetwork, :, ind_df, dcSubtract_type_plot]
            else:
                singleCell_norm_spectra[indNetwork, indParam] = norm_power_spectra[cells, indNetwork, :, dcSubtract_type_plot]

        # cell avg
        cellAvg_norm_spectra[indNetwork, indParam] = np.nanmean(singleCell_norm_spectra[indNetwork, indParam], 0)
        cellSem_norm_spectra[indNetwork, indParam] = scipy.stats.sem(singleCell_norm_spectra[indNetwork, indParam], axis=0, nan_policy='omit') 
        

        # frequency indices for low frequency band ------------------------------
        lowFreq_indLow = np.argmin(np.abs(frequency - lowFreq_band[0]))
        lowFreq_indHigh = np.argmin(np.abs(frequency - lowFreq_band[1]))       

        # low frequency power
        nCells = np.size(cells)
        
        lowFreq_power = np.ones((nCells))*np.nan        

        for indCell in range(0, nCells):
            
            lowFreq_power[indCell] = fcn_avgPower(singleCell_norm_spectra[indNetwork, indParam][indCell, lowFreq_indLow:lowFreq_indHigh+1], \
                                                  frequency[lowFreq_indLow:lowFreq_indHigh+1] )

        # cell-averaged low frequency power
        cellAvg_lowFreqPower[indNetwork, indParam] = np.nanmean(lowFreq_power)
        

    # frequency values
    frequency_spectra[indNetwork] = frequency.copy()
    

#%% average across networks

netAvg_cellAvg_lowFreqPower = np.nanmean(cellAvg_lowFreqPower, 0)
netStd_cellAvg_lowFreqPower = np.nanstd(cellAvg_lowFreqPower, 0)


#%% save the results

params = {}
results = {}

params['windL'] = windL
params['rate_thresh'] = rate_thresh
params['df'] = df_plot
params['estimation'] = estimation_plot
params['dcSubtract_type'] = dcSubtract_type_plot
params['lowFreq_band'] = lowFreq_band
params['nNetworks'] = nNetworks
params['nStim'] = nStim
params['stim_rel_amp'] = stim_rel_amp
params['stim_shape'] = stim_shape
params['nTrials'] = nTrials
params['net_type'] = net_type
params['simID'] = simID
params['data_path'] = data_path
params['fig_path'] = fig_path
params['sweep_param_name'] = base_mod_name
params['simParams_fname'] = simParams_fname
params['sim_params_path'] = sim_params_path
params['sim_path'] = sim_path
params['data_path'] = data_path
results['params'] = params

save_filename = (fig_path + 'spectra_model' + '_params.mat')      
savemat(save_filename, results) 

#%% cell avg spectra from one network

indNetwork = 0

plt.rcParams.update({'font.size': 7})
fig = plt.figure(figsize=(1.,1.))  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 

x = frequency_spectra[indNetwork].copy()
y = cellAvg_norm_spectra[indNetwork, 0].copy()
yerr = cellSem_norm_spectra[indNetwork, 0].copy()
ax.fill_between(x, y-yerr, y+yerr, where=None, color='steelblue', alpha=0.3)
ax.plot(x, y, color='steelblue', label='low arousal')


x = frequency_spectra[indNetwork].copy()
y = cellAvg_norm_spectra[indNetwork, 4].copy()
yerr = cellSem_norm_spectra[indNetwork, 4].copy()
ax.fill_between(x, y-yerr, y+yerr, where=None, color='cornflowerblue', alpha=0.3)
ax.plot(x, y, color='cornflowerblue', label='mid arousal')


x = frequency_spectra[indNetwork].copy()
y = cellAvg_norm_spectra[indNetwork, -1].copy()
yerr = cellSem_norm_spectra[indNetwork, -1].copy()
ax.fill_between(x, y-yerr, y+yerr, where=None, color='lightsteelblue', alpha=0.3)
ax.plot(x, y, color='lightsteelblue', label='high arousal')    

ax.set_xlim([0.8, 500])
ax.set_yticks([0.5, 2.5, 4.5])
ax.set_xscale('log')

ax.set_xlabel('freq. [Hz]')
ax.set_ylabel('norm. power')
ax.legend(loc='upper right', fontsize=5, frameon=False)
    

plt.savefig('%s%s.pdf' % (fig_path, figID1), bbox_inches='tight', pad_inches=0, transparent=True)



#%% cell avg low frequency power vs baseline modulation

plt.rcParams.update({'font.size': 7})
fig = plt.figure(figsize=(1.,1.))  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 

x = arousal_level.copy() 
y = netAvg_cellAvg_lowFreqPower.copy()
yerr = netStd_cellAvg_lowFreqPower.copy()

ax.errorbar(x, y, yerr=yerr, xerr=None, color='k', linewidth=0.5, fmt='none')
ax.plot(x, y, '-',  color='k', markersize=1, linewidth=0.5)
ax.plot(x, y, 'o',  color=np.array([175,54,60])/256, markersize=1)

ax.set_xticks([0,100])
ax.set_xlim([-2,102])
ax.set_xlabel('arousal level [%]')
ax.set_ylabel('$\\langle P_{L,spont} \\rangle$')

plt.savefig('%s%s.pdf' % (fig_path, figID2), bbox_inches='tight', pad_inches=0, transparent=True)

