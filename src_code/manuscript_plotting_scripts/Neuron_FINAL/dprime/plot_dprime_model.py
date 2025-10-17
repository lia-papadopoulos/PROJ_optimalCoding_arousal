
'''
This script generates different figure panels depending on the value of figPlot

    figPlot = 'cluster_mainArousal'
        Fig5C
    
    figPlot = 'hom_mainArousal'
        Fig5D
    
    figPlot = 'cluster_altArousal'
        FigS3D

'''


#%% basic imports
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
from fcn_simulation_loading import fcn_set_sweepParam_string
from fcn_simulation_setup import fcn_define_arousalSweep
from fcn_statistics import fcn_pctChange_max

#%% SETTINGS

# paths
sim_params_path = global_settings.path_to_src_code + 'run_simulations/'
data_path = global_settings.path_to_sim_output + 'singleCell_dPrime/'

 
figPlot = 'hom_mainArousal'
    

if figPlot == 'cluster_mainArousal':
    simParams_fname = 'simParams_051325_clu'
    sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'
    nNetworks = 10
    net_type = 'baseEIclu'
    fig_path = global_settings.path_to_manuscript_figs_final + 'dprime_model/cluster_mainArousal/'
    figureSize = (1.4, 1.4)
    figID = 'Fig5C'
    savename = 'dprime_model_cluster_'

elif figPlot == 'hom_mainArousal':
    simParams_fname = 'simParams_051325_hom'
    sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'
    nNetworks = 10
    net_type = 'baseHOM'
    fig_path = global_settings.path_to_manuscript_figs_final + 'dprime_model/hom_mainArousal/'
    figureSize = (1.4, 1.4)
    figID = 'Fig5D'
    savename = 'dprime_model_hom_'

elif figPlot == 'cluster_altArousal':
    simParams_fname = 'simParams_050925_clu'
    sweep_param_name = 'zeroMean_sd_nu_ext_ee'
    nNetworks = 5
    net_type = 'baseEIclu'   
    fig_path = global_settings.path_to_manuscript_figs_final + 'dprime_model/cluster_altArousal/'
    figureSize = (1.35, 1.35)
    figID = 'FigS3D'
    savename = 'dprime_model_cluster_altArousal_'
    
else:
    sys.exit('invalid figPlot')



# analysis
windL = 100e-3
windStep = 20e-3
rate_thresh = 0
base_window = np.array([-0.8, 0.])
stimCells_only = False
fontSize = 8


#%% LOAD SIM PARAMETERS

sys.path.append(sim_params_path)
params = importlib.import_module(simParams_fname) 
s_params = params.sim_params

#%% AROUSAL SWEEP

s_params = fcn_define_arousalSweep(s_params)

#%% UNPACK

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


#%% SET FILENAMES

if os.path.isdir(fig_path) == False:
    os.makedirs(fig_path)


fname_end = ( '_stimType_%s_stim_rel_amp%0.3f_' % (stim_shape, stim_rel_amp) )

figname = ('%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f_windL%0.3f_rateThresh%0.1fHz' % \
          (simID, net_type, sweep_param_name, stim_shape, stim_rel_amp, windL, rate_thresh))
    
    
#%% LOAD ONE EXAMPLE FOR INITIALIZATION PURPOSES

sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, 0) 

filename = ('%s%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f_singleCell_dPrime_windL%0.3f.mat' % \
           (data_path, simID, net_type, sweep_param_str, stim_shape, stim_rel_amp, windL) )

dprime_data = loadmat(filename, simplify_cells=True)

stimAvg_firingRate = dprime_data['stimAvg_firingRate']
tWindow = dprime_data['t_window']


n_Cells = np.size(stimAvg_firingRate, 1)
n_Windows = np.size(tWindow)  
    
#%% INITIALIZE QUANTITIES

n_arousalLevels = np.size(arousal_level)

good_rate_cells = np.zeros((nNetworks), dtype='object')
stim_cells = np.zeros((nNetworks), dtype='object')

max_dPrime_all_good_cells = np.zeros((nNetworks, n_arousalLevels) )
dPrime_all_good_cells =  np.zeros((nNetworks, n_Windows, n_arousalLevels) )


#%% DETERMINE CELLS THAT PASS RATE CUT

for indNetwork in range(0, nNetworks):
    
    timeAvg_baseline_rate = np.zeros(( n_Cells, n_arousalLevels ))


    for indParam in range(0, n_arousalLevels):
    
        sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, indParam) 

        filename = ('%s%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f_singleCell_dPrime_windL%0.3f.mat' % \
                   (data_path, simID, net_type, sweep_param_str, stim_shape, stim_rel_amp, windL) )

        dprime_data = loadmat(filename, simplify_cells=True)
    
        t_window = dprime_data['t_window']
        stimAvg_firingRate = dprime_data['stimAvg_firingRate']
        if nNetworks > 1:
            stimCells = dprime_data['stimCells'][:, indNetwork].copy()
        else:
            stimCells = dprime_data['stimCells'].copy()
        stimCells_all = np.array([])
        for i in range(0, nStim):
            stimCells_all = np.append(stimCells_all, stimCells[i])

        # time points and cells
        n_tPts = np.size(t_window) 
        n_Cells = np.size(stimAvg_firingRate, 1)
        
        # baseline time windows
        base_bins = np.nonzero( (t_window <= base_window[1]) & (t_window >= base_window[0]) )[0]
        
        # baseline rate
        if nNetworks > 1:
            baseline_rate = stimAvg_firingRate[base_bins, :, indNetwork].copy()
        else:
            baseline_rate = stimAvg_firingRate[base_bins, :].copy()

        timeAvg_baseline_rate[:, indParam] = np.mean(baseline_rate, 0) 
        

    # cells with good baseline rate
    if stimCells_only:
        good_rate_cells[indNetwork] = np.unique(stimCells_all).astype(int)
    else:
        good_rate_cells[indNetwork] = np.nonzero(np.all( timeAvg_baseline_rate > rate_thresh, 1 ))[0]
        

#%% LOOP OVER SWEPT PARAMETER

for indParam in range(0, n_arousalLevels):
    
    sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, indParam) 

    filename = ('%s%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f_singleCell_dPrime_windL%0.3f.mat' % \
                (data_path, simID, net_type, sweep_param_str, stim_shape, stim_rel_amp, windL) )

    data = loadmat(filename, simplify_cells=True)
    stimAvg_dPrime = data['freqAvg_dprime']

    
    ## COMPUTE AVERAGE DPRIME EXCLUDING LOW RATE CELLS
    
    for indNet in range(0, nNetworks):

        if nNetworks > 1:
            
            for indCell in range(0, n_Cells):
                mean_dprime = np.mean(stimAvg_dPrime[base_bins, indCell, indNet])
                std_dprime = np.std(stimAvg_dPrime[base_bins, indCell, indNet])
                
            cellAvg_dprime_all_good_cells = np.nanmean(stimAvg_dPrime[:, good_rate_cells[indNet], indNet], axis=1)

        else:
            
            for indCell in range(0, n_Cells):
                mean_dprime = np.mean(stimAvg_dPrime[base_bins, indCell])
                std_dprime = np.std(stimAvg_dPrime[base_bins, indCell])

            cellAvg_dprime_all_good_cells = np.nanmean(stimAvg_dPrime[:, good_rate_cells[indNet]], axis=1)
        
        dPrime_all_good_cells[indNet, :, indParam] = cellAvg_dprime_all_good_cells
        max_dPrime_all_good_cells[indNet, indParam] = np.nanmax(cellAvg_dprime_all_good_cells, axis=0)
    

norm_max_dPrime_all_good_cells = np.zeros((nNetworks, n_arousalLevels))

    
for indNetwork in range(0, nNetworks):
        
    norm_max_dPrime_all_good_cells[indNetwork :] = fcn_pctChange_max(max_dPrime_all_good_cells[indNetwork, :])
        

#%% QUANTITIES TO PLOT

netAvg_max_dPrime_all_good_cells = np.mean(max_dPrime_all_good_cells, axis=0)
netSd_max_dPrime_all_good_cells = np.std(max_dPrime_all_good_cells, axis=0)

netAvg_norm_max_dPrime_all_good_cells = np.mean(norm_max_dPrime_all_good_cells, axis=0)
netSd_norm_max_dPrime_all_good_cells = np.std(norm_max_dPrime_all_good_cells, axis=0)


#%% PLOT DPRIME

plt.rcParams.update({'font.size': fontSize})
fig = plt.figure(figsize=figureSize)  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8])

x = arousal_level
y = netAvg_norm_max_dPrime_all_good_cells
yerr = netSd_norm_max_dPrime_all_good_cells 
ax.errorbar(x, y, yerr=yerr, xerr=None, color='k', linewidth=1, fmt='none')
ax.plot(x, y, '-',  color='k', markersize=1, linewidth=1)
ax.plot(x, y, 'o',  color=np.array([175,54,60])/256, markersize=2)

ax.set_xlim([-2, 102])

if simParams_fname == 'simParams_050925_clu':
    ax.set_yticks(np.arange(-40,5,20))
elif ( net_type == 'baseEIclu' ):
    ax.set_yticks(np.arange(-60, 5, 20))
elif ( net_type == 'baseHOM' ):
    ax.set_yticks(np.arange(-12, 4, 4))

    
ax.set_xlabel('arousal level [%]')
ax.set_ylabel('% change in cell avg. $D\'_{sc}$\n(relative to max)', multialignment='center')
plt.savefig(('%s%s.pdf' % (fig_path, figID)), bbox_inches='tight', pad_inches=0, transparent=True)


