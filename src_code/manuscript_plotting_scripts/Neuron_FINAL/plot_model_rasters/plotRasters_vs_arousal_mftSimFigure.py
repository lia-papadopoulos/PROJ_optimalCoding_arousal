
'''
This script generates different figure panels depending on the value of figPlot

    figPlot = 'cluster_mainArousal'
        Fig6F
        
    figPlot = 'cluster_altArousal'
        FigS3B
'''

#%% standard imports
import sys
import numpy as np
from scipy.io import loadmat
import importlib
import os

#%% global settings
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

#%% load functions
functions_path0 = global_settings.path_to_src_code + 'run_simulations/'
functions_path1 = global_settings.path_to_src_code + 'functions/'

sys.path.append(functions_path0)
from fcn_simulation_setup import fcn_define_arousalSweep
sys.path.append(functions_path1) 
from fcn_compute_firing_stats import Dict2Class
from fcn_simulation_loading import fcn_set_sweepParam_string

#%% settings

sim_params_path = global_settings.path_to_src_code + 'run_simulations/'
load_path = global_settings.path_to_sim_output

figPlot = 'cluster_mainArousal'
    
if figPlot == 'cluster_mainArousal':
    simParams_fname = 'simParams_051325_clu_spontLong'
    sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'
    net_type = 'baseEIclu'
    fig_path = global_settings.path_to_manuscript_figs_final + 'model_rasters_varyArousal/cluster_mainArousal/'    
    netInd = 0
    stimInd = 0
    trialInd = 0
    arousal_levels_plot = np.array([0, 40, 100])
    xlim = [1,2]
    figID = 'Fig6F'

elif figPlot == 'cluster_altArousal':
    simParams_fname = 'simParams_050925_clu'
    sweep_param_name = 'zeroMean_sd_nu_ext_ee'
    net_type = 'baseEIclu'   
    fig_path = global_settings.path_to_manuscript_figs_final + 'model_rasters_varyArousal/cluster_altArousal/'
    netInd = 1
    stimInd = 0
    trialInd = 4
    arousal_levels_plot = np.array([0, 50, 100])
    xlim = [0,1]
    figID = 'FigS3B'

else:
    sys.exit('invalid figPlot')


nPlot_perClusterE = 8*1
nPlot_perClusterI = 2*1

figureSize = (1.4, 1.1)
fontSize = 8

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
all_arousal_levels = s_params['arousal_levels']*100
del s_params
del params


#%% START PLOTTING...


#%% loop over arousal levels

for indPlot in range(0, len(arousal_levels_plot)):
    
    # arousal index to plot
    indArousal_plot = np.argmin(np.abs(all_arousal_levels - arousal_levels_plot[indPlot]))

    # swept parameters string
    sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, indArousal_plot)     

    # simulation filename
    fname = ( '%s%s_%s_sweep_%s_network%d_IC%d_stim%d_stimType_%s_stim_rel_amp%0.3f_simulationData.mat' % \
            ( load_path, simID, net_type, sweep_param_str, netInd, trialInd, stimInd, stim_shape, stim_rel_amp) )
        
    # save name
    save_name = ( '%s_%d' % (figID, indPlot) )

    # load data
    data = loadmat(fname, simplify_cells=True)                
    s_params = Dict2Class(data['sim_params'])
    spikes = data['spikes']
    
    # get relevant parameters
    N_e = s_params.N_e
    N_i = s_params.N_i
    Tf = s_params.TF
    To = s_params.T0  
    p = s_params.p
    popsizeE = data['popSize_E']
    popsizeI = data['popSize_I']

    # E and I cells
    indsE = np.nonzero(spikes[1,:] < N_e)[0]
    indsI = np.nonzero(spikes[1,:] >= N_e)[0]
        
    #--------------------------------------------------------------------------------------------    

    # set up plot (subsampled raster)
    plt.rcParams.update({'font.size': fontSize})
    fig = plt.figure(figsize=figureSize)  
    ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 
    
    # random sample of cells
    cellsPlot_e = np.array([])
    cellsPlot_i = np.array([])
    
    for indClu in range(0, p+1):
        
        cells_in_cluster = np.arange( np.cumsum(np.append(0,popsizeE))[indClu], np.cumsum(np.append(0,popsizeE))[indClu+1])
        cellsPlot_e = np.append(cellsPlot_e, np.random.choice(cells_in_cluster, nPlot_perClusterE, replace='False'))
        
    for indClu in range(0, p+1):
        
        cells_in_cluster = np.arange( N_e + np.cumsum(np.append(0,popsizeI))[indClu], N_e + np.cumsum(np.append(0,popsizeI))[indClu+1])
        cellsPlot_i = np.append(cellsPlot_i, np.random.choice(cells_in_cluster, nPlot_perClusterI, replace='False'))
    
    for count_e, cellInd in enumerate(cellsPlot_e):
        
        spkInds = np.nonzero(spikes[1,:]==cellInd)[0]
        ax.plot(spikes[0,spkInds], np.ones(len(spkInds))*count_e, 'o', markersize=0.1, markerfacecolor='navy', markeredgecolor='navy')
    
    for count_i, cellInd in enumerate(cellsPlot_i):
        
        spkInds = np.nonzero(spikes[1,:]==cellInd)[0]
        ax.plot(spikes[0,spkInds], np.ones(len(spkInds))*count_i + count_e + 1, 'o', markersize=0.1, markerfacecolor='firebrick', markeredgecolor='firebrick')
        
    ax.set_xlim(xlim)
    ax.set_xticks(xlim)
    ax.set_yticks([])
    ax.set_xlabel('time [s]')
    ax.set_ylabel('neurons')
    plt.savefig( ('%s%s.pdf' % (fig_path, save_name)) , bbox_inches='tight', pad_inches=0, transparent=False)
    
    
    
    
