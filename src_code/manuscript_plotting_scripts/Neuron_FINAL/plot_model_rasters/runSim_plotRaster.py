
'''
this script generates different figure panels depending on settings_file.py:
    
    model = 'cluster':
        Fig3A
        
    model = 'hom'
        Fig3B
'''


#%% STANDARD IMPORTS

import settings_file
import sys
import time
import importlib
import numpy as np
import os

sys.path.append('../../../')
import global_settings

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


#%% LOAD CUSTOM FUNCTIONS

# path to simulation and network generation functions
functions_path0 = settings_file.functions_path0
functions_path1 = settings_file.functions_path1

# loading
sys.path.append(functions_path0)   
from fcn_simulation_setup import fcn_define_arousalSweep, fcn_basic_setup, fcn_set_popSizes, fcn_set_initialVoltage, fcn_updateParams_givenArousal, fcn_setup_one_stimulus

sys.path.append(functions_path1)         
from fcn_make_network_cluster import fcn_make_network_cluster
from fcn_simulation_EIextInput import fcn_simulate_expSyn
from fcn_stimulation import get_stimulated_clusters


#%% SETTINGS

# path to simulation parameters
sim_params_path = settings_file.sim_params_path

# sim params name
sim_params_name = settings_file.sim_params_name

# path to figures
fig_path = settings_file.fig_path

# figure ID
figID = settings_file.figID

# simulation settings
arousalLevel = settings_file.arousalLevel
TF = settings_file.TF
stimOn = settings_file.stimOn
nPlot_perClusterE = settings_file.nPlot_perClusterE
nPlot_perClusterI = settings_file.nPlot_perClusterI

# seeds
externalInput_seed = settings_file.externalInput_seed
stimClusters_seed = settings_file.stimClusters_seed
stimNeurons_seed = settings_file.stimNeurons_seed
networkSeed = settings_file.networkSeed


#%% MAKE OUTPUT DIRECTORY

if os.path.isdir(fig_path) == False:
    os.makedirs(fig_path)

#%% LOAD CONFIG FILE

# IMPORT CONFIG FILE FOR SETTING PARAMETERS
sys.path.append(sim_params_path) 
params = importlib.import_module(sim_params_name) 
s_params = params.sim_params
del params

#%% FUNCTION THAT SETS AROUSAL PARAMETERS FOR A GIVEN AROUSAL INDEX

def fcn_set_arousalParameters(s_params, arousal_indx):

    nParams_sweep = s_params['nParams_sweep']
    swept_param_name_dict = s_params['swept_param_name_dict']
    swept_params_dict = s_params['swept_params_dict']
    
    for i in range(1, nParams_sweep+1):
        
        key_name = 'param_vals%d' % i
        param_name =  swept_param_name_dict[key_name]
        s_params[param_name] = swept_params_dict[key_name][arousal_indx]



#%% SIMULATION SETUP

# simulation time
s_params['TF'] = TF
s_params['stim_rel_amp']*stimOn

# arousal indx
arousal_indx = np.argmin(np.abs(s_params['arousal_levels'] - arousalLevel))

# saving of voltage
s_params['save_voltage'] = True

# get arousal parameters for sweeping over
s_params = fcn_define_arousalSweep(s_params)

# set arousal parameters
fcn_set_arousalParameters(s_params, arousal_indx)

# basic setup
s_params = fcn_basic_setup(s_params)

# update sim params given arousal parameters
s_params = fcn_updateParams_givenArousal(s_params, externalInput_seed)

# make network
W, popsizeE, popsizeI = fcn_make_network_cluster(s_params, networkSeed)  

# set popsizes
s_params = fcn_set_popSizes(s_params, popsizeE, popsizeI)

# set selective clusters (random seed)    
selectiveClusters = get_stimulated_clusters(s_params, stimClusters_seed)

# setup stimulus
s_params = fcn_setup_one_stimulus(s_params, selectiveClusters, 0, stimNeurons_seed)

# set initial voltage
s_params = fcn_set_initialVoltage(s_params)
    
# cluster boundaries
clus=np.cumsum(popsizeE)


#%% RUN SIMULATION    


# start timing
t0 = time.time()

# save voltage
timePts, spikes, v, I_exc, I_inh, I_o = fcn_simulate_expSyn(s_params, W)

# end timing
tf = time.time()
print('sim time = %0.3f seconds' %(tf-t0))

    
#%% PLOTTING

if stimOn:
    stimStr = 'True'
else:
    stimStr = 'False'
    

#%% get relevant parameters
N_e = s_params['N_e']
N_i = s_params['N_i']
p = s_params['p']


#%% raster baseline, subsampled

plt.rcParams.update({'font.size': 8})
fig = plt.figure(figsize=(2.2,1.3))  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 

# random sample of cells
cellsPlot_e = np.array([])
cellsPlot_i = np.array([])

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
    ax.plot(spikes[0,spkInds], np.ones(len(spkInds))*count_e, 'o', markersize=0.1, color='navy')

for count_i, cellInd in enumerate(cellsPlot_i):
    
    spkInds = np.nonzero(spikes[1,:]==cellInd)[0]
    ax.plot(spikes[0,spkInds], np.ones(len(spkInds))*count_i + count_e + 1, 'o', markersize=0.1, color='firebrick')
    
ax.set_xlim([0, TF])
ax.set_xticks([0, TF/2, TF])
ax.set_yticks([])
ax.set_xlabel('time [s]')
ax.set_ylabel('neurons')
plt.savefig( ('%s%s.pdf' % (fig_path, figID)) , bbox_inches='tight', pad_inches=0, transparent=True)
