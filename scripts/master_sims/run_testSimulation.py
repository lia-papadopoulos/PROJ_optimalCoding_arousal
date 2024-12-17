


#%% STANDARD IMPORTS

import paths_file
import sys
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# unpack paths
sim_params_path = paths_file.sim_params_path
functions_path1 = paths_file.functions_path1

# import sim params  
sys.path.append(sim_params_path)     
from simParams import sim_params

# import helper functions
sys.path.append(functions_path1)     
from fcn_make_network_cluster import fcn_make_network_cluster
from fcn_simulation_EIextInput import fcn_simulate_expSyn
from fcn_stimulation import get_stimulated_clusters


#%% PLOTTING

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Helvetica'
mpl.rcParams['font.size'] = 14


#%% USER INPUTS

externalInput_seed = np.random.choice(10000)
stimClusters_seed = np.random.choice(10000)
stimNeurons_seed = np.random.choice(1000)
networkSeed = np.random.choice(10000)


#%%   MAIN FUNCTION

# SIM PARAMS
s_params = sim_params()

# saving of voltage
s_params.save_voltage = True

# synaptic weights
s_params.set_Je_reduction()
s_params.update_JplusAB()
    
# set dependent variables
s_params.set_dependent_vars()

# make network
W, popsizeE, popsizeI = fcn_make_network_cluster(s_params, networkSeed)  

# set popsizes
s_params.set_popSizes(popsizeE, popsizeI)   

# set selective clusters (random seed)
selectiveClusters = get_stimulated_clusters(s_params,stimClusters_seed)

# set initial voltage
s_params.fcn_set_initialVoltage()

# set external inputs (random seed)
s_params.set_external_inputs_ei(externalInput_seed) 
  
# set selective clusters for one stimulus
s_params.selectiveClusters = selectiveClusters[0].copy()
    
# determine which neurons are stimulated (random seed)
s_params.get_stimulated_neurons(stimNeurons_seed, popsizeE, popsizeI)
    
# determine maximum stimulus strength
s_params.set_max_stim_rate()
    
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



#%% raster


plt.figure(figsize=(5.0,4))

indsE = np.nonzero(spikes[1,:] < s_params.N_e)[0]
indsI = np.nonzero(spikes[1,:] >= s_params.N_e)[0]

plt.plot(spikes[0,indsE],spikes[1,indsE], '.', markersize=0.5, color='navy')
plt.plot(spikes[0,indsI],spikes[1,indsI], '.', markersize=0.5, color='firebrick')
plt.yticks([])
plt.xlabel('time [s]')
plt.ylabel('neuron ID')
plt.tight_layout()


#%% raster with stimulated neurons highlighted

if s_params.stim_type != 'noStim':

    plt.figure(figsize=(5.0,4))

    # stim onset
    stimOnset = s_params.stim_onset
    
    # stimulated E and I cells
    stimECells = np.nonzero(s_params.stim_Ecells)[0]
    stimICells = np.nonzero(s_params.stim_Icells)[0] + s_params.N_e
    
    # non stimulated E and I cells
    nonStimECells = np.setdiff1d(np.arange(0,s_params.N_e), stimECells)
    nonStimICells = np.setdiff1d(np.arange(0,s_params.N_i), stimICells) + s_params.N_e
    
    
    for cell in nonStimECells:
  
        spkInds = np.nonzero(spikes[1,:] == cell)[0]
        plt.plot(spikes[0,spkInds],spikes[1,spkInds], '.', markersize=0.5, color='navy')
    
    for cell in nonStimICells:
  
        spkInds = np.nonzero(spikes[1,:] == cell)[0]
        plt.plot(spikes[0,spkInds],spikes[1,spkInds], '.', markersize=0.5, color='firebrick')         
    
    for cell in stimECells:
  
        spkInds = np.nonzero(spikes[1,:] == cell)[0]
        plt.plot(spikes[0,spkInds],spikes[1,spkInds], '.', markersize=0.5, color='k')
    
    for cell in stimICells:
  
        spkInds = np.nonzero(spikes[1,:] == cell)[0]
        plt.plot(spikes[0,spkInds],spikes[1,spkInds], '.', markersize=0.5, color='k')        
    
    
    plt.plot([stimOnset, stimOnset], [0, s_params.N_e], color = 'k')
    plt.xlabel('time [s]')
    plt.ylabel('neuron ID')
    plt.tight_layout() 


#%% network

plt.figure(figsize=(5.0,4))
plt.imshow(W,cmap='bwr_r')
plt.xlabel('neurons')
plt.ylabel('neurons')
plt.xticks([],[])
plt.yticks([],[])
plt.clim([-np.max(np.abs(W)),np.max(np.abs(W))])
plt.tight_layout()


#%% show plots

plt.show()