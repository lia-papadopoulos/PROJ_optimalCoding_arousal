#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLOT clusterRates_numActiveClusters_MFT_Sims
"""

#%% BASIC IMPORTS

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


#%% PATH TO DATA

loadMFT_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/test_stim_expSyn/MFT_sweep_JeePlus_baseline/')
data_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/test_stim_expSyn/clusterRates_numActiveClusters/')
loadSIM_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/test_stim_expSyn/')
fig_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/Figures/test_stim_expSyn/clusterRates_MFT_sims_baselineArousal/')

#%% PARAMETERS FOR LOADING DATA

# simulation ID
simID = '113020232105' #[cluster, sd pert, 5 stim, poisson external inputs, Jeeplus = 15.75]
#simID = '102320221109'
mftID = '102320221109'

# network name
net_type = 'baseEIclu'
#net_type = 'baseHOM'

# stim shape
stim_shape = 'diff2exp'

# stim type
stim_type = ''

# relative stimulation amplitude
stim_rel_amp = 0.05

# sweep param name
sim_sweep_param_name = 'same_eachClustersd_nu_ext_e_pert'
#sim_sweep_param_name = 'sd_nu_ext_e_pert'
mft_sweep_param_name = 'sd_nu_ext_e_pert'

# sweep param value
sweep_param_val = 0.

# analysis parameters
rateThresh = 15
windowStd = 25e-3


# simulation filename
sim_filename = ( '%s_%s_sweep_%s%0.3f_network%d_IC%d_stim%d_stimType_%s_stim_rel_amp%0.3f_simulationData.mat' % \
               (simID, net_type, sim_sweep_param_name, sweep_param_val, 0, 0, 0, stim_shape, stim_rel_amp ) )

    
# simulation filename
data_filename = ( '%s_%s_sweep_%s%0.3f_stimType_%s_stim_rel_amp%0.3f__clusterRates_numActiveClusters.mat' % \
                 (simID, net_type, sim_sweep_param_name, sweep_param_val, stim_shape, stim_rel_amp ) )
    
# mft filename
mft_filename = ( '%s_%s_sweep_%s%0.3f_stimType_%s_stim_rel_amp%0.3f_MFT_sweepJeePlus_baseline.mat' % \
               ( mftID, net_type, mft_sweep_param_name, sweep_param_val, stim_shape, stim_rel_amp ) )
    

# fig filename
fig_filename = ( 'mft%s_sim%s_%s_sweep_%s%0.3f_stimType_%s_stim_rel_amp%0.3f_rateThresh%0.1fHz_windowStd%0.3fs_' % \
               ( mftID, simID, net_type, mft_sweep_param_name, sweep_param_val, stim_shape, stim_rel_amp, rateThresh, windowStd ) )


fig_path = fig_path + mft_sweep_param_name + '/'
    
#%% LOAD THE SIMULATION AND MFT DATA

MFT_data = loadmat(loadMFT_path + mft_filename, simplify_cells=True)
SIM_data = loadmat(loadSIM_path + sim_filename, simplify_cells=True)

ANALYSIS_data = loadmat(data_path + data_filename, simplify_cells=True)
analysis_params = ANALYSIS_data['parameters']
nNets = analysis_params['nNets']
rate_thresh_array = analysis_params['rate_thresh']

#%% WINDOW LENGTH AND RATE THRESHOLD TO PLOT
indThresh_plot = np.nonzero(rate_thresh_array==rateThresh)[0][0]


#%% UNPACK SIM DATA

sim_params = SIM_data['sim_params']


#%% UNPACK MFT DATA

mft_params = MFT_data['mft_params']
JeePlus_backSweep_results = MFT_data['JeePlus_backSweep_results']
n_activeClusters_sweep = mft_params['n_active_clusters_sweep']

JplusEE_back = JeePlus_backSweep_results['JplusEE_back']
nu_e_backSweep = JeePlus_backSweep_results['nu_e_backSweep']
nu_i_backSweep = JeePlus_backSweep_results['nu_i_backSweep']
n_activeClustersE_back = JeePlus_backSweep_results['n_activeClustersE_back']
n_activeClustersI_back = JeePlus_backSweep_results['n_activeClustersI_back']


#%% UNPACK SIM ANALYSIS DATA

netAvg_avgRate_background_XActiveClusters_E = ANALYSIS_data['netAvg_avgRate_background_XActiveClusters_E'][:, indThresh_plot].copy()
netAvg_avgRate_active_XActiveClusters_E = ANALYSIS_data['netAvg_avgRate_active_XActiveClusters_E'][:, indThresh_plot].copy()
netAvg_avgRate_inactive_XActiveClusters_E = ANALYSIS_data['netAvg_avgRate_inactive_XActiveClusters_E'][:, indThresh_plot].copy()
    

#%% MATCH DATA BETWEEN SIMULATIONS AND MFT

# JeePlus from MFT that matches JeePlus from simulations
JeePlus_sim = sim_params['JplusEE']
indJeePlus_mft = np.argmin(np.abs(JplusEE_back - JeePlus_sim))

# possibilities for number of clusters active from simulations
n_activeClusters_sim = (np.nonzero(~np.isnan(netAvg_avgRate_active_XActiveClusters_E))[0]).astype(int)
ind_n_activeClusters_mft = n_activeClusters_sim - 1

# active, inactive, and background rates simulations
activeRate_XActiveClusters_E_sim = netAvg_avgRate_active_XActiveClusters_E[n_activeClusters_sim]
inactiveRate_XActiveClusters_E_sim = netAvg_avgRate_inactive_XActiveClusters_E[n_activeClusters_sim]
backgroundRate_XActiveClusters_E_sim = netAvg_avgRate_background_XActiveClusters_E[n_activeClusters_sim]

# active, inactive, and background rates mft
activeRate_XActiveClusters_E_mft = nu_e_backSweep[0, indJeePlus_mft, ind_n_activeClusters_mft]
inactiveRate_XActiveClusters_E_mft = nu_e_backSweep[n_activeClusters_sim, indJeePlus_mft, ind_n_activeClusters_mft]
backgroundRate_XActiveClusters_E_mft = nu_e_backSweep[-1, indJeePlus_mft, ind_n_activeClusters_mft]


#%% PLOTTING

Ecolors = ['darkmagenta','thistle','mediumorchid']
Icolors = ['darkcyan', 'lightblue', 'mediumturquoise']


#%% plot rate of active E clusters vs # clusters active


plt.figure()

plt.rcParams['font.size'] = '16'


            
x = n_activeClusters_sim
y = activeRate_XActiveClusters_E_sim
plt.plot(x, y, 'o', color=Ecolors[0], linewidth=2, label=('sim') )

x = n_activeClusters_sim
y = activeRate_XActiveClusters_E_mft
plt.plot(x, y, '*', color=Ecolors[0], linewidth=2, label=('mft') )        


plt.xlabel('# active clusters')
plt.ylabel('active E cluster rate')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(fig_path + fig_filename + 'activeErate_vs_numClustersActive_mft_sim.pdf', transparent=True)


#%% plot rate of inactive E clusters vs # clusters active


plt.figure()

plt.rcParams['font.size'] = '16'


            
x = n_activeClusters_sim
y = inactiveRate_XActiveClusters_E_sim
plt.plot(x, y, 'o', color=Ecolors[1], linewidth=2, label=('sim') )

x = n_activeClusters_sim
y = inactiveRate_XActiveClusters_E_mft
plt.plot(x, y, '*', color=Ecolors[1], linewidth=2, label=('mft') )        


plt.xlabel('# active clusters')
plt.ylabel('inactive E cluster rate')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(fig_path + fig_filename + 'inactiveErate_vs_numClustersActive_mft_sim.pdf', transparent=True)


#%% plot rate of background E clusters vs # clusters active


plt.figure()

plt.rcParams['font.size'] = '16'

            
x = n_activeClusters_sim
y = backgroundRate_XActiveClusters_E_sim
plt.plot(x, y, 'o', color=Ecolors[2], linewidth=2, label=('sim') )

x = n_activeClusters_sim
y = backgroundRate_XActiveClusters_E_mft
plt.plot(x, y, '*', color=Ecolors[2], linewidth=2, label=('mft') )        


plt.xlabel('# active clusters')
plt.ylabel('background E cluster rate')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(fig_path + fig_filename + 'backgroundErate_vs_numClustersActive_mft_sim.pdf', transparent=True)
