#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLOT 
"""

#%% BASIC IMPORTS

import numpy as np
from scipy.io import loadmat
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
matplotlib.use('agg')


#%% PATH TO DATA
loadMFT_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/test_stim_expSyn/MFT_sweep_sd_nu_ext_e_pert/')
fig_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/Figures/test_stim_expSyn/MFT_sweepArousal_fixedJeePlus/')

#%% PARAMETERS FOR LOADING DATA

# simulation ID
mft_ID = '102320221109' #[cluster, sd pert, 5 stim, poisson external inputs, Jeeplus = 15.75]

# network name
net_type = 'baseEIclu'
#net_type = 'baseHOM'

# stim shape
stim_shape = 'diff2exp'

# stim type
stim_type = ''

# relative stimulation amplitude
stim_rel_amp = 0.05

# value of Jplus
JplusEE = 16.725

# sweep param name
mft_sweep_param_name = 'sd_nu_ext_e_pert'


fname_end = ''


fig_filename = ( '%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f_reducedMFT_sweep_%s_JplusEE%0.3f_%s_' % \
               ( mft_ID, net_type, mft_sweep_param_name, stim_shape, stim_rel_amp, mft_sweep_param_name, JplusEE, fname_end ) )
        

fig_path = fig_path + mft_sweep_param_name + '/'
    
#%% plotting parameters

Ecolors = ['darkmagenta','thistle','mediumorchid']
Icolors = ['darkcyan', 'lightblue', 'mediumturquoise']


#%% load one example in order to get dimensions for arrays

    
# filename
mft_filename = ( '%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f_reducedMFT_sweep_%s_JplusEE%0.3f_%s.mat' % \
               ( mft_ID, net_type, mft_sweep_param_name, stim_shape, stim_rel_amp, mft_sweep_param_name, JplusEE, fname_end ) )

    
MFT_data = loadmat(loadMFT_path + mft_filename, simplify_cells=True)

mft_params = MFT_data['mft_params']
backSweep_results = MFT_data['backSweep_results']
forSweep_results = MFT_data['forSweep_results']

n_activeClusters_sweep = np.array([mft_params['n_active_clusters_sweep']])
param_back = backSweep_results['sweepParam_back']
param_for = forSweep_results['sweepParam_for']

print(np.shape(backSweep_results['nu_bar_e_back']))

#%% plot rate of active E clusters

cmap = cm.get_cmap('viridis', len(n_activeClusters_sweep)+1)
cmap = cmap(range(len(n_activeClusters_sweep)+1))

plt.figure()
plt.rcParams['font.size'] = '16'

# set # of active clusters
for ind, nActive in enumerate(n_activeClusters_sweep):

    ind_active = np.nonzero( n_activeClusters_sweep == nActive )[0][0]
        
    x = backSweep_results['sweepParam_back'].copy()
    if np.size(n_activeClusters_sweep) ==1:
        y = backSweep_results['nu_bar_e_back'][0,:].copy()

    else:
        y = backSweep_results['nu_bar_e_back'][0,:,ind_active].copy()

    
    plt.plot(x, y, '-o', color=cmap[ind,:], markersize=2, label=('n_active=%d' % nActive) )


    x = forSweep_results['sweepParam_for'].copy()
    if np.size(n_activeClusters_sweep) ==1:
        y = forSweep_results['nu_bar_e_for'][0,:].copy()

    else:
        y = forSweep_results['nu_bar_e_for'][0,:,ind_active].copy()

    plt.plot(x, y, '--o', color=cmap[ind,:], markersize=2, label=('n_active=%d' % nActive) )
        
    
    plt.xlabel(mft_sweep_param_name)
    plt.ylabel('active cluster rate')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig( ( fig_path + fig_filename + '_activeErate.pdf') , transparent=True)



#%% plot rate of active E clusters

cmap = cm.get_cmap('viridis', len(n_activeClusters_sweep)+1)
cmap = cmap(range(len(n_activeClusters_sweep)+1))

plt.figure()
plt.rcParams['font.size'] = '16'

# set # of active clusters
for ind, nActive in enumerate(n_activeClusters_sweep):

    ind_active = np.nonzero( n_activeClusters_sweep == nActive )[0][0]
        
    x = backSweep_results['sweepParam_back'].copy()
    if np.size(n_activeClusters_sweep) ==1:
        y = backSweep_results['nu_bar_e_back'][0,:].copy()
        y1 = backSweep_results['nu_bar_e_back'][1,:].copy()
        y2 = backSweep_results['nu_bar_e_back'][2,:].copy()

    else:
        y = backSweep_results['nu_bar_e_back'][0,:,ind_active].copy()
        y1 = backSweep_results['nu_bar_e_back'][1,:,ind_active].copy()
        y2 = backSweep_results['nu_bar_e_back'][2,:,ind_active].copy()
    
    plt.plot(x, y, '-o', color='k', markersize=2, label=('n_active=%d' % nActive) )
    plt.plot(x, y1, '-o', color='gray', markersize=2, label=('n_active=%d' % nActive) )
    plt.plot(x, y2, '-o', color='b', markersize=2, label=('n_active=%d' % nActive) )


    x = forSweep_results['sweepParam_for'].copy()
    if np.size(n_activeClusters_sweep) ==1:
        y = forSweep_results['nu_bar_e_for'][0,:].copy()
        y1 = forSweep_results['nu_bar_e_for'][1,:].copy()
        y2 = forSweep_results['nu_bar_e_for'][2,:].copy()

    else:
        y = forSweep_results['nu_bar_e_for'][0,:,ind_active].copy()
        y1 = forSweep_results['nu_bar_e_for'][1,:,ind_active].copy()
        y2 = forSweep_results['nu_bar_e_for'][2,:,ind_active].copy()
    
    plt.plot(x, y, '-', color='k', linewidth=3 ,label=('active') )
    plt.plot(x, y1, '-', color='gray', linewidth=2, label=('inactive') )
    plt.plot(x, y2, '-', color='b', linewidth=1, label=('background') )
    
    
    plt.xlabel(mft_sweep_param_name)
    plt.ylabel('cluster rate')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig( ( fig_path + fig_filename + '_Erates.pdf') , transparent=True)
