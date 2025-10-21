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
loadMFT_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/MFT_2Ecluster/MFT_sweep_sd_nu_ext_e_pert_2Ecluster/')
fig_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/Figures/MFT_2Ecluster/MFT_sweep_sd_nu_ext_e_pert_2Ecluster/')

#%% PARAMETERS FOR LOADING DATA


# value of Jplus
JplusEE = 10.00

# sweep param name
mft_sweep_param_name = 'sd_nu_ext_e_pert'

end_fname = 'smallBackground'


mft_filename = (( 'MFT_sweep_%s_JplusEE%0.3f_2Ecluster_%s.mat' ) % (mft_sweep_param_name, JplusEE, end_fname) )


fig_filename = ((  'MFT_sweep_%s_JplusEE%0.3f_2Ecluster_%s' ) % ( mft_sweep_param_name, JplusEE, end_fname) )
        


#%% plotting parameters

Ecolors = ['darkmagenta','thistle','mediumorchid']
Icolors = ['darkcyan', 'lightblue', 'mediumturquoise']


#%% load one example in order to get dimensions for arrays


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

    
    plt.plot(x, y, '-o', color=cmap[ind,:], markersize=2, label=('n_activet=%d' % nActive) )


    x = forSweep_results['sweepParam_for'].copy()
    if np.size(n_activeClusters_sweep) ==1:
        y = forSweep_results['nu_bar_e_for'][0,:].copy()

    else:
        y = forSweep_results['nu_bar_e_for'][0,:,ind_active].copy()

    plt.plot(x, y, '--o', color=cmap[ind,:], markersize=2, label=('n_activet=%d' % nActive) )
        
    
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
    
    plt.plot(x, y, '-o', color='k', markersize=2, label=('active') )
    plt.plot(x, y1, '-o', color='gray', markersize=2, label=('inactive') )
    plt.plot(x, y2, '-o', color='b', markersize=2, label=('background') )


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
