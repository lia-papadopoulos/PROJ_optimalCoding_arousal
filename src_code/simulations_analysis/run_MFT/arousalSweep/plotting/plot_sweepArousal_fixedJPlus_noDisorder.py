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
loadMFT_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/test_stim_expSyn/MFT_arousalSweep/')
fig_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/Figures/test_stim_expSyn/MFT_sweepArousal_fixedJeePlus/')

#%% PARAMETERS FOR LOADING DATA

# simulation ID
mft_ID = '121900002024_a' #[cluster, sd pert, 5 stim, poisson external inputs, Jeeplus = 15.75]

# network name
net_type = 'baseEIclu'

# stim shape
stim_shape = 'diff2exp'

# relative stimulation amplitude
stim_rel_amp = 0.05

# value of Jplus
JplusEE = 16.775

# sweep param name
mft_sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'

# plot param name
plot_param_name = 'arousal [%]'


fig_filename = ( '%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f_JplusEE%0.3f_MFT_noDisorder' % \
               ( mft_ID, net_type, mft_sweep_param_name, stim_shape, stim_rel_amp, JplusEE ) )
        

# filename
mft_filename = ( '%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f_JplusEE%0.3f_MFT_noDisorder.mat' % \
               ( mft_ID, net_type, mft_sweep_param_name, stim_shape, stim_rel_amp, JplusEE ) )
    

fig_path = fig_path + mft_sweep_param_name + '/'

    
#%% plotting parameters

Ecolors = ['darkmagenta','thistle','mediumorchid']
Icolors = ['darkcyan', 'lightblue', 'mediumturquoise']


#%% load one example in order to get dimensions for arrays

    
MFT_data = loadmat(loadMFT_path + mft_filename, simplify_cells=True)
mft_params = MFT_data['mft_params']

backSweep_results = MFT_data['backSweep_results']
forSweep_results = MFT_data['forSweep_results']

n_activeClusters_sweep = mft_params['n_active_clusters_sweep']
rate_backSweep = backSweep_results['nu_e_backSweep']
rate_forSweep = forSweep_results['nu_e_forSweep']
n_sweepParam_vals = np.size(rate_backSweep,1)

arousal_level = np.arange(0, n_sweepParam_vals)/(n_sweepParam_vals-1)

#%% plot rate of E clusters

cmap = cm.get_cmap('viridis', len(n_activeClusters_sweep)+1)
cmap = cmap(range(len(n_activeClusters_sweep)+1))

plt.figure()
plt.rcParams['font.size'] = '16'

# set # of active clusters
for ind, nActive in enumerate(n_activeClusters_sweep):

    ind_active = np.nonzero( n_activeClusters_sweep == nActive )[0][0]

    if np.size(n_activeClusters_sweep) ==1:
        y = rate_backSweep[0,:].copy()

    else:
        y = rate_backSweep[0,:,ind_active].copy()

    plt.plot(arousal_level, y, '-o', color=cmap[ind,:], markersize=2, label=('n_active=%d' % nActive) )


    if np.size(n_activeClusters_sweep) ==1:
        y = rate_forSweep[0,:].copy()

    else:
        y = rate_forSweep[0,:,ind_active].copy()

    plt.plot(arousal_level, y, '--o', color=cmap[ind,:], markersize=2 )
        
    
    plt.xlabel(plot_param_name)
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
        
    if np.size(n_activeClusters_sweep) == 1:
        y = rate_backSweep[0,:].copy()
        y1 = rate_backSweep[-2,:].copy()
        y2 = rate_backSweep[-1,:].copy()
    else:
        y = rate_backSweep[0,:,ind_active].copy()
        y1 = rate_backSweep[-2,:,ind_active].copy()
        y2 = rate_backSweep[-1,:,ind_active].copy()
    
    plt.plot(arousal_level, y, '-o', color=cmap[ind,:], markersize=2, label=('n_active=%d' % nActive) )
    plt.plot(arousal_level, y1, '-x', color=cmap[ind,:], markersize=2)
    plt.plot(arousal_level, y2, '-', color=cmap[ind,:], markersize=2)


    if np.size(n_activeClusters_sweep) == 1:
        y = rate_forSweep[0,:].copy()
        y1 = rate_forSweep[-2,:].copy()
        y2 = rate_forSweep[-1,:].copy()

    else:
        y = rate_forSweep[0,:,ind_active].copy()
        y1 = rate_forSweep[-2,:,ind_active].copy()
        y2 = rate_forSweep[-1,:,ind_active].copy()
    
    plt.plot(arousal_level, y, '--o', color=cmap[ind,:], linewidth=3 )
    plt.plot(arousal_level, y1, '--x', color=cmap[ind,:], linewidth=2 )
    plt.plot(arousal_level, y2, '--', color=cmap[ind,:], linewidth=1 )
    
    
    plt.xlabel(plot_param_name)
    plt.ylabel('cluster rate')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig( ( fig_path + fig_filename + '_Erates.pdf') , transparent=True)
    
    
#%% plot rate of active E clusters


cmap = cm.get_cmap('viridis', len(n_activeClusters_sweep)+1)
cmap = cmap(range(len(n_activeClusters_sweep)+1))

plt.figure()
plt.rcParams['font.size'] = '16'

nActive_plot = 3

# set # of active clusters
for ind, nActive in enumerate(n_activeClusters_sweep):

    if nActive == nActive_plot:
        
        ind_active = np.nonzero( n_activeClusters_sweep == nActive )[0][0]
            
        if np.size(n_activeClusters_sweep) == 1:
            y = rate_backSweep[0,:].copy()
            y1 = rate_backSweep[-2,:].copy()
            y2 = rate_backSweep[-1,:].copy()
        else:
            y = rate_backSweep[0,:,ind_active].copy()
            y1 = rate_backSweep[-2,:,ind_active].copy()
            y2 = rate_backSweep[-1,:,ind_active].copy()
        
        plt.plot(arousal_level, y, '-o', color=cmap[ind,:], markersize=2, label=('n_active=%d' % nActive) )
        plt.plot(arousal_level, y1, '-x', color=cmap[ind,:], markersize=2)
        plt.plot(arousal_level, y2, '-', color=cmap[ind,:], markersize=2)
    
    
        if np.size(n_activeClusters_sweep) == 1:
            y = rate_forSweep[0,:].copy()
            y1 = rate_forSweep[-2,:].copy()
            y2 = rate_forSweep[-1,:].copy()
    
        else:
            y = rate_forSweep[0,:,ind_active].copy()
            y1 = rate_forSweep[-2,:,ind_active].copy()
            y2 = rate_forSweep[-1,:,ind_active].copy()
        
        plt.plot(arousal_level, y, '--o', color=cmap[ind,:], linewidth=3 )
        plt.plot(arousal_level, y1, '--x', color=cmap[ind,:], linewidth=2 )
        plt.plot(arousal_level, y2, '--', color=cmap[ind,:], linewidth=1 )
        
        
        plt.xlabel(plot_param_name)
        plt.ylabel('cluster rate')
        plt.legend(fontsize=8)
        plt.tight_layout()
    plt.savefig( ( fig_path + fig_filename + '_Erates_nActive%d.pdf' % nActive_plot) , transparent=True)
