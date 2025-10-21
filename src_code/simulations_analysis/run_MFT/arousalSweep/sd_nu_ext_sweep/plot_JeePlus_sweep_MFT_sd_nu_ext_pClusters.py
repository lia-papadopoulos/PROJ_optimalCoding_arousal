#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLOT JeePlus_sweep_MFT_sd_nu_ext_pEIclusters
"""

#%% BASIC IMPORTS

import numpy as np
from scipy.io import loadmat
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
matplotlib.use('agg')


#%% PATH TO DATA
loadMFT_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/test_stim_expSyn/MFT_sweep_JeePlus_sweep_sd_nu_ext_e_pert/')
fig_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/Figures/test_stim_expSyn/MFT_sweepJeePlus_varyArousal/')

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

# sweep param name
mft_sweep_param_name = 'sd_nu_ext_e_pert'

# sweep param value
sweep_param_array = np.arange(0,0.41,0.01)


fig_filename = ( '%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f_reducedMFT_sweepJeePlus_' % \
               ( mft_ID, net_type, mft_sweep_param_name, stim_shape, stim_rel_amp ) )
        
fig_path = fig_path + mft_sweep_param_name + '/'

#%% plotting parameters

Ecolors = ['darkmagenta','thistle','mediumorchid']
Icolors = ['darkcyan', 'lightblue', 'mediumturquoise']


#%% load one example in order to get dimensions for arrays

    
# filename
mft_filename = ( '%s_%s_sweep_%s%0.3f_stimType_%s_stim_rel_amp%0.3f_reducedMFT_sweepJeePlus.mat' % \
               ( mft_ID, net_type, mft_sweep_param_name, 0, stim_shape, stim_rel_amp ) )

    
MFT_data = loadmat(loadMFT_path + mft_filename, simplify_cells=True)

mft_params = MFT_data['mft_params']
JeePlus_backSweep_results = MFT_data['JeePlus_backSweep_results']
JeePlus_forSweep_results = MFT_data['JeePlus_forSweep_results']

n_activeClusters_sweep = mft_params['n_active_clusters_sweep']
JplusEE_back = JeePlus_backSweep_results['JplusEE_back']
JplusEE_for = JeePlus_forSweep_results['JplusEE_for']


#%% initialize arrays

nu_e_backSweep = np.zeros((3, np.size(JplusEE_back), np.size(n_activeClusters_sweep), np.size(sweep_param_array)))
nu_i_backSweep = np.zeros((3, np.size(JplusEE_back), np.size(n_activeClusters_sweep), np.size(sweep_param_array)))
nu_e_forSweep = np.zeros((3, np.size(JplusEE_for), np.size(n_activeClusters_sweep), np.size(sweep_param_array)))
nu_i_forSweep = np.zeros((3, np.size(JplusEE_for), np.size(n_activeClusters_sweep), np.size(sweep_param_array)))


#%% loop over swept parameter and get data for plotting

for ind_sweep_param in range(0, np.size(sweep_param_array)):
    
    
    sweep_param_val = sweep_param_array[ind_sweep_param]
    

    ### FILENAMES

        
    # filename
    mft_filename = ( '%s_%s_sweep_%s%0.3f_stimType_%s_stim_rel_amp%0.3f_reducedMFT_sweepJeePlus.mat' % \
                   ( mft_ID, net_type, mft_sweep_param_name, sweep_param_val, stim_shape, stim_rel_amp ) )
        
    
    ### LOAD THE MFT DATA
    MFT_data = loadmat(loadMFT_path + mft_filename, simplify_cells=True)
    
    
    ### UNPACK MFT DATA
    
    mft_params = MFT_data['mft_params']
    JeePlus_backSweep_results = MFT_data['JeePlus_backSweep_results']
    JeePlus_forSweep_results = MFT_data['JeePlus_forSweep_results']
    
    
    if sweep_param_val == 0:
        
        nu_e_backSweep[:,:,:,ind_sweep_param] = JeePlus_backSweep_results['nu_e_backSweep']
        nu_i_backSweep[:,:,:,ind_sweep_param] = JeePlus_backSweep_results['nu_i_backSweep']
    
        nu_e_forSweep[:,:,:,ind_sweep_param] = JeePlus_forSweep_results['nu_e_forSweep']
        nu_i_forSweep[:,:,:,ind_sweep_param] = JeePlus_forSweep_results['nu_i_forSweep']
        
    else:
        
        nu_e_backSweep[:,:,:,ind_sweep_param] = JeePlus_backSweep_results['nu_bar_e_back']
        nu_i_backSweep[:,:,:,ind_sweep_param] = JeePlus_backSweep_results['nu_bar_i_back']
    
        nu_e_forSweep[:,:,:,ind_sweep_param] = JeePlus_forSweep_results['nu_bar_e_for']
        nu_i_forSweep[:,:,:,ind_sweep_param] = JeePlus_forSweep_results['nu_bar_i_for']        



#%% plot rate of active E cluster vs Jplus
### fix nClusters_active
### plot different curves for different values of perturbation

# set # of active clusters
for nActive in n_activeClusters_sweep:

    plt.figure()
    plt.rcParams['font.size'] = '16'
    
    cmap = cm.get_cmap('viridis', len(sweep_param_array)+1)
    cmap = cmap(range(len(sweep_param_array)+1))
    
    for ind_param in range(0,len(sweep_param_array)):
        
        sweep_param_val = sweep_param_array[ind_param]
        ind_active = np.nonzero( n_activeClusters_sweep == nActive )[0][0]
        
        x = JplusEE_back
        y = nu_e_backSweep[0,:,ind_active, ind_param]
        plt.plot(x, y, 'o', color=cmap[ind_param,:], markersize=2, label=('sd_pert=%0.1f' % sweep_param_val) )
        
    
    plt.xlabel('Jee+')
    plt.ylabel('active cluster rate')
    plt.title('# active clusters = %d' % nActive)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig( ( fig_path + fig_filename + 'activeErate_vs_JeePlusBack_nActiveClusters%d.pdf' % (nActive) ) , transparent=True)



#%% plot rate of active E cluster vs Jplus
### fix nClusters_active
### plot different curves for different values of perturbation

# set # of active clusters
for nActive in n_activeClusters_sweep:

    plt.figure()
    plt.rcParams['font.size'] = '16'
    
    cmap = cm.get_cmap('viridis', len(sweep_param_array)+1)
    cmap = cmap(range(len(sweep_param_array)+1))
    
    for ind_param in range(0,len(sweep_param_array)):
        
        sweep_param_val = sweep_param_array[ind_param]
        ind_active = np.nonzero( n_activeClusters_sweep == nActive )[0][0]
    
        x = JplusEE_for
        y = nu_e_forSweep[0,:,ind_active, ind_param]
        plt.plot(x, y, 'o', color=cmap[ind_param,:], markersize=2, label=('sd_pert=%0.1f' % sweep_param_val) )
    
    
    plt.xlabel('Jee+')
    plt.ylabel('active cluster rate')
    plt.title('# active clusters = %d' % nActive)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig( ( fig_path + fig_filename + 'activeErate_vs_JeePlusFor_nActiveClusters%d.pdf' % (nActive) ) , transparent=True)

#%% plot rate of active E cluster vs Jplus
### vary number of clusters active
### different subplot for three different values of sd_pert

cmap = cm.get_cmap('magma', len(n_activeClusters_sweep)+1)
cmap = cmap(range(len(n_activeClusters_sweep)+1))

param_vals_plot = np.array([0, 0.2, 0.6])
JplusEE_slice = 16.725

fig, axs = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(5,10))
plt.rcParams['font.size'] = '12'

for ind_paramPlot in range(0,len(param_vals_plot)):
    
    
    ind_param = np.argmin( np.abs(sweep_param_array-param_vals_plot[ind_paramPlot]) )
    
    ax = axs[ind_paramPlot]
    
    for ind_active in range(0, len(n_activeClusters_sweep) ):
                       
        x = JplusEE_back
        y = nu_e_backSweep[0,:,ind_active, ind_param]
        ax.plot(x, y, 'o', color=cmap[ind_active,:], markersize=2, label=('nActive=%d' % n_activeClusters_sweep[ind_active]))
        ax.plot([JplusEE_slice, JplusEE_slice], [0, 125], color='gray', linewidth=0.5)

    ax.set_ylabel('active rate [spks/s]')
    ax.set_title( ('$\Delta \sigma[I_{E0}]$ = %0.3f' % sweep_param_array[ind_param]) )

plt.xlabel('Jee+')
plt.legend(fontsize=6, loc='upper left')
plt.tight_layout()
plt.savefig( ( fig_path + fig_filename + 'activeErate_vs_JeePlusBack_nActiveClusters.pdf' ) , transparent=True)



#%% plot rate of all populations 
### fix # clusters active
### different subplot for three different values of sd_pert

cmap = cm.get_cmap('magma', len(n_activeClusters_sweep)+1)
cmap = cmap(range(len(n_activeClusters_sweep)+1))

# plotting parameters
param_vals_plot = np.array([0, 0.4, 0.8])
JplusEE_slice = 16.725
nActive = 3
ind_active = np.nonzero( n_activeClusters_sweep == nActive )[0][0]

# begin plotting
fig, axs = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(5,10))
plt.rcParams['font.size'] = '12'

for ind_paramPlot in range(0,len(param_vals_plot)):
    
    ind_param = np.argmin( np.abs(sweep_param_array-param_vals_plot[ind_paramPlot]) )
    ax = axs[ind_paramPlot]
    
    for ind_pop in range(0, 3):
                       
        x = JplusEE_back
        y = nu_e_backSweep[ind_pop,:,ind_active, ind_param]
        ax.plot(x, y, 'o', color=Ecolors[ind_pop], markersize=2, label=('pop %d' % ind_pop))
        ax.plot([JplusEE_slice, JplusEE_slice], [0, 125], color='gray', linewidth=0.5)

    ax.set_ylabel('active cluster rate [spks/sec]')
    ax.set_title( ('$\Delta \sigma[I_{E0}]$ = %0.3f' % sweep_param_array[ind_param]) )

plt.xlabel('Jee+')
plt.suptitle('nActive=%d' % nActive)
plt.legend()
plt.tight_layout()
plt.savefig( ( fig_path + fig_filename + 'allErates_vs_JeePlus_nActiveClusters%d.pdf' % (nActive) ) , transparent=True)


#%% plot rate of active E cluster vs perturbation
### fix JeePlus
### plot different curves for different # clusters active

# set JeePlus
JplusEE = 16.725

plt.figure()
plt.rcParams['font.size'] = '16'

cmap = cm.get_cmap('magma', len(n_activeClusters_sweep)+1)
cmap = cmap(range(len(n_activeClusters_sweep)+1))


for ind_active in range(0, len(n_activeClusters_sweep) ):

    nActive =  n_activeClusters_sweep[ind_active]
    ind_JplusEE_back = np.argmin( np.abs(JplusEE_back - JplusEE) )
    
    x = sweep_param_array
    y = nu_e_backSweep[0,ind_JplusEE_back,ind_active,:]
    plt.plot(x, y, '.', color=cmap[ind_active,:], linewidth=2, label=('nActive=%d' % nActive) )


plt.xlabel('sd E perturbation')
plt.ylabel('active cluster rate')
plt.title('JeePlus = %0.3f' % JplusEE)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig( ( fig_path + fig_filename + 'activeErate_vs_%s_JeePlusBack%0.3f.pdf' % (mft_sweep_param_name, JplusEE) ) , transparent=True)


#%% plot rate of active E cluster vs perturbation
### fix JeePlus
### plot different curves for different # clusters active

# set JeePlus
JplusEE = 16.725

plt.figure()
plt.rcParams['font.size'] = '16'

cmap = cm.get_cmap('magma', len(n_activeClusters_sweep)+1)
cmap = cmap(range(len(n_activeClusters_sweep)+1))

nActive = 3
ind_active = np.nonzero(n_activeClusters_sweep == nActive)[0][0]

ind_JplusEE_back = np.argmin( np.abs(JplusEE_back - JplusEE) )
    
x = sweep_param_array
y = nu_e_backSweep[0,ind_JplusEE_back,ind_active,:]
plt.plot(x, y, '-o', color=cmap[ind_active,:], linewidth=2, label=('nActive=%d' % nActive) )


plt.xlabel('sd E perturbation')
plt.ylabel('active cluster rate')
plt.title('JeePlus = %0.3f' % JplusEE)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig( ( fig_path + fig_filename + 'activeErate_nActive%d_vs_%s_JeePlusBack%0.3f.pdf' % (nActive, mft_sweep_param_name, JplusEE) ) , transparent=True)


#%% plot rate of inactive E cluster vs perturbation
### fix JeePlus
### plot different curves for different # clusters active

# set JeePlus
JplusEE = 16.725

plt.figure()
plt.rcParams['font.size'] = '16'

cmap = cm.get_cmap('magma', len(n_activeClusters_sweep)+1)
cmap = cmap(range(len(n_activeClusters_sweep)+1))


for ind_active in range(0, len(n_activeClusters_sweep) ):

    nActive =  n_activeClusters_sweep[ind_active]
    ind_JplusEE_back = np.argmin( np.abs(JplusEE_back - JplusEE) )
    
    x = sweep_param_array
    y = nu_e_backSweep[1,ind_JplusEE_back,ind_active,:]
    plt.plot(x, y, 'o', color=cmap[ind_active,:], linewidth=2, label=('nActive=%d' % nActive) )


plt.xlabel('sd E perturbation')
plt.ylabel('inactive cluster rate')
plt.title('JeePlus = %0.3f' % JplusEE)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig( ( fig_path + fig_filename + 'inactiveErate_vs_%s_JeePlusBack%0.3f.pdf' % (mft_sweep_param_name, JplusEE) ) , transparent=True)

#%% plot rate of active E cluster vs perturbation
### fix JeePlus
### plot different curves for different # clusters active

# set JeePlus
JplusEE =16.725

plt.figure()
plt.rcParams['font.size'] = '16'

cmap = cm.get_cmap('magma', len(n_activeClusters_sweep)+1)
cmap = cmap(range(len(n_activeClusters_sweep)+1))


for ind_active in range(0, len(n_activeClusters_sweep) ):

    nActive =  n_activeClusters_sweep[ind_active]
    ind_JplusEE_for = np.argmin( np.abs(JplusEE_for - JplusEE) )


    x = sweep_param_array
    y = nu_e_forSweep[0,ind_JplusEE_for,ind_active,:]
    plt.plot(x, y, 'o', color=cmap[ind_active,:], linewidth=2, label=('nActive=%d' % nActive) )


plt.xlabel('sd E perturbation')
plt.ylabel('active cluster rate')
plt.title('JeePlus = %0.3f' % JplusEE)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig( ( fig_path + fig_filename + 'activeErate_vs_%s_JeePlusFor%0.3f.pdf' % (mft_sweep_param_name, JplusEE) ) , transparent=True)

