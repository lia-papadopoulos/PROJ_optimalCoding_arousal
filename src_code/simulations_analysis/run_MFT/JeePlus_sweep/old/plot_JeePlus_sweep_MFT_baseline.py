#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLOT JeePlus_sweep_MFT_fixedInDeg_pEIclusters
"""

#%% BASIC IMPORTS

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib import cm
import sys

import params_JeePlus_sweep_MFT_baseline_pEIclusters as params
func_path2 = params.func_path2
sys.path.append(func_path2)  

from fcn_simulation_loading import fcn_set_sweepParam_string


#%% PATH TO DATA
loadMFT_path = params.loadMFT_path
fig_path = params.fig_path

#%% PARAMETERS FOR LOADING DATA

simID = params.simID
net_type = params.net_type
stim_shape = params.stim_shape
stim_type = params.stim_type
stim_rel_amp = params.stim_rel_amp
sweep_param_name = params.sweep_param_name
n_sweepParams = params.n_sweepParams
swept_params_dict = params.swept_params_dict
indParam = params.indParam
mft_reduced = params.mft_reduced

#%% filenames

sweep_param_str_val = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, indParam)

fname =  ( ('%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f') % \
           ( simID, net_type, sweep_param_str_val, stim_shape, stim_rel_amp) )
    
  
# filename
if mft_reduced == True:
    mft_filename = fname + '_reducedMFT_sweepJeePlus_baseline.mat'
        
    fig_filename = fname + '_reducedMFT_sweepJeePluss_baseline'
else:
    mft_filename = fname + '_MFT_sweepJeePlus_baseline.mat'

    fig_filename = fname + '_MFT_sweepJeePluss_baseline'


#%% LOAD THE MFT DATA

MFT_data = loadmat(loadMFT_path + mft_filename, simplify_cells=True)

#%% UNPACK MFT DATA

mft_params = MFT_data['mft_params']
JeePlus_backSweep_results = MFT_data['JeePlus_backSweep_results']
JeePlus_forSweep_results = MFT_data['JeePlus_forSweep_results']

n_activeClusters_sweep = mft_params['n_active_clusters_sweep']

JplusEE_back = JeePlus_backSweep_results['JplusEE_back']
nu_e_backSweep = JeePlus_backSweep_results['nu_e_backSweep']
nu_i_backSweep = JeePlus_backSweep_results['nu_i_backSweep']
maxRealEig_backSweep = JeePlus_backSweep_results['maxRealEig_backSweep']
stabilityMatrix_backSweep = JeePlus_backSweep_results['stabilityMatrix_backSweep']
maxRealEig_alt_backSweep = JeePlus_backSweep_results['maxRealEig_alt_backSweep']
stabilityMatrix_alt_backSweep = JeePlus_backSweep_results['stabilityMatrix_alt_backSweep']
n_activeClustersE_back = JeePlus_backSweep_results['n_activeClustersE_back']
n_activeClustersI_back = JeePlus_backSweep_results['n_activeClustersI_back']


JplusEE_for = JeePlus_forSweep_results['JplusEE_for']
nu_e_forSweep = JeePlus_forSweep_results['nu_e_forSweep']
nu_i_forSweep = JeePlus_forSweep_results['nu_i_forSweep']
maxRealEig_forSweep = JeePlus_forSweep_results['maxRealEig_forSweep']
stabilityMatrix_forSweep = JeePlus_forSweep_results['stabilityMatrix_forSweep']
maxRealEig_alt_forSweep = JeePlus_forSweep_results['maxRealEig_alt_forSweep']
stabilityMatrix_alt_forSweep = JeePlus_forSweep_results['stabilityMatrix_alt_forSweep']
n_activeClustersE_for = JeePlus_forSweep_results['n_activeClustersE_for']
n_activeClustersI_for = JeePlus_forSweep_results['n_activeClustersI_for']

nClu = int(np.size(mft_params['nu_vec'])/2 - 1)
    
#%%  EIGENVALUE STABILITY

back_unstable = maxRealEig_backSweep>0
back_unstable = np.where(back_unstable, np.nan, 1)
back_unstable_alt = maxRealEig_alt_backSweep>0
back_unstable_alt = np.where(back_unstable_alt, np.nan, 1)

for_unstable = maxRealEig_forSweep>0
for_unstable = np.where(for_unstable, np.nan, 1)
for_unstable_alt = maxRealEig_alt_forSweep>0
for_unstable_alt = np.where(for_unstable_alt, np.nan, 1)


#%% PLOTTING

Ecolors = ['darkmagenta','thistle','mediumorchid']
Icolors = ['darkcyan', 'lightblue', 'mediumturquoise']


#%% plot rate of active E cluster vs Jplus without and with stability


plt.figure()

plt.rcParams['font.size'] = '16'
cmap = cm.get_cmap('plasma', len(n_activeClusters_sweep)+1)
cmap = cmap(range(len(n_activeClusters_sweep)+1))

for ind_active in range(0,len(n_activeClusters_sweep)):
    
    n_active = n_activeClusters_sweep[ind_active]

    
    for indPop in range(0, 1):
        
        x = JplusEE_back
        y = nu_e_backSweep[indPop,:,ind_active]
        plt.plot(x, y, 'o', color=cmap[ind_active,:], markersize=2, label=('n=%d' % n_active) )
        
        x = JplusEE_for
        y = nu_e_forSweep[indPop,:,ind_active]
        plt.plot(x, y, '--', color=cmap[ind_active,:], markersize=1)        


plt.xlim([12, 20])
plt.ylim([0, 100])
plt.xlabel('Jee+')
plt.ylabel('active cluster rate')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(fig_path + fig_filename + 'activeErate_noStability.pdf', transparent=True)



plt.figure()

plt.rcParams['font.size'] = '16'
cmap = cm.get_cmap('plasma', len(n_activeClusters_sweep)+1)
cmap = cmap(range(len(n_activeClusters_sweep)+1))

for ind_active in range(0,len(n_activeClusters_sweep)):
    
    n_active = n_activeClusters_sweep[ind_active]
    
    for indPop in range(0, 1):
        
        x = JplusEE_back
        y = nu_e_backSweep[indPop,:,ind_active]*back_unstable[:, ind_active]
        plt.plot(x, y, '-', color=cmap[ind_active,:], linewidth=2, label=('n=%d' % n_active) )
        
        x = JplusEE_for
        y = nu_e_forSweep[indPop,:,ind_active]*for_unstable[:, ind_active]
        plt.plot(x, y, '--', color=cmap[ind_active,:], linewidth=2)        


plt.xlabel('Jee+')
plt.ylabel('active cluster rate')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(fig_path + fig_filename + 'activeErate_withStability.pdf', transparent=True)


plt.figure()

plt.rcParams['font.size'] = '16'
cmap = cm.get_cmap('plasma', len(n_activeClusters_sweep)+1)
cmap = cmap(range(len(n_activeClusters_sweep)+1))

for ind_active in range(0,len(n_activeClusters_sweep)):
    
    n_active = n_activeClusters_sweep[ind_active]
    
    for indPop in range(0, 1):
        
        x = JplusEE_back
        y = nu_e_backSweep[indPop,:,ind_active]*back_unstable_alt[:, ind_active]
        plt.plot(x, y, 'o', color=cmap[ind_active,:], markersize=2, label=('n=%d' % n_active) )
        
        x = JplusEE_for
        y = nu_e_forSweep[indPop,:,ind_active]*for_unstable_alt[:, ind_active]
        plt.plot(x, y, '-', color=cmap[ind_active,:], linewidth=1)        

plt.xlim([12, 20])
plt.ylim([0, 100])
plt.xlabel('Jee+')
plt.ylabel('active cluster rate')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(fig_path + fig_filename + 'activeErate_withStability_alt.pdf', transparent=True)



#%% plot rate of inactive E cluster vs Jplus without stability


plt.figure()

plt.rcParams['font.size'] = '16'
cmap = cm.get_cmap('plasma', len(n_activeClusters_sweep)+1)
cmap = cmap(range(len(n_activeClusters_sweep)+1))

for ind_active in range(0,len(n_activeClusters_sweep)):
    
    n_active = n_activeClusters_sweep[ind_active]
    
    indPop = n_active
        
    x = JplusEE_back
    y = nu_e_backSweep[indPop,:,ind_active]
    plt.plot(x, y, '-', color=cmap[ind_active,:], linewidth=2, label=('n=%d' % n_active) )
    
    x = JplusEE_for
    y = nu_e_forSweep[indPop,:,ind_active]
    plt.plot(x, y, '--', color=cmap[ind_active,:], linewidth=2)        


plt.xlabel('Jee+')
plt.ylabel('inactive cluster rate')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(fig_path + fig_filename + 'inactiveErate_noStability.pdf', transparent=True)



plt.figure()

plt.rcParams['font.size'] = '16'
cmap = cm.get_cmap('plasma', len(n_activeClusters_sweep)+1)
cmap = cmap(range(len(n_activeClusters_sweep)+1))

for ind_active in range(0,len(n_activeClusters_sweep)):
    
    n_active = n_activeClusters_sweep[ind_active]
    
    indPop = n_active
        
    x = JplusEE_back
    y = nu_e_backSweep[indPop,:,ind_active]*back_unstable[:, ind_active]
    plt.plot(x, y, '-', color=cmap[ind_active,:], linewidth=2, label=('n=%d' % n_active) )
    
    x = JplusEE_for
    y = nu_e_forSweep[indPop,:,ind_active]*for_unstable[:, ind_active]
    plt.plot(x, y, '--', color=cmap[ind_active,:], linewidth=2)        


plt.xlabel('Jee+')
plt.ylabel('inactive cluster rate')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(fig_path + fig_filename + 'inactiveErate_withStability.pdf', transparent=True)


plt.figure()

plt.rcParams['font.size'] = '16'
cmap = cm.get_cmap('plasma', len(n_activeClusters_sweep)+1)
cmap = cmap(range(len(n_activeClusters_sweep)+1))

for ind_active in range(0,len(n_activeClusters_sweep)):
    
    n_active = n_activeClusters_sweep[ind_active]
    
    indPop = n_active
        
    x = JplusEE_back
    y = nu_e_backSweep[indPop,:,ind_active]*back_unstable_alt[:, ind_active]
    plt.plot(x, y, '-', color=cmap[ind_active,:], linewidth=2, label=('n=%d' % n_active) )
    
    x = JplusEE_for
    y = nu_e_forSweep[indPop,:,ind_active]*for_unstable_alt[:, ind_active]
    plt.plot(x, y, '--', color=cmap[ind_active,:], linewidth=2)        


plt.xlabel('Jee+')
plt.ylabel('inactive cluster rate')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(fig_path + fig_filename + 'inactiveErate_withStability_alt.pdf', transparent=True)

    
#%% plot rate of background E cluster vs Jplus without stability


plt.figure()

plt.rcParams['font.size'] = '16'
cmap = cm.get_cmap('plasma', len(n_activeClusters_sweep)+1)
cmap = cmap(range(len(n_activeClusters_sweep)+1))

for ind_active in range(0,len(n_activeClusters_sweep)):
    
    n_active = n_activeClusters_sweep[ind_active]

    indPop = nClu
        
    x = JplusEE_back
    y = nu_e_backSweep[indPop,:,ind_active]
    plt.plot(x, y, '-', color=cmap[ind_active,:], linewidth=2, label=('n=%d' % n_active) )
    
    x = JplusEE_for
    y = nu_e_forSweep[indPop,:,ind_active]
    plt.plot(x, y, '--', color=cmap[ind_active,:], linewidth=2)        


plt.xlabel('Jee+')
plt.ylabel('background cluster rate')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(fig_path + fig_filename + 'backgroundErate_noStability.pdf', transparent=True)


plt.figure()

plt.rcParams['font.size'] = '16'
cmap = cm.get_cmap('plasma', len(n_activeClusters_sweep)+1)
cmap = cmap(range(len(n_activeClusters_sweep)+1))

for ind_active in range(0,len(n_activeClusters_sweep)):
    
    n_active = n_activeClusters_sweep[ind_active]
    
    indPop = nClu
        
    x = JplusEE_back
    y = nu_e_backSweep[indPop,:,ind_active]*back_unstable[:, ind_active]
    plt.plot(x, y, '-', color=cmap[ind_active,:], linewidth=2, label=('n=%d' % n_active) )
    
    x = JplusEE_for
    y = nu_e_forSweep[indPop,:,ind_active]*for_unstable[:, ind_active]
    plt.plot(x, y, '--', color=cmap[ind_active,:], linewidth=2)        


plt.xlabel('Jee+')
plt.ylabel('background cluster rate')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(fig_path + fig_filename + 'backgroundErate_withStability.pdf', transparent=True)


plt.figure()

plt.rcParams['font.size'] = '16'
cmap = cm.get_cmap('plasma', len(n_activeClusters_sweep)+1)
cmap = cmap(range(len(n_activeClusters_sweep)+1))

for ind_active in range(0,len(n_activeClusters_sweep)):
    
    n_active = n_activeClusters_sweep[ind_active]
    
    indPop = nClu
        
    x = JplusEE_back
    y = nu_e_backSweep[indPop,:,ind_active]*back_unstable_alt[:, ind_active]
    plt.plot(x, y, '-', color=cmap[ind_active,:], linewidth=2, label=('n=%d' % n_active) )
    
    x = JplusEE_for
    y = nu_e_forSweep[indPop,:,ind_active]*for_unstable_alt[:, ind_active]
    plt.plot(x, y, '--', color=cmap[ind_active,:], linewidth=2)        


plt.xlabel('Jee+')
plt.ylabel('background cluster rate')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(fig_path + fig_filename + 'backgroundErate_withStability_alt.pdf', transparent=True)


#%% plot rate of active I cluster vs Jplus without and with stability


plt.figure()

plt.rcParams['font.size'] = '16'
cmap = cm.get_cmap('plasma', len(n_activeClusters_sweep)+1)
cmap = cmap(range(len(n_activeClusters_sweep)+1))

for ind_active in range(0,len(n_activeClusters_sweep)):
    
    n_active = n_activeClusters_sweep[ind_active]
    
    for indPop in range(0, 1):
        
        x = JplusEE_back
        y = nu_i_backSweep[indPop,:,ind_active]
        plt.plot(x, y, '-', color=cmap[ind_active,:], linewidth=2, label=('n=%d' % n_active) )
        
        x = JplusEE_for
        y = nu_i_forSweep[indPop,:,ind_active]
        plt.plot(x, y, '--', color=cmap[ind_active,:], linewidth=2)        


plt.xlabel('Jee+')
plt.ylabel('active I cluster rate')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(fig_path + fig_filename + 'activeIrate_noStability.pdf', transparent=True)



plt.figure()

plt.rcParams['font.size'] = '16'
cmap = cm.get_cmap('plasma', len(n_activeClusters_sweep)+1)
cmap = cmap(range(len(n_activeClusters_sweep)+1))

for ind_active in range(0,len(n_activeClusters_sweep)):
    
    n_active = n_activeClusters_sweep[ind_active]
    
    for indPop in range(0, 1):
        
        x = JplusEE_back
        y = nu_i_backSweep[indPop,:,ind_active]*back_unstable[:, ind_active]
        plt.plot(x, y, '-', color=cmap[ind_active,:], linewidth=2, label=('n=%d' % n_active) )
        
        x = JplusEE_for
        y = nu_i_forSweep[indPop,:,ind_active]*for_unstable[:, ind_active]
        plt.plot(x, y, '--', color=cmap[ind_active,:], linewidth=2)        


plt.xlabel('Jee+')
plt.ylabel('active I cluster rate')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(fig_path + fig_filename + 'activeIrate_withStability.pdf', transparent=True)


plt.figure()

plt.rcParams['font.size'] = '16'
cmap = cm.get_cmap('plasma', len(n_activeClusters_sweep)+1)
cmap = cmap(range(len(n_activeClusters_sweep)+1))

for ind_active in range(0,len(n_activeClusters_sweep)):
    
    n_active = n_activeClusters_sweep[ind_active]
    
    for indPop in range(0, 1):
        
        x = JplusEE_back
        y = nu_i_backSweep[indPop,:,ind_active]*back_unstable_alt[:, ind_active]
        plt.plot(x, y, '-', color=cmap[ind_active,:], linewidth=2, label=('n=%d' % n_active) )
        
        x = JplusEE_for
        y = nu_i_forSweep[indPop,:,ind_active]*for_unstable_alt[:, ind_active]
        plt.plot(x, y, '--', color=cmap[ind_active,:], linewidth=2)        


plt.xlabel('Jee+')
plt.ylabel('active I cluster rate')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(fig_path + fig_filename + 'activeIrate_withStability_alt.pdf', transparent=True)


#%% for a fixed number of active clusters, plot all pops

n_activePlot = 4 
ind_active = np.nonzero(n_activeClusters_sweep==n_activePlot)[0][0]

plt.figure()
  

for indPop in range(0, nClu+1):
    
    if indPop < n_activePlot:
        cE = Ecolors[0]
        cI = Icolors[0]
    elif ( (indPop >= n_activePlot) and (indPop < nClu)):
        cE = Ecolors[1]
        cI = Icolors[1]
    else:
        cE = Ecolors[2]
        cI = Icolors[2]
        
    x = JplusEE_back
    y = nu_e_backSweep[indPop,:,ind_active]
    plt.plot(x, y, '-', color=cE, linewidth=2)
    
    x = JplusEE_for
    y = nu_e_forSweep[indPop,:,ind_active]
    plt.plot(x, y, '--', color=cE, linewidth=2)   

    x = JplusEE_back
    y = nu_i_backSweep[indPop,:,ind_active]
    plt.plot(x, y, '-', color=cI, linewidth=2)
    
    x = JplusEE_for
    y = nu_i_forSweep[indPop,:,ind_active]
    plt.plot(x, y, '--', color=cI, linewidth=2) 


plt.xlabel('Jee+')
plt.ylabel('pop rates')
plt.title('%d active' % n_activePlot)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig((fig_path + fig_filename + 'allRates_noStability_nCluActive%d.pdf' % (n_activePlot)), transparent=True)



#%% plot largest real part of eigenvalues of stability matrix


plt.figure()

plt.rcParams['font.size'] = '16'
cmap = cm.get_cmap('plasma', len(n_activeClusters_sweep)+1)
cmap = cmap(range(len(n_activeClusters_sweep)+1))

for ind_active in range(0,len(n_activeClusters_sweep)):
    
    n_active = n_activeClusters_sweep[ind_active]
    
        
    x = JplusEE_back
    y = maxRealEig_backSweep[:, ind_active]
    plt.plot(x, y, 'o', color=cmap[ind_active,:], markersize=1, label=('n=%d' % n_active) )
        
    x = JplusEE_for
    y = maxRealEig_forSweep[:, ind_active]
    plt.plot(x, y, 'x', color=cmap[ind_active,:], markersize=1)    

    x = JplusEE_for
    y = np.zeros(len(JplusEE_for))
    plt.plot(x,y,color='black')    


plt.xlabel('Jee+')
plt.ylabel('max Re(\lambda) S')
plt.legend(fontsize=8)
plt.tight_layout()

plt.savefig(fig_path + fig_filename + 'largest_realPart_eig_stabilityMatrix.pdf', transparent=True)


plt.figure()

plt.rcParams['font.size'] = '16'
cmap = cm.get_cmap('plasma', len(n_activeClusters_sweep)+1)
cmap = cmap(range(len(n_activeClusters_sweep)+1))

for ind_active in range(0,len(n_activeClusters_sweep)):
    
    n_active = n_activeClusters_sweep[ind_active]
    
        
    x = JplusEE_for
    y = np.zeros(len(JplusEE_for))
    plt.plot(x,y,color='black', linewidth=1)  
    
    x = JplusEE_for
    y = maxRealEig_alt_forSweep[:, ind_active]
    plt.plot(x, y, '--', color='gray', linewidth=1) 
    
    x = JplusEE_back
    y = maxRealEig_alt_backSweep[:, ind_active]
    plt.plot(x, y, 'o', color=cmap[ind_active,:], markersize=1, label=('n=%d' % n_active) )
        


plt.ylim([-20,30])
plt.xlim([12,20])
plt.xlabel('Jee+')
plt.ylabel('max Re(\lambda) S alt')
plt.legend(fontsize=8)
plt.tight_layout()

plt.savefig(fig_path + fig_filename + 'largest_realPart_eig_stabilityMatrix_alt.pdf', transparent=True)