#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLOT cluster statistics vs sd perturbation
"""

#%% BASIC IMPORTS

import sys
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib import cm


#%% PATH TO DATA

loadMFT_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/test_stim_expSyn/MFT_sweep_JeePlus_sweep_sd_nu_ext_e_pert/')
loadSIM_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/test_stim_expSyn/')
loadANALYSIS_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/test_stim_expSyn/clusterRates_numActiveClusters/')

fig_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/Figures/test_stim_expSyn/clusterRates_MFT_sims_sweepArousal/')


#%% PARAMETERS FOR LOADING DATA

# simulation ID
simID = '113020232105' #[cluster, sd pert, 5 stim, poisson external inputs, Jeeplus = 15.75]
#simID = '102320221109' #[cluster, sd pert, 5 stim, poisson external inputs, Jeeplus = 15.75]

# network name
net_type = 'baseEIclu'

# stim shape
stim_shape = 'diff2exp'

# stim type
stim_type = ''

# relative stimulation amplitude
stim_rel_amp = 0.05

# sweep param name
sweep_param_name_sim = 'same_eachClustersd_nu_ext_e_pert'
#sweep_param_name_sim = 'sd_nu_ext_e_pert'

# sweep param value
sweep_param_array = np.arange(0,0.45,0.05)

# min probability active
min_probActive = 0.1
    
# window length
windLength = 25e-3

# rate threshold
rateThresh = 0

# nActive array
nActive_array_plot_sims = np.arange(1,19)

gainBased = True




#%% PARAMETERS FOR LOADING MFT DATA


# mft reduced
mft_reduced = True

# network name
net_type_mft = 'baseEIclu'

# simulation ID
mftID = '102320221109' #[cluster, sd pert, 5 stim, poisson external inputs, Jeeplus = 15.75]

# stim shape
stim_shape_mft = 'diff2exp'

# relative stimulation amplitude
stim_rel_amp_mft = 0.05

# sweep param name
sweep_param_name_mft = 'sd_nu_ext_e_pert'

# sweep param value
sweep_param_val_plot_mft = np.array([0, 0.2, 0.4])
sweep_param_array_mft = np.arange(0,0.41,0.01)


# mft value of JeePlus
#JeePlus_mft_plot = 16.725

# number of active clusters to plot rate for
# want to set to most likely # of active clusters in simulations
nActive_plot_mft = 3

# nActive array
nActive_array_plot_mft = np.arange(1,11)


fig_path = fig_path + sweep_param_name_mft + '/'

#%% FILENAMES
    
# simulation filename
sim_filename = ( '%s_%s_sweep_%s%0.3f_network%d_IC%d_stim%d_stimType_%s_stim_rel_amp%0.3f_simulationData.mat' % \
               (simID, net_type, sweep_param_name_sim, sweep_param_array[0], 0, 0, 0, stim_shape, stim_rel_amp ) )


# analysis filename
if gainBased:
    analysis_filename = ( '%s_%s_sweep_%s%0.3f_stimType_%s_stim_rel_amp%0.3f__clusterRates_numActiveClusters_gainBased.mat' % \
                    ( simID, net_type, sweep_param_name_sim, sweep_param_array[0], stim_shape, stim_rel_amp) )
else:
    analysis_filename = ( '%s_%s_sweep_%s%0.3f_stimType_%s_stim_rel_amp%0.3f__clusterRates_numActiveClusters.mat' % \
                    ( simID, net_type, sweep_param_name_sim, sweep_param_array[0], stim_shape, stim_rel_amp) )

# mft filename
if mft_reduced == True:
    
    # filename
    mft_filename = ( '%s_%s_sweep_%s%0.3f_stimType_%s_stim_rel_amp%0.3f_reducedMFT_sweepJeePlus.mat' % \
                   ( mftID, net_type, sweep_param_name_mft, sweep_param_array[0], stim_shape, stim_rel_amp ) )
        
else:
    
    sys.exit('have only run reduced MFT')
    
# filename for figures
fig_filename_sim = ( '%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f_rateThresh%0.1fHz_windowStd%0.3fs_' % \
               ( simID, net_type, sweep_param_name_sim, stim_shape, stim_rel_amp, rateThresh, windLength ) )

fig_filename_mft = ( '%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f_rateThresh%0.1fHz_windowStd%0.3fs_' % \
               ( mftID, net_type, sweep_param_name_mft, stim_shape, stim_rel_amp, rateThresh, windLength ) )
    
#%% LOAD EXAMPLE SIMULATION DATA

SIM_data = loadmat(loadSIM_path + sim_filename, simplify_cells=True)
sim_params = SIM_data['sim_params']
nClu = sim_params['p']
Jee_plus_sims = sim_params['JplusEE']


#%% LOAD EXAMPLE ANALYSIS DATA

ANALYSIS_data = loadmat(loadANALYSIS_path + analysis_filename, simplify_cells=True)
analysis_params = ANALYSIS_data['parameters']
nNets = analysis_params['nNets']
rate_thresh_array = analysis_params['rate_thresh']


#%% LOAD EXAMPLE MFT DATA

MFT_data = loadmat(loadMFT_path + mft_filename, simplify_cells=True)
mft_params = MFT_data['mft_params']
JeePlus_backSweep_results = MFT_data['JeePlus_backSweep_results']
JeePlus_forSweep_results = MFT_data['JeePlus_forSweep_results']

n_activeClusters_sweep = mft_params['n_active_clusters_sweep']
JplusEE_back = JeePlus_backSweep_results['JplusEE_back']
JplusEE_for = JeePlus_forSweep_results['JplusEE_for']

n_nActive_clusters = np.size(n_activeClusters_sweep)
n_JplusEE_back = np.size(JplusEE_back)
n_JplusEE_for = np.size(JplusEE_for)


#%% WINDOW LENGTH AND RATE THRESHOLD TO PLOT
indThresh_plot = np.nonzero(rate_thresh_array==rateThresh)[0][0]


#%% SIMULATIONS

# initialize arrays for quantities that we want to plot
activeRate_XActiveClusters_E = np.zeros((nClu+1, np.size(sweep_param_array)))
activeRate_XActiveClusters_E_error = np.zeros((nClu+1, np.size(sweep_param_array)))

# initialize arrays for quantities that we want to plot
inactiveRate_XActiveClusters_E = np.zeros((nClu+1, np.size(sweep_param_array)))
inactiveRate_XActiveClusters_E_error = np.zeros((nClu+1, np.size(sweep_param_array)))


# initialize arrays for quantities that we want to plot
activeRate_XActiveClusters_E_eachNet = np.zeros((nNets, nClu+1, np.size(sweep_param_array)))

# initialize arrays for quantities that we want to plot
inactiveRate_XActiveClusters_E_eachNet = np.zeros((nNets, nClu+1, np.size(sweep_param_array)))

# probability of n active clusters
prob_nActive_clusters_E_eachNet = np.zeros((nNets, nClu+1, np.size(sweep_param_array)))


prob_nActive_clusters_E = np.zeros((nClu+1, np.size(sweep_param_array)))
prob_nActive_clusters_E_error = np.zeros((nClu+1, np.size(sweep_param_array)))

popAvg_rate_E = np.zeros((np.size(sweep_param_array)))
popAvg_rate_E_error = np.zeros((np.size(sweep_param_array)))

popAvg_activeCluster_rate_e = np.zeros((np.size(sweep_param_array)))
popAvg_activeCluster_rate_e_error = np.zeros((np.size(sweep_param_array)))

popAvg_inactiveCluster_rate_e = np.zeros((np.size(sweep_param_array)))
popAvg_inactiveCluster_rate_e_error = np.zeros((np.size(sweep_param_array)))


# loop over perturbation
for ind_sweep_param in range(0, np.size(sweep_param_array)):
    
    # value of swept parameter
    sweep_param_val = sweep_param_array[ind_sweep_param]
      
        
    # analysis filename
    if gainBased:
        analysis_filename = ( '%s_%s_sweep_%s%0.3f_stimType_%s_stim_rel_amp%0.3f__clusterRates_numActiveClusters_gainBased.mat' % \
                        ( simID, net_type, sweep_param_name_sim, sweep_param_val, stim_shape, stim_rel_amp ) )   
    else:
        analysis_filename = ( '%s_%s_sweep_%s%0.3f_stimType_%s_stim_rel_amp%0.3f__clusterRates_numActiveClusters.mat' % \
                        ( simID, net_type, sweep_param_name_sim, sweep_param_val, stim_shape, stim_rel_amp ) )      
        
    # load the data
    ANALYSIS_data = loadmat(loadANALYSIS_path + analysis_filename, simplify_cells=True)

    # fill up the arrays
    
    activeRate_XActiveClusters_E[:, ind_sweep_param] = ANALYSIS_data['netAvg_avgRate_active_XActiveClusters_E'][:, indThresh_plot].copy()
    activeRate_XActiveClusters_E_error[:, ind_sweep_param] = ANALYSIS_data['netStd_avgRate_active_XActiveClusters_E'][:, indThresh_plot].copy()
    
    inactiveRate_XActiveClusters_E[:, ind_sweep_param] = ANALYSIS_data['netAvg_avgRate_inactive_XActiveClusters_E'][:, indThresh_plot].copy()
    inactiveRate_XActiveClusters_E_error[:, ind_sweep_param] = ANALYSIS_data['netStd_avgRate_inactive_XActiveClusters_E'][:, indThresh_plot].copy()

    prob_nActive_clusters_E[:, ind_sweep_param] = ANALYSIS_data['netAvg_prob_nActive_clusters_E'][:, indThresh_plot].copy()
    prob_nActive_clusters_E_error[:, ind_sweep_param] = ANALYSIS_data['netStd_prob_nActive_clusters_E'][:, indThresh_plot].copy()

    popAvg_rate_E[ind_sweep_param] = ANALYSIS_data['netAvg_popAvg_rate_E']
    popAvg_rate_E_error[ind_sweep_param] = ANALYSIS_data['netStd_popAvg_rate_E']
    

    popAvg_activeCluster_rate_e[ind_sweep_param]  = ANALYSIS_data['netAvg_popAvg_activeCluster_rate_e'][indThresh_plot].copy()
    popAvg_activeCluster_rate_e_error[ind_sweep_param]  = ANALYSIS_data['netStd_popAvg_activeCluster_rate_e'][indThresh_plot].copy()
    
    popAvg_inactiveCluster_rate_e[ind_sweep_param]  = ANALYSIS_data['netAvg_popAvg_inactiveCluster_rate_e'][indThresh_plot].copy()
    popAvg_inactiveCluster_rate_e_error[ind_sweep_param]  = ANALYSIS_data['netStd_popAvg_inactiveCluster_rate_e'][indThresh_plot].copy()    
    
    
    # loop over networks
    for indNet in range(0, nNets):
        
        activeRate_XActiveClusters_E_eachNet[:, :, ind_sweep_param] = ANALYSIS_data['trialAvg_avgRate_active_XActiveClusters_E'][:, :, indThresh_plot].copy()
        inactiveRate_XActiveClusters_E_eachNet[:, :, ind_sweep_param] = ANALYSIS_data['trialAvg_avgRate_inactive_XActiveClusters_E'][:, :, indThresh_plot].copy()
        prob_nActive_clusters_E_eachNet[:, :, ind_sweep_param] = ANALYSIS_data['trialAvg_prob_nActive_clusters_E'][:, :, indThresh_plot].copy()


# most likely # active clusters at each perturbation
mostLikely_nActive_clusters_E = np.argmax(prob_nActive_clusters_E,0)


#%% MFT

# initialize
activeRate_XActiveClusters_E_mftBack = np.zeros((n_JplusEE_back, n_nActive_clusters, np.size(sweep_param_array_mft)))
activeRate_XActiveClusters_E_mftFor = np.zeros((n_JplusEE_for, n_nActive_clusters, np.size(sweep_param_array_mft)))

inactiveRate_XActiveClusters_E_mftBack = np.zeros((n_JplusEE_back, n_nActive_clusters, np.size(sweep_param_array_mft)))
inactiveRate_XActiveClusters_E_mftFor = np.zeros((n_JplusEE_for, n_nActive_clusters, np.size(sweep_param_array_mft)))

# loop over perturbation
for ind_sweep_param in range(0, np.size(sweep_param_array_mft)):
    
    # value of swept parameter
    sweep_param_val = sweep_param_array_mft[ind_sweep_param]


    # filename
    mft_filename = ( '%s_%s_sweep_%s%0.3f_stimType_%s_stim_rel_amp%0.3f_reducedMFT_sweepJeePlus.mat' % \
                   ( mftID, net_type, sweep_param_name_mft, sweep_param_val, stim_shape, stim_rel_amp ) )

    # load the data
    MFT_data = loadmat(loadMFT_path + mft_filename, simplify_cells=True)

    # backwards and forwards sweep
    JeePlus_backSweep_results = MFT_data['JeePlus_backSweep_results']
    JeePlus_forSweep_results = MFT_data['JeePlus_forSweep_results']
    
    # unpack 
    if sweep_param_val == 0.:
        nu_bar_e_back = JeePlus_backSweep_results['nu_e_backSweep']
        nu_bar_e_for = JeePlus_forSweep_results['nu_e_forSweep']

    else:
        nu_bar_e_back = JeePlus_backSweep_results['nu_bar_e_back']
        nu_bar_e_for = JeePlus_forSweep_results['nu_bar_e_for']

    # backwards
    activeRate_XActiveClusters_E_mftBack[:, :, ind_sweep_param] = nu_bar_e_back[0, :, :]
    inactiveRate_XActiveClusters_E_mftBack[:, :, ind_sweep_param] = nu_bar_e_back[1, :, :]
    
    # forwards
    activeRate_XActiveClusters_E_mftFor[:, :, ind_sweep_param] = nu_bar_e_for[0, :, :]
    inactiveRate_XActiveClusters_E_mftFor[:, :, ind_sweep_param] = nu_bar_e_for[1, :, :]


#%% FIND VALUE OF J AT WHICH BASELINE SIMULATIONS AND MFT BEST MATCH

# most likely # active clusters baseline
mostLikely_nActive_base = int(np.argmax(prob_nActive_clusters_E[:, np.nonzero(sweep_param_array==0)[0]]))
indMFT_mostLikely_nActive_base = np.nonzero(n_activeClusters_sweep == mostLikely_nActive_base)[0]
activeRate_mostLikely_nActive_base = activeRate_XActiveClusters_E[mostLikely_nActive_base, np.nonzero(sweep_param_array==0)[0]]

# mft J+ with closest match
activeRate_mft_base = activeRate_XActiveClusters_E_mftBack[:, indMFT_mostLikely_nActive_base, np.nonzero(sweep_param_array_mft==0)[0]].copy()
mft_indJeePlus_back_simMatch = np.argmin(np.abs(activeRate_mostLikely_nActive_base - activeRate_mft_base))
mft_JeePlus_back_simMatch = JplusEE_back[mft_indJeePlus_back_simMatch]
print(mft_JeePlus_back_simMatch)

JeePlus_mft_plot = mft_JeePlus_back_simMatch

#%% PLOTTING


#%% plot in/active cluster rate vs JeePlus for n=X; three different values of perturbation


ind_nActivePlot = np.nonzero(n_activeClusters_sweep == nActive_plot_mft)[0]

plt.rcParams.update({'font.size': 5})
fig = plt.figure(figsize=(5,4))
       
cmap_active = cm.get_cmap('Greens', np.size(sweep_param_val_plot_mft)+1)
cmap_active = cmap_active(range(np.size(sweep_param_val_plot_mft)+1))
cmap_active = cmap_active[1:,:]

cmap_inactive = cm.get_cmap('Purples', np.size(sweep_param_val_plot_mft)+1)
cmap_inactive = cmap_inactive(range(np.size(sweep_param_val_plot_mft)+1))
cmap_inactive = cmap_inactive[1:,:]


for ind_plotParam, param in enumerate(sweep_param_val_plot_mft):
    
    ind_param = np.argmin( np.abs(sweep_param_array_mft - param))
    
    x = JplusEE_back
    y = activeRate_XActiveClusters_E_mftBack[:, ind_nActivePlot, ind_param]

    if ind_plotParam == 1:
        label_active = 'active'
        label_inactive = 'inactive'
    else:
        label_active = ''
        label_inactive = ''
        
    plt.plot(x, y, color=cmap_active[ind_plotParam,:], linewidth=2, label = label_active)
    
    
    x = JplusEE_back
    y = inactiveRate_XActiveClusters_E_mftBack[:, ind_nActivePlot, ind_param]

    plt.plot(x, y, color=cmap_inactive[ind_plotParam,:], linewidth=2, label = label_inactive)
    
    plt.text(22, 50-8*ind_plotParam, (r'$\Delta s[\mathrm{I_{EO}}] =  %0.1f$' % param), fontsize=4)

x = np.array([JeePlus_mft_plot, JeePlus_mft_plot])
y = np.array([0, 125])
plt.plot(x, y, color='gray', linewidth=2)
plt.ylim([0, 125])

#plt.title('nActive = %d' % nActive_plot_mft)
plt.xlabel('synpatic potentiation $J_{EE}^{+}$')
plt.ylabel('E cluster rate [spks/sec]')
plt.legend(fontsize=4, loc='upper left')
plt.savefig(fig_path + fig_filename_mft + 'mft_activeErate_vs_JeePlus_nActive%d.pdf' % (nActive_plot_mft), transparent=True)



#%% plot in/active cluster rate vs perturbation for fixed Jee+ simulations; avg across # active clusters


plt.rcParams.update({'font.size': 5})
fig = plt.figure(figsize=(5,4))
       

x = sweep_param_array
y = popAvg_activeCluster_rate_e
yerr = popAvg_activeCluster_rate_e_error
plt.fill_between(x, y-yerr, y+yerr, color='mediumseagreen', alpha=0.3)
plt.plot( x, y, '-o', linewidth=1, color='mediumseagreen', label='active')


x = sweep_param_array
y = popAvg_inactiveCluster_rate_e
yerr = popAvg_inactiveCluster_rate_e_error
plt.fill_between(x, y-yerr, y+yerr, color='rebeccapurple', alpha=0.3)
plt.plot( x, y, '-o', color='rebeccapurple', linewidth=1, label='inactive')
    

x = sweep_param_array    
y = popAvg_rate_E
yerr = popAvg_rate_E_error
plt.fill_between(x, y-yerr, y+yerr, color='gray', alpha=0.3)
plt.plot(x, y, color='gray')


#plt.title(r'$J^{+}_{\mathrm{EE}} = %0.3f$' % Jee_plus_sims)
plt.xlabel(r'arousal modulation $\Delta s[\mathrm{I_{EO}}]$')
plt.ylabel('E rate [spks/sec]')
plt.legend(fontsize=4, loc='upper right')
plt.savefig( ( (fig_path + fig_filename_sim + 'sims_active_inactive_Erate_vs_%s_JeePlus%0.3f_avg_nActive.pdf') % (sweep_param_name_sim, Jee_plus_sims)), transparent=True)


#%% plot in/active cluster rate for fixed Jee+ simulations; most likely # active clusters

plt.rcParams.update({'font.size': 5})
fig = plt.figure(figsize=(5,4))
       

x = sweep_param_array
y = np.zeros(len(sweep_param_array))
yerr = np.zeros(len(sweep_param_array))

for i in range(0,len(y)):
    
    y[i] = activeRate_XActiveClusters_E[mostLikely_nActive_clusters_E[i], i]
    yerr[i] = activeRate_XActiveClusters_E_error[mostLikely_nActive_clusters_E[i], i]

plt.fill_between(x, y-yerr, y+yerr, color='mediumseagreen', alpha=0.3)
plt.plot( x, y, '-o', linewidth=1, color='mediumseagreen', label='active')


x = sweep_param_array
y = np.zeros(len(sweep_param_array))
yerr = np.zeros(len(sweep_param_array))

for i in range(0,len(y)):
    
    y[i] = inactiveRate_XActiveClusters_E[mostLikely_nActive_clusters_E[i], i]
    yerr[i] = inactiveRate_XActiveClusters_E_error[mostLikely_nActive_clusters_E[i], i]

plt.fill_between(x, y-yerr, y+yerr, color='rebeccapurple', alpha=0.3)
plt.plot( x, y, '-o', color='rebeccapurple', linewidth=1, label='inactive')
    



#plt.title(r'$J^{+}_{\mathrm{EE}} = %0.3f$' % Jee_plus_sims)
plt.xlabel(r'arousal modulation $\Delta s[\mathrm{I_{EO}}]$')
plt.ylabel('E rate [spks/sec]')
plt.legend(fontsize=4, loc='upper right')
plt.savefig( ( (fig_path + fig_filename_sim + 'sims_active_inactive_Erate_vs_%s_JeePlus%0.3f_mostLikely_nActive.pdf') % (sweep_param_name_sim, Jee_plus_sims)), transparent=True)

#%% plot in/active cluster rate for fixed Jee+ simulations; average across nActive with prob>x

plt.rcParams.update({'font.size': 5})
fig = plt.figure(figsize=(5,4))
       
### active
x = sweep_param_array
y_eachNet = np.zeros((nNets, len(sweep_param_array)))


for indNet in range(0, nNets):
    for ind_sweptParam in range(0, len(sweep_param_array)):
    
        avg_nActive = np.nonzero(prob_nActive_clusters_E_eachNet[indNet, :, ind_sweptParam] > min_probActive)[0]
        y_eachNet[indNet, ind_sweptParam] = np.mean(activeRate_XActiveClusters_E_eachNet[indNet, avg_nActive, ind_sweptParam])

y = np.mean(y_eachNet, 0)
yerr = np.std(y_eachNet, 0)/np.sqrt(nNets)

plt.fill_between(x, y-yerr, y+yerr, color='mediumseagreen', alpha=0.3)
plt.plot( x, y, '-o', linewidth=1, color='mediumseagreen', label='active')



### inactive
x = sweep_param_array
y_eachNet = np.zeros((nNets, len(sweep_param_array)))

for indNet in range(0, nNets):
    for ind_sweptParam in range(0, len(sweep_param_array)):
    
        avg_nActive = np.nonzero(prob_nActive_clusters_E_eachNet[indNet, :, ind_sweptParam] > min_probActive)[0]
        y_eachNet[indNet, ind_sweptParam] = np.mean(inactiveRate_XActiveClusters_E_eachNet[indNet, avg_nActive, ind_sweptParam])

y = np.mean(y_eachNet, 0)
yerr = np.std(y_eachNet, 0)/np.sqrt(nNets)

plt.fill_between(x, y-yerr, y+yerr, color='rebeccapurple', alpha=0.3)
plt.plot( x, y, '-o', linewidth=1, color='rebeccapurple', label='active') 



### settings
plt.ylim([0, 50])
plt.title(r'$J^{+}_{\mathrm{EE}} = %0.3f$' % Jee_plus_sims)
plt.xlabel(r'arousal modulation $\Delta s[\mathrm{I_{EO}}]$')
plt.ylabel('E rate [spks/sec]')
plt.legend(fontsize=4, loc='upper right')
plt.savefig( ( (fig_path + fig_filename_sim + 'sims_active_inactive_Erate_vs_%s_JeePlus%0.3f_avg_mostLikely_nActive_probThresh%0.2f.pdf') % (sweep_param_name_sim, Jee_plus_sims, min_probActive)), transparent=True)


#%% plot in/active cluster rate vs perturbation for fixed Jee+ mft; average over n active

ind_JeePlus_plot = np.argmin( np.abs(JplusEE_back - JeePlus_mft_plot) )

mft_nActive_plot_array = np.array([2,3,4,5])
inds_nActivePlot = np.zeros( len(mft_nActive_plot_array) )
for count, nActive in enumerate(mft_nActive_plot_array):
    inds_nActivePlot[count] = np.nonzero(n_activeClusters_sweep == mft_nActive_plot_array[count])[0][0]

inds_nActivePlot = inds_nActivePlot.astype(int)

plt.rcParams.update({'font.size': 5})
fig = plt.figure(figsize=(5,4))
    
x = sweep_param_array_mft
y = np.mean(activeRate_XActiveClusters_E_mftBack[ind_JeePlus_plot, inds_nActivePlot, :], axis=0)
plt.plot( x, y, '-o', linewidth=1, color='mediumseagreen', label='active')       

x = sweep_param_array_mft
y = np.mean(inactiveRate_XActiveClusters_E_mftBack[ind_JeePlus_plot, inds_nActivePlot, :], axis=0)
plt.plot( x, y, '-o', linewidth=1, color='rebeccapurple', label='inactive')   


#plt.title(r'$J^{+}_{\mathrm{EE}} = %0.3f$' % Jee_plus_sims)
plt.xlabel(r'arousal modulation $\Delta s[\mathrm{I_{EO}}]$')
plt.ylabel('E rate [spks/sec]')
plt.legend(fontsize=4, loc='upper right')
plt.savefig( ( (fig_path + fig_filename_mft + 'mft_active_inactive_Erate_vs_%s_JeePlus%0.3f_avg_nActive.pdf') % (sweep_param_name_mft, JeePlus_mft_plot)), transparent=True)


#%% plot in/active cluster rate vs perturbation for fixed Jee+ mft; most likely n active from simulations


ind_JeePlus_plot = np.argmin( np.abs(JplusEE_back - JeePlus_mft_plot) )

ind_nActivePlot = np.zeros( len(sweep_param_array) )
for i in range(0, len(sweep_param_array)):
    ind_nActivePlot[i] = np.nonzero(n_activeClusters_sweep == mostLikely_nActive_clusters_E[i])[0][0]

ind_nActivePlot = ind_nActivePlot.astype(int)

plt.rcParams.update({'font.size': 5})
fig = plt.figure(figsize=(5,4))
    
x = sweep_param_array
y = np.zeros(len(x))
for i in range(0,len(x)):
    indMFT = np.argmin(np.abs(sweep_param_array_mft - sweep_param_array[i]))
    y[i] = activeRate_XActiveClusters_E_mftBack[ind_JeePlus_plot, ind_nActivePlot[i], indMFT]
    
plt.plot( x, y, '-o', linewidth=1, color='mediumseagreen', label='active')       

x = sweep_param_array
y = np.zeros(len(x))
for i in range(0,len(x)):
    indMFT = np.argmin(np.abs(sweep_param_array_mft - sweep_param_array[i]))
    y[i] = inactiveRate_XActiveClusters_E_mftBack[ind_JeePlus_plot, ind_nActivePlot[i], indMFT]
    
plt.plot( x, y, '-o', linewidth=1, color='rebeccapurple', label='inactive')   


#plt.title(r'$J^{+}_{\mathrm{EE}} = %0.3f$' % Jee_plus_sims)
plt.xlabel(r'arousal modulation $\Delta s[\mathrm{I_{EO}}]$')
plt.ylabel('E rate [spks/sec]')
plt.legend(fontsize=4, loc='upper right')
plt.savefig( ( (fig_path + fig_filename_mft + 'mft_active_inactive_Erate_vs_%s_JeePlus%0.3f_mostLikely_nActive.pdf') % (sweep_param_name_mft, JeePlus_mft_plot)), transparent=True)


#%% plot in/active cluster rate vs perturbation for fixed Jee+ mft; most likely n active from simulations

ind_JeePlus_plot = np.argmin( np.abs(JplusEE_back - JeePlus_mft_plot) )
prob_thresh = 0.1

x = sweep_param_array
yactive = np.zeros(len(x))
yinactive = np.zeros(len(x))

for i in range(0, len(sweep_param_array)):
    avg_nActive_sims = np.nonzero(prob_nActive_clusters_E[:, i] > prob_thresh)[0]
    inds_nActive_plot = np.array([])
    for j in range(0, len(avg_nActive_sims)):
        inds_nActivePlot = np.append(inds_nActivePlot, np.nonzero(n_activeClusters_sweep ==avg_nActive_sims[j])[0])
    
    indMFT = np.argmin(np.abs(sweep_param_array_mft - sweep_param_array[i]))
    yactive[i] = np.mean(activeRate_XActiveClusters_E_mftBack[ind_JeePlus_plot, inds_nActivePlot, indMFT])
    yinactive[i] = np.mean(inactiveRate_XActiveClusters_E_mftBack[ind_JeePlus_plot, inds_nActivePlot, indMFT])

plt.rcParams.update({'font.size': 5})
fig = plt.figure(figsize=(5,4))

plt.plot( x, yactive, '-o', linewidth=1, color='mediumseagreen', label='active')       
plt.plot( x, yinactive, '-o', linewidth=1, color='rebeccapurple', label='inactive')   
plt.ylim([0, 50]) 

#plt.title(r'$J^{+}_{\mathrm{EE}} = %0.3f$' % Jee_plus_sims)
plt.xlabel(r'arousal modulation $\Delta s[\mathrm{I_{EO}}]$')
plt.ylabel('E rate [spks/sec]')
plt.legend(fontsize=4, loc='upper right')
plt.savefig( ( (fig_path + fig_filename_mft + 'mft_active_inactive_Erate_vs_%s_JeePlus%0.3f_avg_mostLikely_nActive_probThresh%0.2f.pdf') % (sweep_param_name_mft, JeePlus_mft_plot, prob_thresh)), transparent=True)


#%% plot active cluster rate vs perturbation for fixed Jee+ mft; different curves for different # active clusters

ind_JeePlus_plot = np.argmin( np.abs(JplusEE_back - JeePlus_mft_plot) )

plt.rcParams.update({'font.size': 5})
fig = plt.figure(figsize=(5,4))

legend_vals = np.array([2, 6, 10])       

cmap = cm.get_cmap('magma', np.size(nActive_array_plot_sims)+2)
cmap = cmap(range(np.size(nActive_array_plot_sims)+2))
cmap = cmap[1:,:]

for ind, nActive in enumerate(nActive_array_plot_mft):
    
    ind_nActivePlot = np.argmin( np.abs(n_activeClusters_sweep - nActive))
    
    x = sweep_param_array_mft
    y = activeRate_XActiveClusters_E_mftBack[ind_JeePlus_plot, ind_nActivePlot, :]

    if nActive in legend_vals:
        plt.plot( x, y, '-o', color=cmap[nActive, :], linewidth=1, label=('nActive = %d' % nActive) )
    else:
        plt.plot( x, y, '-o', color=cmap[nActive, :], linewidth=1 )        
        
        
    x = sweep_param_array_mft
    y = inactiveRate_XActiveClusters_E_mftBack[ind_JeePlus_plot, ind_nActivePlot, :]

    if nActive in legend_vals:
        plt.plot( x, y, '-', color=cmap[nActive, :], linewidth=1 )
    else:
        plt.plot( x, y, '-', color=cmap[nActive, :], linewidth=1 )   


#plt.title(r'$J^{+}_{\mathrm{EE}} = %0.3f$' % JeePlus_mft_plot)
plt.xlabel(r'arousal modulation $\Delta s[\mathrm{I_{EO}}]$')
plt.ylabel('active E rate [spks/sec]')
plt.legend(fontsize=4, loc='upper right')
plt.savefig( ( (fig_path + fig_filename_mft + 'mft_activeErate_vs_%s_JeePlus%0.3f.pdf') % (sweep_param_name_mft, JeePlus_mft_plot)), transparent=True)



#%% plot active cluster rate vs perturbation for fixed Jee+ simulations; different curves for different # active clusters

plt.rcParams.update({'font.size': 2})
fig, axs = plt.subplots(1,figsize=(2.85, 2.15))
ax = axs
       
legend_vals = np.array([2, 6, 10, 14, 18])

cmap = cm.get_cmap('magma', np.size(nActive_array_plot_sims)+2)
cmap = cmap(range(np.size(nActive_array_plot_sims)+2))
cmap = cmap[1:,:]

for ind, ind_nActivePlot in enumerate(nActive_array_plot_sims):
        
    prob_nActive = prob_nActive_clusters_E[ind_nActivePlot, :].copy()
    prob_nActive[prob_nActive < min_probActive] = np.nan
    prob_nActive[prob_nActive >= min_probActive] = 1

    x = sweep_param_array
    y = activeRate_XActiveClusters_E[ind_nActivePlot, :]
    yerr = activeRate_XActiveClusters_E_error[ind_nActivePlot, :]

    ax.fill_between(x, y-yerr, y+yerr, color=cmap[ind_nActivePlot, :], alpha=0.3)
    
    if ind_nActivePlot in legend_vals:
        ax.plot( x, y, '-o', color=cmap[ind_nActivePlot, :], linewidth=1, label=('nActive = %d' % ind_nActivePlot) )
    else:
        ax.plot( x, y, '-o', color=cmap[ind_nActivePlot, :], linewidth=1)
    
x = sweep_param_array    
y = popAvg_rate_E
yerr = popAvg_rate_E_error
ax.fill_between(x, y-yerr, y+yerr, color='gray', alpha=0.3)
ax.plot(x, y, color='gray')

#plt.title(r'$J^{+}_{\mathrm{EE}} = %0.3f$' % Jee_plus_sims)
plt.xlabel(r'arousal modulation $\Delta s[\mathrm{I_{EO}}]$')
plt.ylabel('active E rate [spks/sec]')
plt.legend(fontsize=4, loc='upper right')
plt.savefig( ( (fig_path + fig_filename_sim + 'sims_activeErate_vs_%s_JeePlus%0.3f.pdf') % (sweep_param_name_sim, Jee_plus_sims)), transparent=True)


#%% probability of x active clusters vs perturbation


plt.rcParams.update({'font.size': 5})
fig = plt.figure(figsize=(5,4))
       
for i in range(0, len(sweep_param_array)):

    x = np.arange(0, nClu+1)
    y = prob_nActive_clusters_E[:, i]
    yerr = prob_nActive_clusters_E_error[:, i]
    plt.fill_between(x, y-yerr, y+yerr, alpha=0.3)
    plt.plot(x, y, '-o', linewidth=1, label=r'$\Delta s[\mathrm{I_{EO}}] = %0.3f$' % sweep_param_array[i])

#plt.title(r'$J^{+}_{\mathrm{EE}} = %0.3f$' % Jee_plus_sims)
plt.xlabel('# active clusters')
plt.ylabel('probability')
plt.legend(fontsize=4, loc='upper right')
plt.savefig( ( (fig_path + fig_filename_sim + 'sims_prob_XactiveClusters_vs_%s_JeePlus%0.3f.pdf') % (sweep_param_name_sim, Jee_plus_sims)), transparent=True)

