

#%% BASIC IMPORTS

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import sys
from matplotlib import cm


sys.path.insert(0,'/home/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/scripts/functions/')  
from fcn_simulation_loading import fcn_set_sweepParam_string


#%% PATH TO DATA

simParams_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/test_stim_expSyn/simParams_mft/')
loadMFT_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/test_stim_expSyn/MFT_sweep_JeePlus_arousalSweep/')
loadANALYSIS_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/test_stim_expSyn/clusterRates_numActiveClusters/')
loadSIM_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/test_stim_expSyn/')
fig_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/Figures/test_stim_expSyn/clusterRates_MFT_sims_sweepArousal/')

#%% PARAMETERS FOR LOADING DATA

# simulation ID
simID = '051300002025_clu'  

# netowrk name
net_type = 'baseEIclu'

# stim shape
stim_shape = 'diff2exp'

# relative stimulation amplitude
stim_rel_amp = 0.05

# sweep param name
sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'

# number of swept parameters
n_sweepParams = 3

# min probability active
min_probActive = 0.1
    
# window length
windLength = 25e-3

# rate threshold
rateThresh = 0.

# gain based
gain_based = True

# nActive array
nActive_array_plot_sims = np.arange(1,10)

fig_path = fig_path + sweep_param_name + '/'



#%% PARAMETERS FOR LOADING MFT DATA

mftID = '051300002025_clu' 

# mft reduced
mft_reduced = True

# network name
net_type_mft = 'baseEIclu'

# stim shape
stim_shape_mft = 'diff2exp'

# relative stimulation amplitude
stim_rel_amp_mft = 0.05

# number of active clusters to plot rate for
# want to set to most likely # of active clusters in simulations
nActive_plot_mft = 3

# nActive array
nActive_array_plot_mft = np.arange(1,10)



#%% LOAD EXAMPLE MFT DATA

mft_fname =  ( ('%s_%s_%s_%0.3fpct_stimType_%s_stim_rel_amp%0.3f') % \
               ( simID, net_type, sweep_param_name, 0., stim_shape, stim_rel_amp) ) 
    
# mft filename
if mft_reduced == True:
    mft_filename = mft_fname + '_reducedMFT_noDisorder_sweepJeePlus.mat' 
        
else:
    sys.exit('have only run reduced MFT')
    
MFT_data = loadmat(loadMFT_path + mft_filename, simplify_cells=True)

n_paramVals_mft = MFT_data['sim_params']['n_paramVals_mft']
swept_params_dict_sims = MFT_data['sim_params']['swept_params_dict_sims']

JeePlus_backSweep_results = MFT_data['JeePlus_backSweep_results']
JeePlus_forSweep_results = MFT_data['JeePlus_forSweep_results']

mft_params = MFT_data['mft_params']
n_activeClusters_sweep = mft_params['n_active_clusters_sweep']
JplusEE_back = JeePlus_backSweep_results['JplusEE_back']
JplusEE_for = JeePlus_forSweep_results['JplusEE_for']

n_nActive_clusters = np.size(n_activeClusters_sweep)
n_JplusEE_back = np.size(JplusEE_back)
n_JplusEE_for = np.size(JplusEE_for)

ind_param_plotMFT = np.array([0, int(n_paramVals_mft/2), -1])

arousal_level_mft = MFT_data['sim_params']['arousal_levels']*100


#%% LOAD EXAMPLE SIMULATION DATA

sweep_param_str_val = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict_sims, 0)

sim_fname =  ( ('%s_%s_sweep_%s_network%d_IC%d_stim%d_stimType_%s_stim_rel_amp%0.3f') % \
               ( simID, net_type, sweep_param_str_val, 0, 0, 0, stim_shape, stim_rel_amp) )
    
# simulation filename
sim_filename = sim_fname + '_simulationData.mat'
    
SIM_data = loadmat(loadSIM_path + sim_filename, simplify_cells=True)
sim_params = SIM_data['sim_params']
nClu = sim_params['p']
Jee_plus_sims = sim_params['JplusEE']


arousal_level = sim_params['arousal_levels']*100

#%% LOAD EXAMPLE ANALYSIS DATA

sweep_param_str_val = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict_sims, 0)

analysis_fname =  ( ('%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f') % \
                  ( simID, net_type, sweep_param_str_val, stim_shape, stim_rel_amp) )    
    
 
if gain_based:
    analysis_filename = analysis_fname + '__clusterRates_numActiveClusters_gainBased.mat'
else:
    analysis_filename = analysis_fname + '__clusterRates_numActiveClusters.mat'


ANALYSIS_data = loadmat(loadANALYSIS_path + analysis_filename, simplify_cells=True)
analysis_params = ANALYSIS_data['parameters']
nNets = analysis_params['nNets']
rate_thresh_array = analysis_params['rate_thresh']

print(nNets)

    
#%% WINDOW LENGTH AND RATE THRESHOLD TO PLOT

indThresh_plot = np.nonzero(rate_thresh_array==rateThresh)[0][0]

rateThresh = rate_thresh_array[indThresh_plot]


#%% FIGURE FILENAMES

fig_fname =  ( ('%s_%s_arousal%0.3fpct') % ( simID, net_type, 0.) ) 

fig_filename_sim = ((fig_fname + '_rateThresh%0.1fHz_windowStd%0.3fs_') % (rateThresh, windLength))
fig_filename_mft = (( fig_fname + '_rateThresh%0.1fHz_windowStd%0.3fs_') % (rateThresh, windLength))

#%% SIMULATIONS

n_paramVals_sweep = np.size(swept_params_dict_sims['param_vals1'])

# initialize arrays for quantities that we want to plot
activeRate_XActiveClusters_E = np.zeros((nClu+1, n_paramVals_sweep))
activeRate_XActiveClusters_E_error = np.zeros((nClu+1, n_paramVals_sweep))

# initialize arrays for quantities that we want to plot
inactiveRate_XActiveClusters_E = np.zeros((nClu+1, n_paramVals_sweep))
inactiveRate_XActiveClusters_E_error = np.zeros((nClu+1, n_paramVals_sweep))


# initialize arrays for quantities that we want to plot
activeRate_XActiveClusters_E_eachNet = np.zeros((nNets, nClu+1, n_paramVals_sweep))

# initialize arrays for quantities that we want to plot
inactiveRate_XActiveClusters_E_eachNet = np.zeros((nNets, nClu+1, n_paramVals_sweep))

# probability of n active clusters
prob_nActive_clusters_E_eachNet = np.zeros((nNets, nClu+1, n_paramVals_sweep))


prob_nActive_clusters_E = np.zeros((nClu+1, n_paramVals_sweep))
prob_nActive_clusters_E_error = np.zeros((nClu+1, n_paramVals_sweep))

popAvg_rate_E = np.zeros((n_paramVals_sweep))
popAvg_rate_E_error = np.zeros((n_paramVals_sweep))

popAvg_activeCluster_rate_e = np.zeros((n_paramVals_sweep))
popAvg_activeCluster_rate_e_error = np.zeros((n_paramVals_sweep))

popAvg_inactiveCluster_rate_e = np.zeros((n_paramVals_sweep))
popAvg_inactiveCluster_rate_e_error = np.zeros((n_paramVals_sweep))


# loop over perturbation
for ind_sweep_param in range(0, n_paramVals_sweep):
    

    sweep_param_str_val = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict_sims, ind_sweep_param)
      
        
    # analysis filename
    if gain_based:
        analysis_filename =  ( ('%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f__clusterRates_numActiveClusters_gainBased.mat') % \
                        ( simID, net_type, sweep_param_str_val, stim_shape, stim_rel_amp) )    
    else:
        analysis_filename =  ( ('%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f__clusterRates_numActiveClusters.mat') % \
                        ( simID, net_type, sweep_param_str_val, stim_shape, stim_rel_amp) )    

        
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
        
        if nNets == 1:
            activeRate_XActiveClusters_E_eachNet[indNet, :, ind_sweep_param] = ANALYSIS_data['trialAvg_avgRate_active_XActiveClusters_E'][:, indThresh_plot].copy()
            inactiveRate_XActiveClusters_E_eachNet[indNet, :, ind_sweep_param] = ANALYSIS_data['trialAvg_avgRate_inactive_XActiveClusters_E'][:, indThresh_plot].copy()
            prob_nActive_clusters_E_eachNet[indNet, :, ind_sweep_param] = ANALYSIS_data['trialAvg_prob_nActive_clusters_E'][:, indThresh_plot].copy()
            
        else:
            activeRate_XActiveClusters_E_eachNet[indNet, :, ind_sweep_param] = ANALYSIS_data['trialAvg_avgRate_active_XActiveClusters_E'][indNet, :, indThresh_plot].copy()
            inactiveRate_XActiveClusters_E_eachNet[indNet, :, ind_sweep_param] = ANALYSIS_data['trialAvg_avgRate_inactive_XActiveClusters_E'][indNet, :, indThresh_plot].copy()
            prob_nActive_clusters_E_eachNet[indNet, :, ind_sweep_param] = ANALYSIS_data['trialAvg_prob_nActive_clusters_E'][indNet, :, indThresh_plot].copy()


# most likely # active clusters at each perturbation
mostLikely_nActive_clusters_E = np.argmax(prob_nActive_clusters_E,0)


#%% MFT

# initialize
activeRate_XActiveClusters_E_mftBack = np.zeros((n_JplusEE_back, n_nActive_clusters, n_paramVals_mft))
activeRate_XActiveClusters_E_mftFor = np.zeros((n_JplusEE_for, n_nActive_clusters, n_paramVals_mft))

inactiveRate_XActiveClusters_E_mftBack = np.zeros((n_JplusEE_back, n_nActive_clusters, n_paramVals_mft))
inactiveRate_XActiveClusters_E_mftFor = np.zeros((n_JplusEE_for, n_nActive_clusters, n_paramVals_mft))

# loop over perturbation
for ind_sweep_param in range(0, n_paramVals_mft):
         
    # arousal level
    arousal_level_mft_ind = arousal_level_mft[ind_sweep_param]
    #arousal_level_mft_ind = ind_sweep_param/(n_paramVals_mft-1) 

        
    # analysis filename
    mft_filename =  ( ('%s_%s_%s_%0.3fpct_stimType_%s_stim_rel_amp%0.3f_reducedMFT_noDisorder_sweepJeePlus.mat') % \
                   ( simID, net_type, sweep_param_name, arousal_level_mft_ind, stim_shape, stim_rel_amp) )    


    # load the data
    MFT_data = loadmat(loadMFT_path + mft_filename, simplify_cells=True)

    # backwards and forwards sweep
    JeePlus_backSweep_results = MFT_data['JeePlus_backSweep_results']
    JeePlus_forSweep_results = MFT_data['JeePlus_forSweep_results']
    
    # unpack 
    nu_bar_e_back = JeePlus_backSweep_results['nu_e_backSweep']
    nu_bar_e_for = JeePlus_forSweep_results['nu_e_forSweep']

    # backwards
    activeRate_XActiveClusters_E_mftBack[:, :, ind_sweep_param] = nu_bar_e_back[0, :, :]
    inactiveRate_XActiveClusters_E_mftBack[:, :, ind_sweep_param] = nu_bar_e_back[1, :, :]
    
    # forwards
    activeRate_XActiveClusters_E_mftFor[:, :, ind_sweep_param] = nu_bar_e_for[0, :, :]
    inactiveRate_XActiveClusters_E_mftFor[:, :, ind_sweep_param] = nu_bar_e_for[1, :, :]


#%% FIND VALUE OF J AT WHICH BASELINE SIMULATIONS AND MFT BEST MATCH

# most likely # active clusters baseline
mostLikely_nActive_base = int(np.argmax(prob_nActive_clusters_E[:, 0]))
indMFT_mostLikely_nActive_base = np.nonzero(n_activeClusters_sweep == mostLikely_nActive_base)[0]
activeRate_mostLikely_nActive_base = activeRate_XActiveClusters_E[mostLikely_nActive_base, 0]

# mft J+ with closest match
activeRate_mft_base = activeRate_XActiveClusters_E_mftBack[:, indMFT_mostLikely_nActive_base, 0].copy()
mft_indJeePlus_back_simMatch = np.argmin(np.abs(activeRate_mostLikely_nActive_base - activeRate_mft_base))
mft_JeePlus_back_simMatch = JplusEE_back[mft_indJeePlus_back_simMatch]
print(mft_JeePlus_back_simMatch)

JeePlus_mft_plot = mft_JeePlus_back_simMatch

#%% PLOTTING


#%% plot mft and simulation rates for baseline arousal condition


ind_JeePlus_plot = np.argmin( np.abs(JplusEE_back - JeePlus_mft_plot) )

plt.rcParams.update({'font.size': 10})
fig = plt.figure(figsize=(5, 4))
ax = fig.add_axes([0.3, 0.3, 0.6, 0.6]) 


for ind, nActive in enumerate(nActive_array_plot_mft):
    
    ind_nActivePlot = np.argmin( np.abs(n_activeClusters_sweep - nActive))
    
    x = nActive
    y = activeRate_XActiveClusters_E_mftBack[ind_JeePlus_plot, ind_nActivePlot, 0]
    if ind == 0:
        label = 'mft'
    else:
        label = ''
    ax.plot( x, y, 'o', color='gray', label=label)      
        

for ind, ind_nActivePlot in enumerate(nActive_array_plot_sims):
        

    x = ind_nActivePlot
    y = activeRate_XActiveClusters_E[ind_nActivePlot, 0]
    if ind == 0:
        label = 'sims'
    else:
        label = ''
    ax.plot( x, y, 'o', color='r', label=label)


ax.set_xlabel(r'# active clusters $')
ax.set_ylabel('active E cluster rate [spks/sec]')
plt.legend(fontsize=6, loc='upper right')
plt.savefig( ( (fig_path + fig_filename_mft + 'mft_sims_activeErate_vs_numActiveClusters_JeePlus%0.3f_baseArousal.pdf') % (JeePlus_mft_plot)), transparent=True)



#%% plot in/active cluster rate vs JeePlus for n=X; three different values of perturbation


ind_nActivePlot = np.nonzero(n_activeClusters_sweep == nActive_plot_mft)[0]

plt.rcParams.update({'font.size': 10})
fig = plt.figure(figsize=(5, 4))
ax = fig.add_axes([0.3, 0.3, 0.6, 0.6]) 
       
cmap_active = cm.get_cmap('Greens', 4)
cmap_active = cmap_active(range(4))
cmap_active = cmap_active[1:,:]

cmap_inactive = cm.get_cmap('Purples', 4)
cmap_inactive = cmap_inactive(range(4))
cmap_inactive = cmap_inactive[1:,:]


for i in range(0,3):
    
    ind_param = ind_param_plotMFT[i]
    
    x = JplusEE_back
    y = activeRate_XActiveClusters_E_mftBack[:, ind_nActivePlot, ind_param]

    if i == 0:
        label_active = 'active'
        label_inactive = 'inactive'
    else:
        label_active = ''
        label_inactive = ''
        
    ax.plot(x, y, color=cmap_active[i,:], linewidth=2, label = label_active)
    
    
    x = JplusEE_back
    y = inactiveRate_XActiveClusters_E_mftBack[:, ind_nActivePlot, ind_param]

    ax.plot(x, y, color=cmap_inactive[i,:], linewidth=2, label = label_inactive)
    

x = np.array([JeePlus_mft_plot, JeePlus_mft_plot])
y = np.array([0, 125])
ax.plot(x, y, color='gray', linewidth=2)
ax.set_ylim([0, 125])
ax.set_xlabel('synpatic potentiation $J_{EE}^{+}$')
ax.set_ylabel('E cluster rate [spks/sec]')
plt.legend(fontsize=6, loc='upper left')
plt.savefig(fig_path + fig_filename_mft + 'mft_activeErate_vs_JeePlus_nActive%d.pdf' % (nActive_plot_mft), transparent=True)



#%% plot in/active cluster rate vs perturbation for fixed Jee+ simulations; avg across # active clusters



plt.rcParams.update({'font.size': 10})
fig = plt.figure(figsize=(5, 4))
ax = fig.add_axes([0.3, 0.3, 0.6, 0.6]) 
       

x = arousal_level
y = popAvg_activeCluster_rate_e
yerr = popAvg_activeCluster_rate_e_error
ax.fill_between(x, y-yerr, y+yerr, color='mediumseagreen', alpha=0.3)
ax.plot( x, y, '-o', linewidth=1, color='mediumseagreen', label='active')


x = arousal_level
y = popAvg_inactiveCluster_rate_e
yerr = popAvg_inactiveCluster_rate_e_error
ax.fill_between(x, y-yerr, y+yerr, color='rebeccapurple', alpha=0.3)
ax.plot( x, y, '-o', color='rebeccapurple', linewidth=1, label='inactive')
    

x = arousal_level    
y = popAvg_rate_E
yerr = popAvg_rate_E_error
ax.fill_between(x, y-yerr, y+yerr, color='gray', alpha=0.3)
ax.plot(x, y, color='gray')


ax.set_xlabel('arousal [%]')
ax.set_ylabel('E cluster rate [spks/sec]')
ax.legend(fontsize=6, loc='upper right')
plt.savefig( ( (fig_path + fig_filename_sim + 'sims_active_inactive_Erate_vs_%s_JeePlus%0.3f_avg_nActive.pdf') % (sweep_param_name, Jee_plus_sims)), transparent=True)


#%% plot in/active cluster rate for fixed Jee+ simulations; most likely # active clusters

plt.rcParams.update({'font.size': 10})
fig = plt.figure(figsize=(5, 4))
ax = fig.add_axes([0.3, 0.3, 0.6, 0.6]) 
       

x = arousal_level
y = np.zeros(len(x))
yerr = np.zeros(len(x))

for i in range(0,len(y)):
    
    y[i] = activeRate_XActiveClusters_E[mostLikely_nActive_clusters_E[i], i]
    yerr[i] = activeRate_XActiveClusters_E_error[mostLikely_nActive_clusters_E[i], i]
    
    print(mostLikely_nActive_clusters_E[i])
    print(yerr[i])

ax.fill_between(x, y-yerr, y+yerr, color='mediumseagreen', alpha=0.3)
ax.plot( x, y, '-o', linewidth=1, color='mediumseagreen', label='active')


x = arousal_level
y = np.zeros(len(x))
yerr = np.zeros(len(x))

for i in range(0,len(y)):
    
    y[i] = inactiveRate_XActiveClusters_E[mostLikely_nActive_clusters_E[i], i]
    yerr[i] = inactiveRate_XActiveClusters_E_error[mostLikely_nActive_clusters_E[i], i]

ax.fill_between(x, y-yerr, y+yerr, color='rebeccapurple', alpha=0.3)
ax.plot( x, y, '-o', color='rebeccapurple', linewidth=1, label='inactive')
    


ax.set_xlabel('arousal [%]')
ax.set_ylabel('E cluster rate [spks/sec]')
ax.legend(fontsize=6, loc='upper right')
plt.savefig( ( (fig_path + fig_filename_sim + 'sims_active_inactive_Erate_vs_%s_JeePlus%0.3f_mostLikely_nActive.pdf') % (sweep_param_name, Jee_plus_sims)), transparent=True)

#%% plot in/active cluster rate for fixed Jee+ simulations; average across nActive with prob>x

plt.rcParams.update({'font.size': 10})
fig = plt.figure(figsize=(5, 4))
ax = fig.add_axes([0.3, 0.3, 0.6, 0.6]) 
       
### active
x = arousal_level
y_eachNet = np.zeros((nNets, len(x)))


for indNet in range(0, nNets):
    for ind_sweptParam in range(0, len(x)):
    
        avg_nActive = np.nonzero(prob_nActive_clusters_E_eachNet[indNet, :, ind_sweptParam] > min_probActive)[0]
        y_eachNet[indNet, ind_sweptParam] = np.nanmean(activeRate_XActiveClusters_E_eachNet[indNet, avg_nActive, ind_sweptParam])

y = np.mean(y_eachNet, 0)
yerr = np.std(y_eachNet, 0)/np.sqrt(nNets)

ax.fill_between(x, y-yerr, y+yerr, color='mediumseagreen', alpha=0.3)
ax.plot( x, y, '-o', linewidth=1, color='mediumseagreen', label='active')



### inactive
x = arousal_level
y_eachNet = np.zeros((nNets, len(x)))

for indNet in range(0, nNets):
    for ind_sweptParam in range(0, len(x)):
    
        avg_nActive = np.nonzero(prob_nActive_clusters_E_eachNet[indNet, :, ind_sweptParam] > min_probActive)[0]
        y_eachNet[indNet, ind_sweptParam] = np.nanmean(inactiveRate_XActiveClusters_E_eachNet[indNet, avg_nActive, ind_sweptParam])

y = np.mean(y_eachNet, 0)
yerr = np.std(y_eachNet, 0)/np.sqrt(nNets)

ax.fill_between(x, y-yerr, y+yerr, color='rebeccapurple', alpha=0.3)
ax.plot( x, y, '-o', linewidth=1, color='rebeccapurple', label='active') 



### settings
ax.set_ylim([0, 50])
ax.set_title(r'$J^{+}_{\mathrm{EE}} = %0.3f$' % Jee_plus_sims)
ax.set_xlabel('arousal [%]')
ax.set_ylabel('E cluster rate [spks/sec]')
plt.legend(fontsize=6, loc='upper right')
plt.savefig( ( (fig_path + fig_filename_sim + 'sims_active_inactive_Erate_vs_%s_JeePlus%0.3f_avg_mostLikely_nActive_probThresh%0.2f.pdf') % (sweep_param_name, Jee_plus_sims, min_probActive)), transparent=True)


#%% plot in/active cluster rate vs perturbation for fixed Jee+ mft; average over n active



ind_JeePlus_plot = np.argmin( np.abs(JplusEE_back - JeePlus_mft_plot) )

mft_nActive_plot_array = np.array([2,3,4])
inds_nActivePlot = np.zeros( len(mft_nActive_plot_array) )
for count, nActive in enumerate(mft_nActive_plot_array):
    inds_nActivePlot[count] = np.nonzero(n_activeClusters_sweep == mft_nActive_plot_array[count])[0][0]

inds_nActivePlot = inds_nActivePlot.astype(int)

plt.rcParams.update({'font.size': 10})
fig = plt.figure(figsize=(5, 4))
ax = fig.add_axes([0.3, 0.3, 0.6, 0.6]) 
    
x = arousal_level_mft
y = np.mean(activeRate_XActiveClusters_E_mftBack[ind_JeePlus_plot, inds_nActivePlot, :], axis=0)
ax.plot( x, y, '-o', linewidth=1, color='mediumseagreen', label='active')       

x = arousal_level_mft
y = np.mean(inactiveRate_XActiveClusters_E_mftBack[ind_JeePlus_plot, inds_nActivePlot, :], axis=0)
ax.plot( x, y, '-o', linewidth=1, color='rebeccapurple', label='inactive')   


ax.set_xlabel('arousal [%]')
ax.set_ylabel('E cluster rate [spks/sec]')
ax.legend(fontsize=6, loc='upper right')
plt.savefig( ( (fig_path + fig_filename_mft + 'mft_active_inactive_Erate_vs_%s_JeePlus%0.3f_avg_nActive.pdf') % (sweep_param_name, JeePlus_mft_plot)), transparent=True)


#%% plot in/active cluster rate vs perturbation for fixed Jee+ mft; most likely n active from simulations


ind_JeePlus_plot = np.argmin( np.abs(JplusEE_back - JeePlus_mft_plot) )

ind_nActivePlot = np.zeros( len(arousal_level) )
for i in range(0, len(arousal_level)):
    ind_nActivePlot[i] = np.nonzero(n_activeClusters_sweep == mostLikely_nActive_clusters_E[i])[0][0]

ind_nActivePlot = ind_nActivePlot.astype(int)

plt.rcParams.update({'font.size': 10})
fig = plt.figure(figsize=(5, 4))
ax = fig.add_axes([0.3, 0.3, 0.6, 0.6]) 
    
x = arousal_level
y = np.zeros(len(x))
for i in range(0,len(x)):
    indMFT = np.argmin(np.abs(arousal_level_mft - arousal_level[i]))
    y[i] = activeRate_XActiveClusters_E_mftBack[ind_JeePlus_plot, ind_nActivePlot[i], indMFT]
    
ax.plot( x, y, '-o', linewidth=1, color='mediumseagreen', label='active')       

x = arousal_level
y = np.zeros(len(x))
for i in range(0,len(x)):
    indMFT = np.argmin(np.abs(arousal_level_mft - arousal_level[i]))
    y[i] = inactiveRate_XActiveClusters_E_mftBack[ind_JeePlus_plot, ind_nActivePlot[i], indMFT]
    
ax.plot( x, y, '-o', linewidth=1, color='rebeccapurple', label='inactive')   


ax.set_xlabel(r'arousal [%]')
ax.set_ylabel('E cluster rate [spks/sec]')
ax.legend(fontsize=6, loc='upper right')
plt.savefig( ( (fig_path + fig_filename_mft + 'mft_active_inactive_Erate_vs_%s_JeePlus%0.3f_mostLikely_nActive.pdf') % (sweep_param_name, JeePlus_mft_plot)), transparent=True)


#%% plot in/active cluster rate vs perturbation for fixed Jee+ mft; most likely n active from simulations

ind_JeePlus_plot = np.argmin( np.abs(JplusEE_back - JeePlus_mft_plot) )
prob_thresh = 0.1

x = arousal_level
yactive = np.zeros(len(x))
yinactive = np.zeros(len(x))

for i in range(0, len(x)):
    avg_nActive_sims = np.nonzero(prob_nActive_clusters_E[:, i] > prob_thresh)[0]
    inds_nActive_plot = np.array([])
    for j in range(0, len(avg_nActive_sims)):
        inds_nActivePlot = np.append(inds_nActivePlot, np.nonzero(n_activeClusters_sweep ==avg_nActive_sims[j])[0])
    
    indMFT = np.argmin(np.abs(arousal_level_mft - arousal_level[i]))
    yactive[i] = np.mean(activeRate_XActiveClusters_E_mftBack[ind_JeePlus_plot, inds_nActivePlot, indMFT])
    yinactive[i] = np.mean(inactiveRate_XActiveClusters_E_mftBack[ind_JeePlus_plot, inds_nActivePlot, indMFT])

plt.rcParams.update({'font.size': 10})
fig = plt.figure(figsize=(5, 4))
ax = fig.add_axes([0.3, 0.3, 0.6, 0.6]) 

ax.plot( x, yactive, '-o', linewidth=1, color='mediumseagreen', label='active')       
ax.plot( x, yinactive, '-o', linewidth=1, color='rebeccapurple', label='inactive')   
ax.set_ylim([0, 55]) 

ax.set_xlabel(r'arousal [%]')
ax.set_ylabel('E cluster rate [spks/sec]')
ax.legend(fontsize=6, loc='upper right')
plt.savefig( ( (fig_path + fig_filename_mft + 'mft_active_inactive_Erate_vs_%s_JeePlus%0.3f_avg_mostLikely_nActive_probThresh%0.2f.pdf') % (sweep_param_name, JeePlus_mft_plot, prob_thresh)), transparent=True)


#%% plot active cluster rate vs perturbation for fixed Jee+ mft; different curves for different # active clusters

ind_JeePlus_plot = np.argmin( np.abs(JplusEE_back - JeePlus_mft_plot) )

plt.rcParams.update({'font.size': 10})
fig = plt.figure(figsize=(5, 4))
ax = fig.add_axes([0.3, 0.3, 0.6, 0.6]) 

legend_vals = nActive_array_plot_mft      

cmap = cm.get_cmap('magma', np.size(nActive_array_plot_sims)+2)
cmap = cmap(range(np.size(nActive_array_plot_sims)+2))
cmap = cmap[1:,:]

for ind, nActive in enumerate(nActive_array_plot_mft):
    
    ind_nActivePlot = np.argmin( np.abs(n_activeClusters_sweep - nActive))
    
    x = arousal_level_mft
    y = activeRate_XActiveClusters_E_mftBack[ind_JeePlus_plot, ind_nActivePlot, :]

    if nActive in legend_vals:
        ax.plot( x, y, '-o', color=cmap[nActive, :], linewidth=1, markersize=1, label=('nActive = %d' % nActive) )
    else:
        ax.plot( x, y, '-o', color=cmap[nActive, :], linewidth=1, markersize=1, )        
        
        
    x = arousal_level_mft
    y = inactiveRate_XActiveClusters_E_mftBack[ind_JeePlus_plot, ind_nActivePlot, :]

    if nActive in legend_vals:
        ax.plot( x, y, '-', color=cmap[nActive, :], linewidth=1 )
    else:
        ax.plot( x, y, '-', color=cmap[nActive, :], linewidth=1 )   


#plt.title(r'$J^{+}_{\mathrm{EE}} = %0.3f$' % JeePlus_mft_plot)
ax.set_xlabel(r'arousal [%]')
ax.set_ylabel('active E cluster rate [spks/sec]')
ax.legend(fontsize=6, loc='upper right')
plt.savefig( ( (fig_path + fig_filename_mft + 'mft_activeErate_vs_%s_JeePlus%0.3f.pdf') % (sweep_param_name, JeePlus_mft_plot)), transparent=True)



#%% plot active cluster rate vs perturbation for fixed Jee+ simulations; different curves for different # active clusters

plt.rcParams.update({'font.size': 10})
fig = plt.figure(figsize=(5, 4))
ax = fig.add_axes([0.3, 0.3, 0.6, 0.6]) 
       
legend_vals = nActive_array_plot_sims

cmap = cm.get_cmap('magma', np.size(nActive_array_plot_sims)+2)
cmap = cmap(range(np.size(nActive_array_plot_sims)+2))
cmap = cmap[1:,:]

for ind, ind_nActivePlot in enumerate(nActive_array_plot_sims):
        
    prob_nActive = prob_nActive_clusters_E[ind_nActivePlot, :].copy()
    prob_nActive[prob_nActive < min_probActive] = np.nan
    prob_nActive[prob_nActive >= min_probActive] = 1

    x = arousal_level
    y = activeRate_XActiveClusters_E[ind_nActivePlot, :]
    yerr = activeRate_XActiveClusters_E_error[ind_nActivePlot, :]

    ax.fill_between(x, y-yerr, y+yerr, color=cmap[ind_nActivePlot, :], alpha=0.3)
    
    if ind_nActivePlot in legend_vals:
        ax.plot( x, y, '-o', color=cmap[ind_nActivePlot, :], linewidth=1, label=('nActive = %d' % ind_nActivePlot) )
    else:
        ax.plot( x, y, '-o', color=cmap[ind_nActivePlot, :], linewidth=1)
    
x = arousal_level    
y = popAvg_rate_E
yerr = popAvg_rate_E_error
ax.fill_between(x, y-yerr, y+yerr, color='gray', alpha=0.3)
ax.plot(x, y, color='gray')

#plt.title(r'$J^{+}_{\mathrm{EE}} = %0.3f$' % Jee_plus_sims)
ax.set_xlabel(r'arousal [%]')
ax.set_ylabel('active E cluster rate [spks/sec]')
ax.legend(fontsize=6, loc='upper right')
plt.savefig( ( (fig_path + fig_filename_sim + 'sims_activeErate_vs_%s_JeePlus%0.3f.pdf') % (sweep_param_name, Jee_plus_sims)), transparent=True)


#%% probability of x active clusters vs perturbation


plt.rcParams.update({'font.size': 10})
fig = plt.figure(figsize=(5, 4))
ax = fig.add_axes([0.3, 0.3, 0.6, 0.6]) 

cmap = cm.get_cmap('PuBu', n_paramVals_sweep+5)
cmap = cmap(range(n_paramVals_sweep+5))
cmap = cmap[5:,:]
       
for i in range(0, len(arousal_level)):

    x = np.arange(0, nClu+1)
    y = prob_nActive_clusters_E[:, i]
    yerr = prob_nActive_clusters_E_error[:, i]
    ax.fill_between(x, y-yerr, y+yerr, alpha=0.5, color=cmap[i,:])
    ax.plot(x, y, '-o', color=cmap[i,:], linewidth=1, label=('arousal = %d' % (arousal_level[i])))

ax.set_xticks(x[::2])
ax.set_xlabel('# active clusters')
ax.set_ylabel('probability')
ax.legend(fontsize=6, loc='upper right')
plt.savefig( ( (fig_path + fig_filename_sim + 'sims_prob_XactiveClusters_vs_%s_JeePlus%0.3f.pdf') % (sweep_param_name, Jee_plus_sims)), transparent=True)



#%% plot arousal for mft and sims

filename = ('%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f_simParams_mft_noDisorder.mat' % \
            ( simID, net_type, sweep_param_name, stim_shape, stim_rel_amp) )
    
simParams_mft = loadmat(simParams_path + filename, simplify_cells=True)   

arousal_sweep_dict_sims = simParams_mft['swept_params_dict_sims']
arousal_sweep_dict_mft = simParams_mft['swept_params_dict_mft']


plt.figure()
plt.plot(arousal_level,arousal_sweep_dict_sims['param_vals1'], '-o')
plt.plot(arousal_level_mft,arousal_sweep_dict_mft['param_vals1'], '-')

plt.figure()
plt.plot(arousal_level,arousal_sweep_dict_sims['param_vals2'], '-o')
plt.plot(arousal_level_mft,arousal_sweep_dict_mft['param_vals2']/2, '-')

plt.figure()
plt.plot(arousal_level,arousal_sweep_dict_sims['param_vals3'], '-o')
plt.plot(arousal_level_mft,arousal_sweep_dict_mft['param_vals3']/2, '-')