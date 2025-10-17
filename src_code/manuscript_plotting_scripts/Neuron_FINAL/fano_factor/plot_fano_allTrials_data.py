
'''
This script generates
    FigS7D
'''


#%% basic imports

import sys
from scipy.io import loadmat
from scipy.io import savemat
import numpy as np
import scipy.stats
import os


#%% import global settings file
sys.path.append('../../../')
import global_settings

#%% plotting
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

#%% load functions

funcpath0 = global_settings.path_to_src_code + 'data_analysis/fanofactor_vs_pupil/'
funcpath1 = global_settings.path_to_src_code + 'data_analysis/'
funcpath2 = global_settings.path_to_src_code + 'functions/'

sys.path.append(funcpath0)        
import fcn_plot_fanofactor

sys.path.append(funcpath1)        
from fcn_SuData_analysis import fcn_significant_preStim_vs_postStim

sys.path.append(funcpath2)
from fcn_statistics import fcn_Wilcoxon


#%% settings

# paths
psth_path = global_settings.path_to_data_analysis_output + 'psth_allTrials/'
data_path = global_settings.path_to_data_analysis_output + 'fanofactor_pupil/'
fig_path = global_settings.path_to_manuscript_figs_final + 'fanofactor_allTrials_data/'

# filenames
fano_filename_raw = ('spont_evoked_fanofactor_all_pupilPercentile_raw')

# for loading data
pupil_lag = 0.
window_length = 100e-3
window_length_psth = 100e-3
avgRate_thresh = 1

# signficance of cell responses
sig_level = 0.05


# number of tones
nFreq = 5

# evoked period
evoked_window = [0, 150e-3]

# time point at which to evaluate FF evoked
t_eval_FFevoked = 'min_allStim'

# sessions to run
sessions_to_run = ['LA3_session3', \
                       'LA8_session1', 'LA8_session2', \
                       'LA9_session1', \
                       'LA9_session3', 'LA9_session4', 'LA9_session5', \
                       'LA11_session1', \
                       'LA11_session2', \
                       'LA11_session3', 'LA11_session4', \
                       'LA12_session1', \
                      'LA12_session2', \
                       'LA12_session3', \
                        'LA12_session4'
                      ]

# figure IDs
fig1ID = 'FigS7D_T'
fig2ID = 'FigS7D_B'

#savename
savename = 'fanofactor_allTrials_data'

#%% setup based on specified parameters

### make output directory
if os.path.isdir(fig_path) == False:
    os.makedirs(fig_path)

#%% initialize 


# fano factor
freqAvg_fanofactor_sigCells_evoked  = np.zeros((len(sessions_to_run)), dtype='object')
cellAvg_freqAvg_fanofactor_sigCells_evoked  = np.zeros((len(sessions_to_run)))
cellSem_freqAvg_fanofactor_sigCells_evoked  = np.zeros((len(sessions_to_run)))

freqAvg_fanofactor_sigCells_diff  = np.zeros((len(sessions_to_run)), dtype='object')

fanofactor_sigCells_spont  = np.zeros((len(sessions_to_run)), dtype='object')
cellAvg_fanofactor_sigCells_spont  = np.zeros((len(sessions_to_run)))
cellSem_fanofactor_sigCells_spont  = np.zeros((len(sessions_to_run)))


#%% load the data
for count, session_name in enumerate(sessions_to_run):
        
    # load psth data
    load_filename = (psth_path + 'psth_allTrials_%s_windLength%0.3fs.mat' % (session_name, window_length_psth))        
    psth_data = loadmat(load_filename, simplify_cells=True)
    
    # compute significant cells for this session
    resp_pre_vs_post = fcn_significant_preStim_vs_postStim(psth_data, sig_level)
    all_sig_cells = resp_pre_vs_post['allSigCells']
    sig_cells_eachFreq = resp_pre_vs_post['sigCells']

    # load the fano data 
    load_filename = (data_path + '%s_%s_windLength%0.3fs_pupilLag%0.3fs_.mat' %  (fano_filename_raw, sessions_to_run[count], window_length, pupil_lag))        
    fano_data_raw = loadmat(load_filename, simplify_cells=True)
        
    # time window
    t_window = fano_data_raw['t_window'].copy()
    
    # time corresponding to stimulus onset
    ind_t_base = np.nonzero(t_window == 0)[0][0]
    
    # start and stop of evoked window
    evokedInds = np.nonzero( (t_window >= evoked_window[0]) & (t_window <= evoked_window[-1]) )[0]
        
    # cells that pass rate cut
    spont_spikeCount = fano_data_raw['spont_trialAvg_spikeCount'].copy()          
    cells_pass_rate_cut = np.nonzero( spont_spikeCount/window_length >= avgRate_thresh )[0]
   
    # significant cells who pass baseline rate cut
    _, sig_cells_eachFreq = fcn_plot_fanofactor.fcn_sigUnits_pass_rateCut(cells_pass_rate_cut, all_sig_cells, sig_cells_eachFreq, nFreq)
    
    # any significant cell
    all_sig_cells = np.array([])
    for iFreq in range(0, nFreq):
        all_sig_cells = np.append(all_sig_cells, sig_cells_eachFreq[iFreq])
    all_sig_cells = np.unique(all_sig_cells).astype(int)
    
    
    ### spontaneous fano factor
    spont_fano = fano_data_raw['spont_fanofactor'].copy()
    
    # number of cells 
    n_Cells = np.size(spont_fano,0)

    # non_nanCells
    non_sigCells = np.setdiff1d(np.arange(n_Cells), all_sig_cells)
    nanCells = non_sigCells.copy()

    
    ######## spontaneous ff ##################################################
    # keep track of problematic cells
    nanCells_spont = np.array([])
    fanofactor_sigCells_spont[count] = spont_fano.copy()
    nanCells_spont = np.append(nanCells_spont, np.nonzero(np.isnan(fanofactor_sigCells_spont[count]))[0])
    nanCells = np.append(nanCells, nanCells_spont)
    
        
    ######## diff ff ##################################################
    diff_fano = -fano_data_raw['diff_fanofactor'].copy()
    diff_fano = diff_fano[:, :, evokedInds].copy() 

    ######## evoked ff ##################################################
    
    evoked_fano = fano_data_raw['evoked_fanofactor'].copy()
    evoked_fano = evoked_fano[:, :, evokedInds].copy()  
    
    additional_nanCells = np.array([])

    freqAvg_fanofactor_sigCells_evoked[count], \
    freqAvg_fanofactor_sigCells_diff[count] = \
        fcn_plot_fanofactor.fcn_compute_fano_cellSubset_multipleStim(evoked_fano, diff_fano, sig_cells_eachFreq, t_eval_FFevoked)
        
    additional_nanCells = np.append(additional_nanCells, np.nonzero(np.isnan(freqAvg_fanofactor_sigCells_evoked[count]))[0])
    additional_nanCells = np.append(additional_nanCells, np.nonzero(np.isnan(freqAvg_fanofactor_sigCells_diff[count]))[0])

    nanCells = np.append(nanCells, additional_nanCells)
    nanCells = np.unique(nanCells).astype(int)

        
    #### compute average over cells
    freqAvg_fanofactor_sigCells_evoked[count][nanCells] = np.nan
    cellAvg_freqAvg_fanofactor_sigCells_evoked[count] = np.nanmean(freqAvg_fanofactor_sigCells_evoked[count])
    cellSem_freqAvg_fanofactor_sigCells_evoked[count] = scipy.stats.sem(freqAvg_fanofactor_sigCells_evoked[count], nan_policy='omit')


    ##### spontaneous FF
    fanofactor_sigCells_spont[count][nanCells] = np.nan
    cellAvg_fanofactor_sigCells_spont[count] = np.nanmean(fanofactor_sigCells_spont[count])
    cellSem_fanofactor_sigCells_spont[count] = scipy.stats.sem(fanofactor_sigCells_spont[count], nan_policy='omit')


#%% spont - evoked factor -- single cell level

# combine spont and evoked fano factor across sessions
spont_ff_allSessions = np.array([])
evoked_ff_allSessions = np.array([])

for indSession in range(0, len(sessions_to_run)):
    
    spont_ff_allSessions = np.append(spont_ff_allSessions, fanofactor_sigCells_spont[indSession])
    evoked_ff_allSessions = np.append(evoked_ff_allSessions, freqAvg_fanofactor_sigCells_evoked[indSession])

# run statistics
spont_vs_evoked_fanoStats_allSessions_raw = fcn_Wilcoxon(spont_ff_allSessions, evoked_ff_allSessions)
print('ALL SESSIONS')
print('spont: %0.5f +/- %0.5f' % (np.nanmean(spont_ff_allSessions), scipy.stats.sem(spont_ff_allSessions, nan_policy='omit')))
print('evoked: %0.5f +/- %0.5f' % (np.nanmean(evoked_ff_allSessions), scipy.stats.sem(evoked_ff_allSessions, nan_policy='omit')))
print(spont_vs_evoked_fanoStats_allSessions_raw)

print('*'*100)


#%% save parameters and results

params = {}
results = {}

params['window_length'] = window_length
params['window_length_psth'] = window_length_psth
params['avgRate_thresh'] = avgRate_thresh
params['sessions_to_run'] = sessions_to_run
params['psth_path'] = psth_path
params['data_path'] = data_path
params['fig_path'] = fig_path
params['sig_level'] = sig_level
params['evoked_window'] = evoked_window
params['fano_filename'] = fano_filename_raw
params['pupil_lag'] = pupil_lag
params['nFreq'] = nFreq
params['t_eval_FFevoked'] = t_eval_FFevoked
results['params'] = params
results['spont_vs_evoked_fanoStats_allSessions_raw'] = spont_vs_evoked_fanoStats_allSessions_raw

save_filename = (fig_path + savename + '_stats.mat')      
savemat(save_filename, results) 


#%% plotting

# data
x = spont_ff_allSessions.copy()
y = evoked_ff_allSessions.copy()

# limits
low_lim = np.nanmin(np.append(x,y))
high_lim = np.nanmax(np.append(x,y))


#%% for all sessions, plot evoked vs spont

plt.rcParams.update({'font.size': 7})
fig = plt.figure(figsize=(1.,1.))  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8])  

# the scatter plot:
ax.plot(x, y, 'o', alpha=0.3, color='k', markersize=0.5)
ax.plot(np.nanmean(x), np.nanmean(y), marker='d', color='r', markersize=1)
ax.plot( [low_lim-0.5, high_lim+0.5], [low_lim-0.5, high_lim+0.5], color='teal', linewidth=0.5 )

# labels
ax.axis('equal')
ax.set_xticks([0, 7.5])
ax.set_yticks([0, 7.5])

ax.set_xlabel('$FF_{spont}$')
ax.set_ylabel('$FF_{evoked}$')
plt.savefig(fig_path + '%s.pdf' % (fig1ID), bbox_inches='tight', pad_inches=0, transparent=True)

#%% for all sessions, plot evoked and spont bar plot

plt.rcParams.update({'font.size': 7})
fig = plt.figure(figsize=(1.,1.))  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 

ax.bar(0, np.nanmean(x), width=0.6, color='dimgrey')
ax.errorbar(0, np.nanmean(x), yerr=scipy.stats.sem(x, nan_policy='omit'), color='r')

ax.bar(1, np.nanmean(y), width=0.6, color='dimgrey')
ax.errorbar(1, np.nanmean(y), yerr=scipy.stats.sem(y,nan_policy='omit'), color='r')
ax.set_ylabel(r'$\langle FF \rangle$')
ax.set_xticks([0,1])
ax.set_xticklabels(['spont', 'evoked'])
plt.savefig(fig_path + '%s.pdf' % (fig2ID), bbox_inches='tight', pad_inches=0.01, transparent=True)

