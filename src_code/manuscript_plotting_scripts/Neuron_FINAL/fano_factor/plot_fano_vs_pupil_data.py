
'''
This script generates different figure panels depending on the values of certain settings

    cellSelection = ''
    vary_windowLength = False
    window_length = 100e-3
        Fig8D-I    
        FigS7A-C
    
    cellSelection = ''
    vary_windowLength = True
    window_length = 50e-3, 100e-3, 200e-3
        FigS7E, FigS7F, FigS7G
    
    cellSelection = '_spkTemplate_soundResp_cellSelection1'
    vary_windowLength = False
    window_length = 100e-3
        FigS8G-L

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
from fcn_SuData_analysis import fcn_cellAvg_sessionAvg_quantity_vs_pupilSize_bins_alt
    
sys.path.append(funcpath2)
from fcn_statistics import fcn_Wilcoxon


#%% settings

# paths
psth_path = global_settings.path_to_data_analysis_output + 'psth_allTrials/'
data_path = global_settings.path_to_data_analysis_output + 'fanofactor_pupil/'
fig_path = global_settings.path_to_manuscript_figs_final + 'fanofactor_vs_pupil_data/'

# fano filename
fano_filename_raw = ('spont_evoked_fanofactor_pupilPercentile_raw')

# vary window length
vary_windowLength = True

# for loading in the data
pupil_lag = 0.
window_length = 200e-3
window_length_psth = 100e-3
avgRate_thresh = 1.

# signficance
sig_level = 0.05

# number of pupil bins used in analysis
nPupil_bins = 10

# min and max pupil thresh
lowPupil_thresh = 0.33
highPupil_thresh = 0.67
pupilSize_binStep = 0.1

# number of tones
nFreq = 5

# evoked period
evoked_window = [0, 150e-3]

# time point at which to evaulate evoked FF
t_eval_FFevoked = 'min_allStim' 

# cell selection
cellSelection = ''

# sessions to analyze
sessions_to_run = ['LA3_session3', \
                       'LA8_session1', \
                    'LA8_session2', \
                       'LA9_session1', \
                       'LA9_session3', \
                        'LA9_session4', \
                    'LA9_session5', \
                       'LA11_session1', \
                       'LA11_session2', \
                       'LA11_session3', \
                           'LA11_session4', \
                       'LA12_session1', \
                      'LA12_session2', \
                       'LA12_session3', \
                        'LA12_session4'
                      ]
    
# figure IDs

if ( (cellSelection == '') and (vary_windowLength == True) ):
    if window_length == 50e-3:
        fig0ID = 'FigS7E'
    if window_length == 100e-3:
        fig0ID = 'FigS7F'
    if window_length == 200e-3:
        fig0ID = 'FigS7G'
    savename = 'fanofactor_pupil_data_varyWindow_'

if ( (cellSelection == '') and (vary_windowLength == False) and (window_length == 100e-3)):
    fig1ID = 'Fig8D'
    fig2ID = 'Fig8F'
    fig3ID = 'Fig8H'
    fig4ID = 'FigS7A_T'
    fig5ID = 'FigS7A_B'
    fig6ID = 'FigS7B_T'    
    fig7ID = 'FigS7B_B'
    fig8ID = 'FigS7C_T'
    fig9ID = 'FigS7C_B' 
    fig10ID = 'Fig8E'
    fig11ID = 'Fig8G'
    fig12ID = 'Fig8I'
    savename = 'fanofactor_pupil_data_'
    
    
if ( (cellSelection == '_spkTemplate_soundResp_cellSelection1') and (vary_windowLength == False) and (window_length == 100e-3)):
    fig1ID = 'FigS8G'
    fig2ID = 'FigS8I'
    fig3ID = 'FigS8K' 
    fig10ID = 'FigS8H'
    fig11ID = 'FigS8J'
    fig12ID = 'FigS8L'
    savename = 'fanofactor_pupil_data_altCellSelction_'
    
   
#%% setup

# update figure path
if cellSelection == '':
    fig_path = fig_path + 'original_cellSelection/'
elif cellSelection == '_spkTemplate_soundResp_cellSelection1':
    fig_path = fig_path + 'spkTemplate_soundResp_cellSelection1/'
    

if vary_windowLength:
    fig_path = fig_path + 'vary_windL/'

# data name
data_name = '' + cellSelection

# figure name
figname = ('windL%0.3f_rateThresh%0.1fHz%s_' % (window_length, avgRate_thresh, data_name))

# checks
if ( (vary_windowLength==False) and (window_length!=100e-3) ):
    sys.exit('wrong parameters for paper figures')

# make output directory
if os.path.isdir(fig_path) == False:
    os.makedirs(fig_path)


#%% initialize 

# pupil bin centers
pupilBin_centers_raw = np.zeros((len(sessions_to_run), nPupil_bins))
nTrials_subsample = np.zeros((len(sessions_to_run)))

# fano factor
freqAvg_fanofactor_sigCells_spont  = np.ones((len(sessions_to_run), nPupil_bins), dtype='object')*np.nan
cellAvg_freqAvg_fanofactor_sigCells_spont  = np.ones((len(sessions_to_run), nPupil_bins))*np.nan
cellSem_freqAvg_fanofactor_sigCells_spont  = np.ones((len(sessions_to_run), nPupil_bins))*np.nan

freqAvg_fanofactor_sigCells_evoked  = np.ones((len(sessions_to_run), nPupil_bins), dtype='object')*np.nan
cellAvg_freqAvg_fanofactor_sigCells_evoked  = np.ones((len(sessions_to_run), nPupil_bins))*np.nan
cellSem_freqAvg_fanofactor_sigCells_evoked  = np.ones((len(sessions_to_run), nPupil_bins))*np.nan

freqAvg_fanofactor_sigCells_diff  = np.ones((len(sessions_to_run), nPupil_bins), dtype='object')*np.nan
cellAvg_freqAvg_fanofactor_sigCells_diff  = np.ones((len(sessions_to_run), nPupil_bins))*np.nan
cellSem_freqAvg_fanofactor_sigCells_diff  = np.ones((len(sessions_to_run), nPupil_bins))*np.nan

# good sessions
goodSessions = np.array([])
sum_cells = np.array([])


#%% load the data
for count, session_name in enumerate(sessions_to_run):
    
        
    # load psth data
    load_filename = (psth_path + 'psth_allTrials_%s_windLength%0.3fs%s.mat' % (session_name, window_length_psth, data_name))        
    psth_data = loadmat(load_filename, simplify_cells=True)
    
    # compute significant cells for this session
    resp_pre_vs_post = fcn_significant_preStim_vs_postStim(psth_data, sig_level)
    all_sig_cells = resp_pre_vs_post['allSigCells']
    sig_cells_eachFreq = resp_pre_vs_post['sigCells']

        
    # load the fano data 
    if vary_windowLength:
        load_filename = (data_path + '%s_%s_windLength%0.3fs_pupilLag%0.3fs_trialMatch_windL%s.mat' %  (fano_filename_raw, session_name, window_length, pupil_lag, data_name))        
    else:
        load_filename = (data_path + '%s_%s_windLength%0.3fs_pupilLag%0.3fs_%s.mat' %  (fano_filename_raw, session_name, window_length, pupil_lag, data_name))        
    fano_data_raw = loadmat(load_filename, simplify_cells=True)
        
    # pupil bin centers 
    pupilBin_centers_raw[count, :] = fano_data_raw['avg_pupilSize_spontTrials'].copy()

    ### determine if this is a good session
    if ( (np.min(pupilBin_centers_raw[count, :]) <= lowPupil_thresh) and (np.max(pupilBin_centers_raw[count, :]) >= highPupil_thresh)) :
        
        goodSessions = np.append(goodSessions, count)
    
    # number of trials
    nTrials_subsample[count] = fano_data_raw['nTrials_subsample']
    
    # time window
    t_window = fano_data_raw['t_window'].copy()
    
    # time corresponding to stimulus onset
    ind_t_base = np.nonzero(t_window == 0)[0][0]
    
    # start and stop of evoked window
    evokedInds = np.nonzero( (t_window >= evoked_window[0]) & (t_window <= evoked_window[-1]) )[0]
        
    # cells that pass rate cut
    min_spikeCount_allPupil = np.min(fano_data_raw['spont_trialAvg_spikeCount'], 1)
    min_rate_allPupil = min_spikeCount_allPupil/window_length
    cells_pass_rate_cut = np.nonzero( min_rate_allPupil >= avgRate_thresh )[0]
   
    # significant cells who pass baseline rate cut
    all_sig_cells, sig_cells_eachFreq = fcn_plot_fanofactor.fcn_sigUnits_pass_rateCut(cells_pass_rate_cut, all_sig_cells, sig_cells_eachFreq, nFreq)
    
    # any significant cell
    all_sig_cells = np.array([])
    for iFreq in range(0, nFreq):
        sig_cells_eachFreq[iFreq] = sig_cells_eachFreq[iFreq].astype(int)
        all_sig_cells = np.append(all_sig_cells, sig_cells_eachFreq[iFreq])
    all_sig_cells = np.unique(all_sig_cells).astype(int)
    
    # spont fano factor
    spont_fano = fano_data_raw['spont_fanofactor'].copy()
    
    # evoked fano factor
    evoked_fano = fano_data_raw['evoked_fanofactor'].copy()
    evoked_fano = evoked_fano[:, :, :, evokedInds].copy()  
    
    # diff fano factor
    diff_fano = -fano_data_raw['diff_fanofactor'].copy()
    diff_fano = diff_fano[:, :, :, evokedInds].copy() 

    # number of cells and pupil bins
    n_Cells = np.size(spont_fano,0)
    n_pupils = np.size(spont_fano, 1)    

    # non_nanCells
    non_sigCells = np.setdiff1d(np.arange(n_Cells), all_sig_cells)
    nanCells = non_sigCells.copy()
    
    ######## spontaneous ff ##################################################
    # loop over pupil bins and store ff
    # keep track of problematic cells
    nanCells_spont = np.array([])
    for indPupil in range(0, n_pupils):
    
        freqAvg_fanofactor_sigCells_spont[count, indPupil] = spont_fano[:, indPupil].copy()
        nanCells_spont = np.append(nanCells_spont, np.nonzero(np.isnan(freqAvg_fanofactor_sigCells_spont[count, indPupil]))[0])
        
    nanCells = np.append(nanCells, nanCells_spont)
        

    ######## evoked ff ##################################################
    additional_nanCells = np.array([])

    for indPupil in range(0, nPupil_bins):    
        
        evoked_fano_thisPupil = evoked_fano[:, :, indPupil, :].copy()
        diff_fano_thisPupil = diff_fano[:, :, indPupil, :].copy()
        
        freqAvg_fanofactor_sigCells_evoked[count, indPupil], \
        freqAvg_fanofactor_sigCells_diff[count, indPupil] = \
            fcn_plot_fanofactor.fcn_compute_fano_cellSubset_multipleStim(evoked_fano_thisPupil, diff_fano_thisPupil, sig_cells_eachFreq, t_eval_FFevoked)
            
        additional_nanCells = np.append(additional_nanCells, np.nonzero(np.isnan(freqAvg_fanofactor_sigCells_evoked[count, indPupil]))[0])
        additional_nanCells = np.append(additional_nanCells, np.nonzero(np.isnan(freqAvg_fanofactor_sigCells_diff[count, indPupil]))[0])

    nanCells = np.append(nanCells, additional_nanCells)
    nanCells = np.unique(nanCells).astype(int)

        
    #### compute average over cells
    for indPupil in range(0, n_pupils):
        freqAvg_fanofactor_sigCells_evoked[count, indPupil][nanCells] = np.nan
        cellAvg_freqAvg_fanofactor_sigCells_evoked[count, indPupil] = np.nanmean(freqAvg_fanofactor_sigCells_evoked[count, indPupil])
        cellSem_freqAvg_fanofactor_sigCells_evoked[count, indPupil] = scipy.stats.sem(freqAvg_fanofactor_sigCells_evoked[count, indPupil], nan_policy='omit')
        freqAvg_fanofactor_sigCells_diff[count, indPupil][nanCells] = np.nan
        cellAvg_freqAvg_fanofactor_sigCells_diff[count, indPupil] = np.nanmean(freqAvg_fanofactor_sigCells_diff[count, indPupil])
        cellSem_freqAvg_fanofactor_sigCells_diff[count, indPupil] = scipy.stats.sem(freqAvg_fanofactor_sigCells_diff[count, indPupil], nan_policy='omit')

    ##### spontaneous FF
    for indPupil in range(0, n_pupils):
        freqAvg_fanofactor_sigCells_spont[count, indPupil][nanCells] = np.nan
        cellAvg_freqAvg_fanofactor_sigCells_spont[count, indPupil] = np.nanmean(freqAvg_fanofactor_sigCells_spont[count, indPupil])
        cellSem_freqAvg_fanofactor_sigCells_spont[count, indPupil] = scipy.stats.sem(freqAvg_fanofactor_sigCells_spont[count, indPupil], nan_policy='omit')

    #### total number of cells
    sum_cells = np.append(sum_cells, len(np.setdiff1d(np.arange(0,n_Cells), nanCells)))

goodSessions = goodSessions.astype(int)


#%% low vs high fano factor spont -- single cell level

lowPupil_FF_goodSessions_spont, highPupil_FF_goodSessions_spont = \
    fcn_plot_fanofactor.fcn_combine_vecQuantity_low_high_pupil_overSessions(freqAvg_fanofactor_sigCells_spont[goodSessions, :])

low_vs_high_fanoStats_goodSessions_spont = fcn_Wilcoxon(lowPupil_FF_goodSessions_spont, highPupil_FF_goodSessions_spont)


#%% low vs high fano factor evoked -- single cell level

lowPupil_FF_goodSessions_evoked, highPupil_FF_goodSessions_evoked = \
    fcn_plot_fanofactor.fcn_combine_vecQuantity_low_high_pupil_overSessions(freqAvg_fanofactor_sigCells_evoked[goodSessions, :])

low_vs_high_fanoStats_goodSessions_evoked = fcn_Wilcoxon(lowPupil_FF_goodSessions_evoked, highPupil_FF_goodSessions_evoked)


#%% low vs high fano factor diff -- single cell level

lowPupil_FF_goodSessions_diff, highPupil_FF_goodSessions_diff = \
    fcn_plot_fanofactor.fcn_combine_vecQuantity_low_high_pupil_overSessions(freqAvg_fanofactor_sigCells_diff[goodSessions, :])

low_vs_high_fanoStats_goodSessions_diff = fcn_Wilcoxon(lowPupil_FF_goodSessions_diff, highPupil_FF_goodSessions_diff)


#%% print statistics

if vary_windowLength == False:
    
    print('spont_stats:', low_vs_high_fanoStats_goodSessions_spont)
    print('mean_diff_spont:', np.nanmean(lowPupil_FF_goodSessions_spont - highPupil_FF_goodSessions_spont))
    print('spont_low mean +/- sem: %0.5f + %0.5f' % ( np.nanmean(lowPupil_FF_goodSessions_spont), scipy.stats.sem(lowPupil_FF_goodSessions_spont, nan_policy='omit') ) )
    print('spont_high mean +/- sem: %0.5f + %0.5f' % ( np.nanmean(highPupil_FF_goodSessions_spont), scipy.stats.sem(highPupil_FF_goodSessions_spont, nan_policy='omit') ) )
    print('*'*100)
    
    print('evoked_stats:', low_vs_high_fanoStats_goodSessions_evoked)
    print('mean_diff_evoked:',  np.nanmean(lowPupil_FF_goodSessions_evoked - highPupil_FF_goodSessions_evoked))
    print('evoked_low mean +/- sem: %0.5f + %0.5f' % ( np.nanmean(lowPupil_FF_goodSessions_evoked), scipy.stats.sem(lowPupil_FF_goodSessions_evoked, nan_policy='omit') ) )
    print('evoked_high mean +/- sem: %0.5f + %0.5f' % ( np.nanmean(highPupil_FF_goodSessions_evoked), scipy.stats.sem(highPupil_FF_goodSessions_evoked, nan_policy='omit') ) )
    print('*'*100)
    
    
    print('diff_stats:', low_vs_high_fanoStats_goodSessions_diff)
    print('mean_diff_diff:',  np.nanmean(lowPupil_FF_goodSessions_diff - highPupil_FF_goodSessions_diff))
    print('diff_low mean +/- sem: %0.5f + %0.5f' % ( np.nanmean(lowPupil_FF_goodSessions_diff), scipy.stats.sem(lowPupil_FF_goodSessions_diff, nan_policy='omit') ) )
    print('diff_high mean +/- sem: %0.5f + %0.5f' % ( np.nanmean(highPupil_FF_goodSessions_diff), scipy.stats.sem(highPupil_FF_goodSessions_diff, nan_policy='omit') ) )
    print('*'*100)


#%% cell and session average spont

pupilSize_binCenters_spont, avg_fanofactor_vs_pupilSize_bins_spont, std_fanofactor_vs_pupilSize_bins_spont, sem_fanofactor_vs_pupilSize_bins_spont, \
    allSessions_fanofactor_pupilBins_spont, pupilSize_data_in_pupilBins_allSessions_spont = \
    fcn_cellAvg_sessionAvg_quantity_vs_pupilSize_bins_alt(pupilBin_centers_raw, 0, 1, pupilSize_binStep, freqAvg_fanofactor_sigCells_spont, 1)


#%% cell and session average evoked

pupilSize_binCenters_evoked, avg_fanofactor_vs_pupilSize_bins_evoked, std_fanofactor_vs_pupilSize_bins_evoked, sem_fanofactor_vs_pupilSize_bins_evoked, \
    allSessions_fanofactor_pupilBins_evoked, pupilSize_data_in_pupilBins_allSessions_evoked = \
    fcn_cellAvg_sessionAvg_quantity_vs_pupilSize_bins_alt(pupilBin_centers_raw, 0, 1, pupilSize_binStep, freqAvg_fanofactor_sigCells_evoked, 1)


#%% cell and session average diff

pupilSize_binCenters_diff, avg_fanofactor_vs_pupilSize_bins_diff, std_fanofactor_vs_pupilSize_bins_diff, sem_fanofactor_vs_pupilSize_bins_diff, \
    allSessions_fanofactor_pupilBins_diff, pupilSize_data_in_pupilBins_allSessions_diff = \
    fcn_cellAvg_sessionAvg_quantity_vs_pupilSize_bins_alt(pupilBin_centers_raw, 0, 1, pupilSize_binStep, freqAvg_fanofactor_sigCells_diff, 1)

    
#%% save parameters and results

params = {}
results = {}

params['window_length'] = window_length
params['window_length_psth'] = window_length_psth
params['avgRate_thresh'] = avgRate_thresh
params['nPupil_bins'] = nPupil_bins
params['lowPupil_thresh'] = lowPupil_thresh
params['highPupil_thresh'] = highPupil_thresh
params['pupilSize_binStep'] = 0.1
params['sessions_to_run'] = sessions_to_run
params['nFreq'] = nFreq
params['psth_path'] = psth_path
params['data_path'] = data_path
params['fig_path'] = fig_path
params['sig_level'] = sig_level
params['t_eval_FFevoked'] = t_eval_FFevoked
params['cellSelection'] = cellSelection
params['fano_filename'] = fano_filename_raw
params['vary_windowLength'] = vary_windowLength
params['evoked_window'] = evoked_window
params['pupil_lag'] = pupil_lag
results['params'] = params
results['goodSessions'] = goodSessions
results['low_vs_high_fanoStats_goodSessions_spont'] = low_vs_high_fanoStats_goodSessions_spont
results['low_vs_high_fanoStats_goodSessions_evoked'] = low_vs_high_fanoStats_goodSessions_evoked
results['low_vs_high_fanoStats_goodSessions_diff'] = low_vs_high_fanoStats_goodSessions_diff

save_filename = (fig_path + savename + 'stats.mat')      
savemat(save_filename, results) 


#%% plotting

#%%

#######----------------------------------------------------------------#######


if vary_windowLength == True:

    plt.rcParams.update({'font.size': 7})
    fig = plt.figure(figsize=(1.,1.))  
    ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 
    
    x = pupilSize_binCenters_spont.copy() * 100
    y = avg_fanofactor_vs_pupilSize_bins_spont.copy()
    yerr = sem_fanofactor_vs_pupilSize_bins_spont.copy()
    xerr = pupilSize_binStep*100/2
    yup = y + yerr
    ylow = y - yerr
    
    ax.errorbar(x, y, yerr=yerr, xerr=xerr, color='k', linewidth=0.75, fmt='none')
    ax.plot(x, y, '-',  color='k', markersize=1, linewidth=0.5)
    ax.plot(x, y, 'o',  color=np.array([175,54,60])/256, markersize=1)
    
    ax.set_xticks([0,100])
    ax.set_xlim([-2,102])
    
    if window_length == 100e-3:
        if cellSelection == '':
            ax.set_yticks([1.3, 2.4])
        else:
            ax.set_yticks([1.3,2.3])
            
    if window_length == 200e-3:
        if cellSelection == '':
            ax.set_yticks([1.7, 3.2])
    
    if window_length == 50e-3:
        if cellSelection == '':
            ax.set_yticks([1.2, 1.9])
            
    ax.set_xlabel('pupil diameter\n[% max]')
    ax.set_ylabel('$\\langle FF_{spont} \\rangle$')
    plt.savefig(fig_path + '%s.pdf' % (fig0ID), bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close('all')


#%%

#######----------------------------------------------------------------#######

if ( (vary_windowLength == False) and (window_length == 100e-3) ):
    

### cell and session averaged FF -- spont

    
    plt.rcParams.update({'font.size': 7})
    fig = plt.figure(figsize=(1.,1.))  
    ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 
    
    x = pupilSize_binCenters_spont.copy() * 100
    y = avg_fanofactor_vs_pupilSize_bins_spont.copy()
    yerr = sem_fanofactor_vs_pupilSize_bins_spont.copy()
    xerr = pupilSize_binStep*100/2
    yup = y + yerr
    ylow = y - yerr
    
    ax.errorbar(x, y, yerr=yerr, xerr=xerr, color='k', linewidth=0.75, fmt='none')
    ax.plot(x, y, '-',  color='k', markersize=1, linewidth=0.5)
    ax.plot(x, y, 'o',  color=np.array([175,54,60])/256, markersize=1)
    
    ax.set_xticks([0,100])
    ax.set_xlim([-2,102])
    
    if cellSelection == '':
        ax.set_yticks([1.3, 2.4])
    else:
        ax.set_yticks([1.3,2.3])
            
    ax.set_xlabel('pupil diameter\n[% max]')
    ax.set_ylabel('$\\langle FF_{spont} \\rangle$')
    plt.savefig(fig_path + '%s.pdf' % (fig1ID), bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close('all')
    

### cell and session averaged FF -- evoked


    plt.rcParams.update({'font.size': 7})
    fig = plt.figure(figsize=(1.,1.))  
    ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 
    
    x = pupilSize_binCenters_evoked.copy() * 100
    y = avg_fanofactor_vs_pupilSize_bins_evoked.copy()
    yerr = sem_fanofactor_vs_pupilSize_bins_evoked.copy()
    xerr = pupilSize_binStep*100/2
    yup = y + yerr
    ylow = y - yerr
    
    ax.errorbar(x, y, yerr=yerr, xerr=xerr, color='k', linewidth=0.75, fmt='none')
    ax.plot(x, y, '-',  color='k', markersize=1, linewidth=0.5)
    ax.plot(x, y, 'o',  color=np.array([175,54,60])/256, markersize=1)
    
    ax.set_xticks([0,100])
    ax.set_xlim([-2,102])
    
    if cellSelection == '':
        ax.set_yticks([1.1, 1.9])
    else:
        ax.set_yticks([1.0,1.7])
        
    ax.set_xlabel('pupil diameter\n[% max]')
    ax.set_ylabel('$\\langle FF_{evoked} \\rangle$')
    plt.savefig(fig_path + '%s.pdf' % (fig2ID), bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close('all')
    
    
### cell and session averaged FF -- diff
    
    plt.rcParams.update({'font.size': 7})
    fig = plt.figure(figsize=(1.,1.))  
    ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 
    
    x = pupilSize_binCenters_diff.copy() * 100
    y = avg_fanofactor_vs_pupilSize_bins_diff.copy()
    yerr = sem_fanofactor_vs_pupilSize_bins_diff.copy()
    xerr = pupilSize_binStep*100/2
    yup = y + yerr
    ylow = y - yerr
    
    ax.errorbar(x, y, yerr=yerr, xerr=xerr, color='k', linewidth=0.75, fmt='none')
    ax.plot(x, y, '-',  color='k', markersize=1, linewidth=0.5, label='(spont - evoked)')
    ax.plot(x, y, 'o',  color=np.array([175,54,60])/256, markersize=1)
    
    ax.set_xticks([0,100])
    ax.set_xlim([-2,102])
    
    if cellSelection == '':
        ax.set_yticks([0.0, 0.6])
    else:
        ax.set_yticks([0.2,0.65])
             
    ax.set_xlabel('pupil diameter\n[% max]')
    ax.set_ylabel('$\\langle \\Delta FF \\rangle$' + ' $_{(spont-evoked)}$')
    plt.savefig(fig_path + '%s.pdf' % (fig3ID), bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close('all')
    
    
#%%
    
if ( (cellSelection == '') and (vary_windowLength == False) and (window_length == 100e-3)):

    
### for good sessions, plot FF low pupil vs high pupil -- spont
        
    plt.rcParams.update({'font.size': 7})
    fig = plt.figure(figsize=(1.,1.))  
    ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 
        
    x = lowPupil_FF_goodSessions_spont.copy()
    y = highPupil_FF_goodSessions_spont.copy()
     
    min_data = np.nanmin(np.append(x,y))
    max_data = np.nanmax(np.append(x,y))
    
    ax.plot(x, y, 'o', color='black', markersize=0.5, alpha=0.3)
    ax.plot([min_data, max_data], [min_data, max_data], color='teal', linewidth=0.5)
    ax.plot(np.nanmean(x), np.nanmean(y), marker='d', color='r', markersize=1)
    ax.set_xticks([0.5, 8.5])
    ax.set_yticks([0.5, 8.5])
    ax.axis('equal')
    ax.set_xlabel('small pupil')
    ax.set_ylabel('large pupil')
        
    plt.savefig(fig_path + '%s.pdf' % (fig4ID), bbox_inches='tight', pad_inches=0, transparent=True)
    
    
    plt.rcParams.update({'font.size': 7})
    fig = plt.figure(figsize=(1.,1.))  
    ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 
    
    ax.bar(0, np.nanmean(x), width=0.6, color='dimgrey')
    ax.errorbar(0, np.nanmean(x), yerr=scipy.stats.sem(x, nan_policy='omit'), color='r')
    
    ax.bar(1, np.nanmean(y), width=0.6, color='dimgrey')
    ax.errorbar(1, np.nanmean(y), yerr=scipy.stats.sem(y,nan_policy='omit'), color='r')
    ax.set_ylabel('$\\langle FF_{spont} \\rangle$')
    ax.set_xticks([0,1])
    ax.set_xticklabels(['small\npupil', 'large\npupil'])
    plt.savefig(fig_path + '%s.pdf' % (fig5ID), bbox_inches='tight', pad_inches=0, transparent=True)
    
    plt.close('all')
    
    
### for good sessions, plot FF low pupil vs high pupil -- evoked
        
    plt.rcParams.update({'font.size': 7})
    fig = plt.figure(figsize=(1.,1.))  
    ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 
        
    x = lowPupil_FF_goodSessions_evoked.copy()
    y = highPupil_FF_goodSessions_evoked.copy()
     
    min_data = np.nanmin(np.append(x,y))
    max_data = np.nanmax(np.append(x,y))
    
    ax.plot(x, y, 'o', color='black', markersize=0.5, alpha=0.3)
    ax.plot([min_data, max_data], [min_data, max_data], color='teal', linewidth=0.5)
    ax.plot(np.nanmean(x), np.nanmean(y), marker='d', color='r', markersize=1)
    ax.set_xticks([0.5, 4.5])
    ax.set_yticks([0.5, 4.5])
    ax.axis('equal')
    ax.set_xlabel('small pupil')
    ax.set_ylabel('large pupil')
    plt.savefig(fig_path + '%s.pdf' % (fig6ID), bbox_inches='tight', pad_inches=0, transparent=True)
    
    
    plt.rcParams.update({'font.size': 7})
    fig = plt.figure(figsize=(1.,1.))  
    ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 
    
    ax.bar(0, np.nanmean(x), width=0.6, color='dimgrey')
    ax.errorbar(0, np.nanmean(x), yerr=scipy.stats.sem(x, nan_policy='omit'), color='r')
    
    ax.bar(1, np.nanmean(y), width=0.6, color='dimgrey')
    ax.errorbar(1, np.nanmean(y), yerr=scipy.stats.sem(y,nan_policy='omit'), color='r')
    ax.set_ylabel('$\\langle FF_{evoked} \\rangle$')
    ax.set_xticks([0,1])
    ax.set_xticklabels(['small\npupil', 'large\npupil'])
    plt.savefig(fig_path + '%s.pdf' % (fig7ID), bbox_inches='tight', pad_inches=0, transparent=True)
    
    
    plt.close('all')
    
    
### for good sessions, plot FF low pupil vs high pupil -- diff
        
    plt.rcParams.update({'font.size': 7})
    fig = plt.figure(figsize=(1.,1.))  
    ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 
        
    x = lowPupil_FF_goodSessions_diff.copy()
    y = highPupil_FF_goodSessions_diff.copy()
     
    min_data = np.nanmin(np.append(x,y))
    max_data = np.nanmax(np.append(x,y))
    
    ax.plot(x, y, 'o', color='black', markersize=0.5, alpha=0.3)
    ax.plot([min_data, max_data], [min_data, max_data], color='teal', linewidth=0.5)
    ax.plot(np.nanmean(x), np.nanmean(y), marker='d', color='r', markersize=1)
    ax.set_xticks([0., 5])
    ax.set_yticks([0., 5])
    ax.axis('equal')
    
    ax.set_xlabel('small pupil')
    ax.set_ylabel('large pupil')
        
    plt.savefig(fig_path + '%s.pdf' % (fig8ID), bbox_inches='tight', pad_inches=0, transparent=True)
    
    
    plt.rcParams.update({'font.size': 7})
    fig = plt.figure(figsize=(1.,1.))  
    ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 
    
    ax.bar(0, np.nanmean(x), width=0.6, color='dimgrey')
    ax.errorbar(0, np.nanmean(x), yerr=scipy.stats.sem(x, nan_policy='omit'), color='r')
    
    ax.bar(1, np.nanmean(y), width=0.6, color='dimgrey')
    ax.errorbar(1, np.nanmean(y), yerr=scipy.stats.sem(y,nan_policy='omit'), color='r')
    ax.set_ylabel('$\\langle \\Delta FF \\rangle$')
    ax.set_xticks([0,1])
    ax.set_xticklabels(['small\npupil', 'large\npupil'])
    plt.savefig(fig_path + '%s.pdf' % (fig9ID), bbox_inches='tight', pad_inches=0, transparent=True)
    
    plt.close('all')
    

#%% 

if ( (vary_windowLength == False) and (window_length == 100e-3) ):

    
### for good sessions, plot delta FF (low pupil - high pupil) -- spont
    
    colorPlot = 'dimgrey'
    bin_width = 0.2
    outlier_cutoff = 4
    
    hist_data = (lowPupil_FF_goodSessions_spont - highPupil_FF_goodSessions_spont)
    hist_data = hist_data[~np.isnan(hist_data)]
    mean_hist_data = np.mean(hist_data)
    
    neg_outliersInds = np.nonzero(hist_data < -outlier_cutoff)[0]
    neg_outliers = hist_data[neg_outliersInds]
    pos_outliersInds =  np.nonzero(hist_data > outlier_cutoff)[0]
    pos_outliers = hist_data[pos_outliersInds]
    hist_data[neg_outliersInds] = np.nan
    hist_data[pos_outliersInds] = np.nan
    
    hist_data = hist_data[~np.isnan(hist_data)]
    data_extreme = np.nanmax(np.abs(hist_data))
    
    bins = np.arange( np.round(-outlier_cutoff - bin_width, 1), np.round(outlier_cutoff + 3*bin_width, 1), bin_width )
    counts, bin_edges = np.histogram(hist_data, bins)
    bin_centers = bin_edges[:-1] + bin_width/2
    
    plt.rcParams.update({'font.size': 7})
    fig = plt.figure(figsize=(1.,1.))  
    ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 
    ax.tick_params(axis='both', width=0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.bar(-outlier_cutoff - bin_width/2, np.size(neg_outliers), width = bin_width, color=colorPlot)
    ax.bar(outlier_cutoff + bin_width/2, np.size(pos_outliers), width = bin_width, color=colorPlot)
    ax.bar(bin_centers, counts, width=np.diff(bin_edges), color=colorPlot)
    ax.plot([0,0], [0,np.nanmax(counts)], color='teal', linewidth=1)
    ax.plot(mean_hist_data, np.max(counts)+2, 'v', color='r', markersize=1)
    ax.plot([],[],label='p < 0.001')
    
    ax.arrow(-outlier_cutoff-bin_width/2, 8, 0, -5, color='k', linewidth=0.5, head_width=0.02, head_length=0.08)
    ax.arrow(outlier_cutoff+bin_width/2, 8, 0, -5, color='k', linewidth=0.5, head_width=0.02, head_length=0.08)
    ax.text(-outlier_cutoff - bin_width, 10, ('< -%d' % outlier_cutoff), fontsize=4)
    ax.text(outlier_cutoff - bin_width, 10, ('> %d' % outlier_cutoff), fontsize=4)
    
    if cellSelection == '':
        ax.text(outlier_cutoff/2., 0.8*np.nanmax(counts), 'p<0.001', fontsize=5)
    
    ax.set_xticks([-outlier_cutoff,0,outlier_cutoff])
    ax.set_xlim([-outlier_cutoff-1.5*bin_width, outlier_cutoff+1.5*bin_width])
    ax.set_xlabel('$FF_{spont}^{small} - FF_{spont}^{large}$')
    ax.set_ylabel('cell count')
    plt.savefig(fig_path + '%s.pdf' % (fig10ID), bbox_inches='tight', pad_inches=0, transparent=True)
    
    plt.close('all')
    
    
### for good sessions, plot delta FF (low pupil - high pupil) -- evoked
    
    colorPlot = 'dimgrey'
    bin_width = 0.2
    outlier_cutoff = 2
    
    hist_data = (lowPupil_FF_goodSessions_evoked - highPupil_FF_goodSessions_evoked)
    hist_data = hist_data[~np.isnan(hist_data)]
    mean_hist_data = np.mean(hist_data)
    
    neg_outliersInds = np.nonzero(hist_data < -outlier_cutoff)[0]
    neg_outliers = hist_data[neg_outliersInds]
    pos_outliersInds =  np.nonzero(hist_data > outlier_cutoff)[0]
    pos_outliers = hist_data[pos_outliersInds]
    hist_data[neg_outliersInds] = np.nan
    hist_data[pos_outliersInds] = np.nan
    
    hist_data = hist_data[~np.isnan(hist_data)]
    data_extreme = np.nanmax(np.abs(hist_data))
    
    bins = np.arange( np.round(-outlier_cutoff - bin_width, 1), np.round(outlier_cutoff + 3*bin_width, 1), bin_width )
    counts, bin_edges = np.histogram(hist_data, bins)
    bin_centers = bin_edges[:-1] + bin_width/2
    
    plt.rcParams.update({'font.size': 7})
    fig = plt.figure(figsize=(1.,1.))  
    ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 
    ax.tick_params(axis='both', width=0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.bar(-outlier_cutoff - bin_width/2, np.size(neg_outliers), width = 0.2, color=colorPlot)
    ax.bar(outlier_cutoff + bin_width/2, np.size(pos_outliers), width = 0.2, color=colorPlot)
    ax.bar(bin_centers, counts, width=np.diff(bin_edges), color=colorPlot)
    ax.plot([0,0], [0,np.nanmax(counts)], color='teal', linewidth=1)
    ax.plot(mean_hist_data, np.max(counts)+2, 'v', color='r', markersize=1)
    
    ax.arrow(-outlier_cutoff-bin_width/2, 13, 0, -5, color='k', linewidth=0.5, head_width=0.01, head_length=0.08)
    ax.arrow(outlier_cutoff+bin_width/2, 13, 0, -5, color='k', linewidth=0.5, head_width=0.01, head_length=0.08)
    ax.text(-outlier_cutoff - bin_width, 15, ('< -%d' % outlier_cutoff), fontsize=4)
    ax.text(outlier_cutoff - bin_width, 15, ('> %d' % outlier_cutoff), fontsize=4)
    
    if cellSelection == '':
        ax.text(outlier_cutoff/2., 0.8*np.nanmax(counts), 'p<0.001', fontsize=5)
    
    ax.set_xticks([-outlier_cutoff,0,outlier_cutoff])
    ax.set_xlim([-outlier_cutoff-1.5*bin_width, outlier_cutoff+1.5*bin_width])
    ax.set_xlabel('$FF_{evoked}^{small} - FF_{evoked}^{large}$')
    ax.set_ylabel('cell count')
    plt.savefig(fig_path + '%s.pdf' % (fig11ID), bbox_inches='tight', pad_inches=0, transparent=True)
    
    plt.close('all')
    
    
### for good sessions, plot delta FF (low pupil - high pupil) -- diff
    
    colorPlot = 'dimgrey'
    bin_width = 0.2
    outlier_cutoff = 2
    
    hist_data = (lowPupil_FF_goodSessions_diff - highPupil_FF_goodSessions_diff)
    hist_data = hist_data[~np.isnan(hist_data)]
    mean_hist_data = np.mean(mean_hist_data)
    
    neg_outliersInds = np.nonzero(hist_data < -outlier_cutoff)[0]
    neg_outliers = hist_data[neg_outliersInds]
    pos_outliersInds =  np.nonzero(hist_data > outlier_cutoff)[0]
    pos_outliers = hist_data[pos_outliersInds]
    hist_data[neg_outliersInds] = np.nan
    hist_data[pos_outliersInds] = np.nan
    
    hist_data = hist_data[~np.isnan(hist_data)]
    data_extreme = np.nanmax(np.abs(hist_data))
    
    bins = np.arange( np.round(-outlier_cutoff - bin_width, 1), np.round(outlier_cutoff + 3*bin_width, 1), bin_width )
    counts, bin_edges = np.histogram(hist_data, bins)
    bin_centers = bin_edges[:-1] + bin_width/2
    
    plt.rcParams.update({'font.size': 7})
    fig = plt.figure(figsize=(1.,1.))  
    ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 
    ax.tick_params(axis='both', width=0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.bar(-outlier_cutoff - bin_width/2, np.size(neg_outliers), width = 0.2, color=colorPlot)
    ax.bar(outlier_cutoff + bin_width/2, np.size(pos_outliers), width = 0.2, color=colorPlot)
    ax.bar(bin_centers, counts, width=np.diff(bin_edges), color=colorPlot)
    ax.plot([0,0], [0,np.nanmax(counts)], color='teal', linewidth=1)
    ax.plot(mean_hist_data, np.max(counts)+2, 'v', color='r', markersize=1)
    
    ax.arrow(-outlier_cutoff-bin_width/2, 12, 0, -5, color='k', linewidth=0.5, head_width=0.01, head_length=0.08)
    ax.arrow(outlier_cutoff+bin_width/2, 12, 0, -5, color='k', linewidth=0.5, head_width=0.01, head_length=0.08)
    ax.text(-outlier_cutoff - bin_width, 14, ('< -%d' % outlier_cutoff), fontsize=4)
    ax.text(outlier_cutoff - bin_width, 14, ('> %d' % outlier_cutoff), fontsize=4)
    
    if cellSelection == '':
        ax.text(outlier_cutoff/2., 0.8*np.nanmax(counts), 'p<0.001', fontsize=5)
    
    ax.set_xticks([-outlier_cutoff, 0, outlier_cutoff])
    ax.set_xlim([-outlier_cutoff-1.5*bin_width, outlier_cutoff+1.5*bin_width])
    ax.set_xlabel('$\\Delta FF^{small} - \\Delta FF^{large}$')
    ax.set_ylabel('cell count')
    plt.savefig(fig_path + '%s.pdf' % (fig12ID), bbox_inches='tight', pad_inches=0, transparent=True)

    plt.close('all')
