
'''
This script generates
    FigS6H
    FigS6I
    FigS6J
    FigS6K
    FigS6L
'''

#%% basic imports
import sys
import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
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

func_path0 = global_settings.path_to_src_code + 'data_analysis/fanofactor_vs_pupil/'
func_path1 = global_settings.path_to_src_code + 'data_analysis/'
func_path2 = global_settings.path_to_src_code + 'functions/'

sys.path.append(func_path0)
sys.path.append(func_path1)
sys.path.append(func_path2)

import fcn_plot_fanofactor
from fcn_SuData_analysis import fcn_cellAvg_sessionAvg_quantity_vs_pupilSize_bins_alt
from fcn_SuData_analysis import cellAvg_sessionAvg_spectra_vs_pupilDiamBins
from fcn_statistics import fcn_Wilcoxon

#%% settings

# for spectra
window_length = 2.5
avgRate_thresh = 1
df = 1.6
lowFreq_band = [1,4]
rest_only = False
dcSubtract_type = 0
avg_type = 2
estimation = 'mt'

# pupil binning
nPercentile_bins = 10
lowPupil_thresh = 0.33
highPupil_thresh = 0.67
pupilSize_binStep = 0.1

# sessions to run
sessions_to_run = np.array(['LA3_session3', \
                            'LA8_session1', 'LA8_session2', \
                            'LA9_session1', 'LA9_session3', 'LA9_session4', 'LA9_session5', \
                            'LA11_session1','LA11_session2', 'LA11_session3', 'LA11_session4', \
                            'LA12_session1', 'LA12_session2', 'LA12_session3', 'LA12_session4'])
    
# session to plot
session_plot = 'LA11_session1'

# paths
data_path = global_settings.path_to_data_analysis_output + 'spont_spikeSpectra_pupil/'
fig_path = global_settings.path_to_manuscript_figs_final + 'spectra_data/'

# filenames
figname = 'spectra_data_'
filename_spectra = 'spont_powerSpectra_pupilPercentile'

figID1 = 'figS6H'
figID2 = 'figS6I'
figID3 = 'figS6J'
figID4 = 'figS6K'
figID5 = 'figS6L'


#%% functions

def fcn_avgPower(power, freq):
    meanPower = np.mean(power)
    return meanPower

#%% setup

if rest_only:
    fName_end = 'restOnly'
else:
    fName_end = ''
    
if os.path.isdir(fig_path) == False:
    os.makedirs(fig_path)
    
#%% initialize 

# pupil bin centers
pupilBin_centers_raw = np.ones((len(sessions_to_run), nPercentile_bins))*np.nan
nTrials_subsample = np.zeros((len(sessions_to_run)))

# spectra
singleCell_norm_spectra  = np.ones((len(sessions_to_run), nPercentile_bins), dtype='object')*np.nan
cellAvg_norm_spectra = np.ones((len(sessions_to_run), nPercentile_bins), dtype='object')*np.nan
cellSem_norm_spectra = np.ones((len(sessions_to_run), nPercentile_bins), dtype='object')*np.nan

singleCell_lowFreqPower  = np.ones((len(sessions_to_run), nPercentile_bins), dtype='object')*np.nan
cellAvg_lowFreqPower = np.ones((len(sessions_to_run), nPercentile_bins))*np.nan
cellSem_lowFreqPower = np.ones((len(sessions_to_run), nPercentile_bins))*np.nan

allSessions_singleCell_lowFreqPower =  np.ones((nPercentile_bins), dtype='object')*np.nan

# frequency
frequency_spectra = np.ones((len(sessions_to_run)), dtype='object')*np.nan

goodSessions = np.array([])

sum_cells = np.array([])


#%% load the data

for count, session_name in enumerate(sessions_to_run):

    
    ###### spectra ###### 
    
    # filename 
    filename = (('%s%s_%s_windLength%0.3fs_%s_raw%d.mat') % (data_path, filename_spectra, session_name, window_length, fName_end, avg_type))
    
    # data
    data = loadmat(filename, simplify_cells=True)
    
    # spontaneous rate and number of trials
    spont_rate = data['spont_trialAvg_spikeCount']/window_length
    nTrials_subsample[count] = data['nTrials_subsample']
  
    # pupil bin centers 
    pupilBin_centers_raw[count, :] = data['avg_pupilSize_trials'].copy()

    
    ### determine if this is a good session
    if ( (np.min(pupilBin_centers_raw[count, :]) <= lowPupil_thresh) and (np.max(pupilBin_centers_raw[count, :]) >= highPupil_thresh)) :
        
        goodSessions = np.append(goodSessions, count)
    

    # number of cells
    n_Cells = np.size(spont_rate, 0)
    
    # cells that pass rate cut
    min_spikeCount_allPupil = np.min(spont_rate, 1)
    min_rate_allPupil = min_spikeCount_allPupil
    cells_pass_rate_cut = np.nonzero( min_rate_allPupil >= avgRate_thresh )[0]
    
    # cells that pass rate cut and that are significant
    cells_to_use = np.intersect1d(np.arange(n_Cells), cells_pass_rate_cut)
    
    # cells not to use
    cells_not_use = np.setdiff1d(np.arange(n_Cells), cells_to_use)
    

    # load the spectra data
    if estimation == 'raw':
        
        spectra = data['norm_power_spectra_raw']
        spectra = spectra[:, :, :, dcSubtract_type].copy()
        freq_spectra = data['frequency_spectra_raw']        
        
    elif estimation == 'mt':
        
        df_array = data['params']['df_array']
        spectra = data['norm_power_spectra']
        freq_spectra = data['frequency_spectra']
        
        if len(df_array) > 1:
            ind_df = np.nonzero(df_array == df)[0][0]
            spectra = spectra[:, :, :, ind_df, dcSubtract_type].copy()
        else:
            spectra = spectra[:, :, :, dcSubtract_type].copy()    

        
    # nan-out bad cells
    spectra[cells_not_use, :, :] = np.nan
    spont_rate[cells_not_use, :] = np.nan

  
    # frequency indices for low frequency band ------------------------------
    lowFreq_indLow = np.argmin(np.abs(freq_spectra - lowFreq_band[0]))
    lowFreq_indHigh = np.argmin(np.abs(freq_spectra - lowFreq_band[1]))
    


    # loop over pupil bins
    nanCells = np.array([])
    for indPupil in range(0, nPercentile_bins):
        
        # normalized spectra
        norm_spectra = np.ones((n_Cells, len(freq_spectra)))*np.nan
        
        # low frequency power
        lowFreq_power = np.ones((n_Cells))*np.nan
        
        for indCell in range(0, n_Cells):
            
            norm_spectra[indCell, :] = spectra[indCell, indPupil, :].copy()
            lowFreq_power[indCell] = fcn_avgPower(norm_spectra[indCell, lowFreq_indLow:lowFreq_indHigh+1], \
                                                  freq_spectra[lowFreq_indLow:lowFreq_indHigh+1])
                
        
        singleCell_norm_spectra[count, indPupil] = norm_spectra.copy() 
        singleCell_lowFreqPower[count, indPupil] = lowFreq_power.copy()
       
        # nan cells
        nanCells = np.append(nanCells, np.nonzero(np.isnan(lowFreq_power))[0])
        
    nanCells = np.unique(nanCells).astype(int)
        
    
    # loop over pupil bins and average across cells
    for indPupil in range(0, nPercentile_bins):
    
        singleCell_norm_spectra[count, indPupil][nanCells,:] = np.nan
        singleCell_lowFreqPower[count, indPupil][nanCells] = np.nan
            
    
        # single cell and cell-averaged normalize spectra
        cellAvg_norm_spectra[count, indPupil] = np.nanmean(singleCell_norm_spectra[count, indPupil], 0)        
        cellSem_norm_spectra[count, indPupil] = scipy.stats.sem(singleCell_norm_spectra[count, indPupil], axis=0, nan_policy='omit') 
        
        # single cell and cell-averaged low frequency power
        cellAvg_lowFreqPower[count, indPupil] = np.nanmean(singleCell_lowFreqPower[count, indPupil])
        cellSem_lowFreqPower[count, indPupil] = scipy.stats.sem(singleCell_lowFreqPower[count, indPupil], nan_policy='omit' )
        

    # frequency values
    frequency_spectra[count] = freq_spectra.copy()
    
    sum_cells = np.append(sum_cells, len(np.setdiff1d(np.arange(0,n_Cells), nanCells)))


goodSessions = goodSessions.astype(int)
    
#%% low pupil vs high pupil low frequency power -- all cells from good sessions

lowPupil_lowFreq_power_goodSessions, highPupil_lowFreq_power_goodSessions = \
    fcn_plot_fanofactor.fcn_combine_vecQuantity_low_high_pupil_overSessions(singleCell_lowFreqPower[goodSessions, :])

low_vs_high_singleCells_stats_goodSessions = fcn_Wilcoxon(lowPupil_lowFreq_power_goodSessions, highPupil_lowFreq_power_goodSessions)

print('# good sessions:', np.size(goodSessions))
print('stats:', low_vs_high_singleCells_stats_goodSessions)
print('low pupil:', np.nanmean(lowPupil_lowFreq_power_goodSessions), scipy.stats.sem(lowPupil_lowFreq_power_goodSessions, nan_policy='omit'))
print('high pupil:', np.nanmean(highPupil_lowFreq_power_goodSessions), scipy.stats.sem(highPupil_lowFreq_power_goodSessions, nan_policy='omit'))
print('mean_diff_spont:', np.nanmean(lowPupil_lowFreq_power_goodSessions - highPupil_lowFreq_power_goodSessions))


#%% cell and session average

pupilSize_binCenters, avg_quantity_vs_pupilSize_bins_spont, std_quantity_vs_pupilSize_bins_spont, sem_quantity_vs_pupilSize_bins_spont, \
    allSessions_quantity_pupilBins_spont, pupilSize_data_in_pupilBins_allSessions_spont = \
    fcn_cellAvg_sessionAvg_quantity_vs_pupilSize_bins_alt(pupilBin_centers_raw, 0, 1, pupilSize_binStep, singleCell_lowFreqPower, 1)
    
    
#%% average power spectra in 3 large pupil bins

bin_lower = [0, 0.33, 0.67]
bin_upper = [0.33, 0.67, 1]

n_bins = len(bin_lower)

grandAvg_binned_spec, grandStd_binned_spec, grandSem_binned_spec = \
    cellAvg_sessionAvg_spectra_vs_pupilDiamBins(singleCell_norm_spectra, pupilBin_centers_raw, bin_lower, bin_upper)

#%% save parameters and results

params = {}
results = {}

params['sessions_to_run'] = sessions_to_run
params['data_path'] = data_path
params['fig_path'] = fig_path
params['filename_spectra'] = filename_spectra
params['restOnly'] = rest_only
params['avgRate_thresh'] = avgRate_thresh
params['window_length'] = window_length
params['df'] = df
params['dcSubtract_type'] = dcSubtract_type
params['avg_type'] = avg_type
params['estimation'] = estimation
params['lowFreq_band'] = lowFreq_band
params['nPercentile_bins'] = nPercentile_bins
params['lowPupil_thresh'] = lowPupil_thresh
params['highPupil_thresh'] = highPupil_thresh
params['pupilSize_binStep'] = pupilSize_binStep
params['session_plot'] = session_plot
results['params'] = params
results['goodSessions'] = goodSessions
results['singleCell_lowFreq_power_stats_goodSession'] = low_vs_high_singleCells_stats_goodSessions


save_filename = (fig_path + figname + '_stats.mat')      
savemat(save_filename, results) 


#%% for all sessions, plot cell avg power spectra at low, mid and high pupil

plt.rcParams.update({'font.size': 7})
fig = plt.figure(figsize=(1.,1.))  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 

x = frequency_spectra[0][1:].copy()
y = grandAvg_binned_spec[0,1:].copy()
yerr = grandSem_binned_spec[0,1:].copy()
ax.fill_between(x, y-yerr, y+yerr, where=None, color='steelblue', alpha=0.5,linewidth=0)
ax.plot(x, y, color='steelblue', label='small pupil',linewidth=0.5)
ax.set_xscale('log')
ax.set_xlim([0.8, 500])


x = frequency_spectra[0][1:].copy()
y = grandAvg_binned_spec[1,1:].copy()
yerr = grandSem_binned_spec[1,1:].copy()
ax.fill_between(x, y-yerr, y+yerr, where=None, color='cornflowerblue', alpha=0.5,linewidth=0)
ax.plot(x, y, color='cornflowerblue', label='mid pupil',linewidth=0.5)
ax.set_xscale('log')
ax.set_xlim([0.8, 500])

x = frequency_spectra[0][1:].copy()
y = grandAvg_binned_spec[2,1:].copy()
yerr = grandSem_binned_spec[2,1:].copy()
ax.fill_between(x, y-yerr, y+yerr, where=None, color='lightsteelblue', alpha=0.5,linewidth=0)
ax.plot(x, y, color='lightsteelblue', label='large pupil',linewidth=0.5)    
ax.set_xscale('log')
ax.set_xlim([0.8, 500])

ax.set_xlabel('freq. [Hz]')
ax.set_ylabel('norm. power')
ax.set_xscale('log')
ax.legend(loc='upper right', fontsize=5, frameon=False)
    
plt.savefig(fig_path + '%s.pdf' % (figID1), bbox_inches='tight', pad_inches=0, transparent=True)


#%% cell and session averaged low frequency power vs pupil

plt.rcParams.update({'font.size': 7})
fig = plt.figure(figsize=(1.,1.))  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 

x = pupilSize_binCenters.copy() * 100
y = avg_quantity_vs_pupilSize_bins_spont.copy()
yerr = sem_quantity_vs_pupilSize_bins_spont.copy()
xerr = pupilSize_binStep*100/2

ax.errorbar(x, y, yerr=yerr, xerr=xerr, color='k', linewidth=0.75, fmt='none')
ax.plot(x, y, '-',  color='k', markersize=1, linewidth=0.5)
ax.plot(x, y, 'o',  color=np.array([175,54,60])/256, markersize=1)

ax.set_xticks([0,100])
ax.set_yticks([0.8, 1.4, 2.0])

ax.set_xlim([-2,102])
ax.set_xlabel('pupil diameter\n[% max]')
ax.set_ylabel('$\\langle P_{L,spont} \\rangle$')
plt.savefig(fig_path + '%s.pdf' % (figID2), bbox_inches='tight', pad_inches=0, transparent=True)



#%% for good sessions, plot change in low freq power (low pupil - high pupil)

colorPlot = 'dimgrey'
bin_width = 0.2
outlier_cutoff = 3

hist_data = (lowPupil_lowFreq_power_goodSessions - highPupil_lowFreq_power_goodSessions)
hist_data = hist_data[~np.isnan(hist_data)]
mean_data = np.mean(hist_data)

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
ax.plot(mean_data, np.max(counts)+2, 'v', color='r', markersize=1)
ax.plot([],[],label='p < 0.001')

ax.arrow(-outlier_cutoff-bin_width/2, 8, 0, -5, color='k', linewidth=0.5, head_width=0.02, head_length=0.08)
ax.arrow(outlier_cutoff+bin_width/2, 8, 0, -5, color='k', linewidth=0.5, head_width=0.02, head_length=0.08)
ax.text(-outlier_cutoff - bin_width, 10, ('< -%d' % outlier_cutoff), fontsize=4)
ax.text(outlier_cutoff - bin_width, 10, ('> %d' % outlier_cutoff), fontsize=4)

ax.set_xticks([-outlier_cutoff, 0, outlier_cutoff])
ax.set_xlim([-outlier_cutoff-1.5*bin_width, outlier_cutoff+1.5*bin_width])
ax.set_xlabel('$P_{L,spont}^{small} - P_{L,spont}^{large}$')
ax.set_ylabel('cell count')
plt.savefig(fig_path + '%s.pdf' % (figID3), bbox_inches='tight', pad_inches=0, transparent=True)


#%% for good sessions, plot low freq power low pupil vs high pupil 
    
plt.rcParams.update({'font.size': 7})
fig = plt.figure(figsize=(1.,1.))  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 
    
x = lowPupil_lowFreq_power_goodSessions.copy()
y = highPupil_lowFreq_power_goodSessions.copy()
 
min_data = np.nanmin(np.append(x,y))
max_data = np.nanmax(np.append(x,y))

ax.plot(x, y, 'o', color='black', markersize=0.5, alpha=0.3)
ax.plot([min_data, max_data], [min_data, max_data], color='teal', linewidth=0.5)
ax.plot(np.nanmean(x), np.nanmean(y), color='r', marker='d', markersize=1)
ax.axis('equal')
ax.set_xlabel('small pupil')
ax.set_ylabel('large pupil')
ax.set_xticks([0., 4, 8])
ax.set_yticks([0., 4, 8])
plt.savefig(fig_path + '%s.pdf' % (figID4), bbox_inches='tight', pad_inches=0, transparent=True)


plt.rcParams.update({'font.size': 7})
fig = plt.figure(figsize=(1.,1.))  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 

ax.bar(0, np.nanmean(x), width=0.6, color='dimgrey')
ax.errorbar(0, np.nanmean(x), yerr=scipy.stats.sem(x, nan_policy='omit'), color='r')

ax.bar(1, np.nanmean(y), width=0.6, color='dimgrey')
ax.errorbar(1, np.nanmean(y), yerr=scipy.stats.sem(y,nan_policy='omit'), color='r')
ax.set_ylabel('$\\langle P_{L,spont} \\rangle$')
ax.set_xticks([0,1])
ax.set_xticklabels(['small\npupil', 'large\npupil'])
plt.savefig(fig_path + '%s.pdf' % (figID5), bbox_inches='tight', pad_inches=0, transparent=True)

plt.close('all')


