
'''
This script generates
    FigS6D
    FigS6E
    FigS6F
    FigS6G
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

#%% load my functions

func_path0 = global_settings.path_to_src_code + 'data_analysis/fanofactor_vs_pupil/'
func_path1 = global_settings.path_to_src_code + 'data_analysis/'
func_path2 = global_settings.path_to_src_code + 'functions/'

sys.path.append(func_path0)        
import fcn_plot_fanofactor

sys.path.append(func_path1)
from fcn_SuData_analysis import fcn_cellAvg_sessionAvg_quantity_vs_pupilSize_bins_alt
    
sys.path.append(func_path2)
from fcn_statistics import fcn_Wilcoxon


#%% settings

# window lengths
window_length = 2500e-3
avgRate_thresh = 1

# pupil binning
nPupil_bins = 10
lowPupil_thresh = 0.33
highPupil_thresh = 0.67
pupilSize_binStep = 0.1 

# other
binType = 'avg' # avg or percentile

# sessions to analyze
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
    
    
data_path = global_settings.path_to_data_analysis_output + 'isiCV_pupil/'
fig_path = global_settings.path_to_manuscript_figs_final + 'cvISI_data/'
figname = ('windL%0.3f_rateThresh%0.1fHz_' % (window_length, avgRate_thresh))
cv_filename =  ('spont_cvISI_pupilPercentile_raw')

fig1ID = 'FigS6D'
fig2ID = 'FigS6F'
fig3ID = 'FigS6G'
fig4ID = 'FigS6E'


#%% make figure directory
    
if os.path.isdir(fig_path) == False:
    os.makedirs(fig_path)


#%% initialize 

# pupil bin centers
pupilBin_centers_raw = np.zeros((len(sessions_to_run), nPupil_bins))
nTrials_subsample = np.zeros((len(sessions_to_run)))

# cv isi
cvISI_spont_goodCells = np.ones((len(sessions_to_run), nPupil_bins),dtype='object')
cellAvg_cvISI_spont_goodCells  = np.ones((len(sessions_to_run), nPupil_bins))
cellSem_cvISI_spont_goodCells  = np.ones((len(sessions_to_run), nPupil_bins))

# good sessions
goodSessions = np.array([])

sum_cells = np.array([])

#%% load the data
for count, session_name in enumerate(sessions_to_run):
        

    # load the data 
    load_filename = (data_path + '%s_%s_windLength%0.3fs_.mat' %  (cv_filename, sessions_to_run[count], window_length))        
    cv_data = loadmat(load_filename, simplify_cells=True)
        
    # pupil bin centers 
    if binType == 'avg':
        pupilBin_centers_raw[count, :] = cv_data['avg_pupilSize_spontTrials'].copy()
    elif binType == 'percentile':
        pupilBin_centers_raw[count, :] = cv_data['pupilBin_centers'].copy()
    else:
        sys.exit()

    ### determine if this is a good session
    if ( (np.min(pupilBin_centers_raw[count, :]) <= lowPupil_thresh) and (np.max(pupilBin_centers_raw[count, :]) >= highPupil_thresh)) :
        
        goodSessions = np.append(goodSessions, count)
    
    # number of trials
    nTrials_subsample[count] = cv_data['nTrials_subsample']
    
    # cells that pass rate cut
    min_spikeCount_allPupil = np.min(cv_data['spont_trialAvg_spikeCount'], 1)
    min_rate_allPupil = min_spikeCount_allPupil/window_length
    cells_pass_rate_cut = np.nonzero( min_rate_allPupil >= avgRate_thresh )[0]

    # spont isi cv
    spont_cv_isi = cv_data['spont_cv_isi_trialAggreate'].copy()
    
    # number of cells and pupil bins
    n_Cells = np.size(spont_cv_isi,0)
    n_pupils = np.size(spont_cv_isi, 1)    

    # bad cells
    badRate_cells = np.setdiff1d(np.arange(n_Cells), cells_pass_rate_cut)
    nanCells = badRate_cells.copy()
    
    # loop over pupil bins and store cv isi
    # keep track of problematic cells
    nanCells_additional = np.array([])
    for indPupil in range(0, n_pupils):
    
        cvISI_spont_goodCells[count, indPupil] = spont_cv_isi[:, indPupil].copy()
        nanCells_additional = np.append(nanCells_additional, np.nonzero(np.isnan(cvISI_spont_goodCells[count, indPupil]))[0])
    
    nanCells = np.append(nanCells, nanCells_additional)
    nanCells = np.unique(nanCells).astype(int)
    
    ##### compute average over cells
    for indPupil in range(0, n_pupils):
        cvISI_spont_goodCells[count, indPupil][nanCells] = np.nan
        cellAvg_cvISI_spont_goodCells[count, indPupil] = np.nanmean(cvISI_spont_goodCells[count, indPupil])
        cellSem_cvISI_spont_goodCells[count, indPupil] = scipy.stats.sem(cvISI_spont_goodCells[count, indPupil], nan_policy='omit')

    sum_cells = np.append(sum_cells, len(np.setdiff1d(np.arange(0,n_Cells), nanCells)))

goodSessions = goodSessions.astype(int)


#%% low vs high pupil -- single cell level

lowPupil_cvISI_goodSessions_spont, highPupil_cvISI_goodSessions_spont = \
    fcn_plot_fanofactor.fcn_combine_vecQuantity_low_high_pupil_overSessions(cvISI_spont_goodCells[goodSessions, :])

low_vs_high_stats_goodSessions_spont = fcn_Wilcoxon(lowPupil_cvISI_goodSessions_spont, highPupil_cvISI_goodSessions_spont)

#%% cell and session average spont

pupilSize_binCenters_spont, avg_quantity_vs_pupilSize_bins_spont, std_quantity_vs_pupilSize_bins_spont, sem_quantity_vs_pupilSize_bins_spont, \
    allSessions_quantity_pupilBins_spont, pupilSize_data_in_pupilBins_allSessions_spont = \
    fcn_cellAvg_sessionAvg_quantity_vs_pupilSize_bins_alt(pupilBin_centers_raw, 0, 1, pupilSize_binStep, cvISI_spont_goodCells, 1)



#%% print

print('*'*100)
print('*'*100)

print('good sessions: ', goodSessions)
print('stats:', low_vs_high_stats_goodSessions_spont)
print('mean_diff_spont:', np.nanmean(lowPupil_cvISI_goodSessions_spont - highPupil_cvISI_goodSessions_spont))
print('low pupil mean +/- sem %0.5f +/ %0.5f' % (np.nanmean(lowPupil_cvISI_goodSessions_spont), scipy.stats.sem(lowPupil_cvISI_goodSessions_spont, nan_policy='omit')))
print('high pupil mean +/- sem %0.5f +/ %0.5f' % (np.nanmean(highPupil_cvISI_goodSessions_spont), scipy.stats.sem(highPupil_cvISI_goodSessions_spont, nan_policy='omit')))

print('*'*100)
print('*'*100)

  
#%% save parameters and results

params = {}
results = {}

params['window_length'] = window_length
params['avgRate_thresh'] = avgRate_thresh
params['nPupil_bins'] = nPupil_bins
params['lowPupil_thresh'] = lowPupil_thresh
params['highPupil_thresh'] = highPupil_thresh
params['pupilSize_binStep'] = pupilSize_binStep
params['binType'] = binType
params['sessions_to_run'] = sessions_to_run
params['data_path'] = data_path
params['fig_path'] = fig_path
params['cv_filename'] = cv_filename
params['bin_type'] = binType
results['params'] = params
results['goodSessions'] = goodSessions
results['low_vs_high_stats_goodSessions_spont'] = low_vs_high_stats_goodSessions_spont

save_filename = (fig_path + 'cvISI_data_' + 'results.mat')      
savemat(save_filename, results) 


#%% plotting


#%% cell and session averaged 

plt.rcParams.update({'font.size': 7})
fig = plt.figure(figsize=(1.,1.))  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 

x = pupilSize_binCenters_spont.copy() * 100
y = avg_quantity_vs_pupilSize_bins_spont.copy()
yerr = sem_quantity_vs_pupilSize_bins_spont.copy()
xerr = pupilSize_binStep*100/2

ax.errorbar(x, y, yerr=yerr, xerr=xerr, color='k', linewidth=0.75, fmt='none')
ax.plot(x, y, '-',  color='k', markersize=1, linewidth=0.5)
ax.plot(x, y, 'o',  color=np.array([175,54,60])/256, markersize=1)

ax.set_xticks([0,100])
ax.set_xlim([-2,102])
ax.set_yticks([1.1, 1.5])
ax.set_xlabel('pupil diameter\n[% max]')
ax.set_ylabel('$\\langle cvISI_{spont} \\rangle$')
plt.savefig(fig_path + '%s.pdf' % (fig1ID), bbox_inches='tight', pad_inches=0, transparent=True)


#%% for good sessions, plot low pupil vs high pupil
    
plt.rcParams.update({'font.size': 7})
fig = plt.figure(figsize=(1.,1.))  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 
    
x = lowPupil_cvISI_goodSessions_spont.copy()
y = highPupil_cvISI_goodSessions_spont.copy()
 
min_data = np.nanmin(np.append(x,y))
max_data = np.nanmax(np.append(x,y))

ax.plot(x, y, 'o', color='black', markersize=0.5, alpha=0.3)
ax.plot([min_data, max_data], [min_data, max_data], color='teal', linewidth=0.5)
ax.plot(np.nanmean(x), np.nanmean(y), color='r', marker='d', markersize=1)
ax.axis('equal')
ax.set_xlabel('small pupil')
ax.set_ylabel('large pupil')
    
plt.savefig(fig_path + '%s.pdf' % (fig2ID), bbox_inches='tight', pad_inches=0, transparent=True)


plt.rcParams.update({'font.size': 7})
fig = plt.figure(figsize=(1.,1.))  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 

ax.bar(0, np.nanmean(x), width=0.6, color='dimgrey')
ax.errorbar(0, np.nanmean(x), yerr=scipy.stats.sem(x, nan_policy='omit'), color='r')

ax.bar(1, np.nanmean(y), width=0.6, color='dimgrey')
ax.errorbar(1, np.nanmean(y), yerr=scipy.stats.sem(y,nan_policy='omit'), color='r')
ax.set_ylabel('$\\langle cvISI_{spont} \\rangle$')
ax.set_xticks([0,1])
ax.set_xticklabels(['small\npupil', 'large\npupil'])
plt.savefig(fig_path + '%s.pdf' % (fig3ID), bbox_inches='tight', pad_inches=0, transparent=True)

plt.close('all')


#%% for good sessions, plot (low pupil - high pupil) -- no overflow

colorPlot = 'dimgrey'
bin_width = 0.2

hist_data = (lowPupil_cvISI_goodSessions_spont - highPupil_cvISI_goodSessions_spont)
mean_hist_data = np.nanmean(hist_data)

# remove nan
hist_data = hist_data[~np.isnan(hist_data)]
# bins
data_extreme = np.nanmax(np.abs(hist_data))
bins = np.arange( np.round(-data_extreme - 1*bin_width, 1), np.round(data_extreme + 2*bin_width, 1), bin_width )
counts, bin_edges = np.histogram(hist_data, bins)
bin_centers = bin_edges[:-1] + bin_width/2

plt.rcParams.update({'font.size': 7})
fig = plt.figure(figsize=(1.,1.))  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 
ax.tick_params(axis='both', width=0.5)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.bar(bin_centers, counts, width=np.diff(bin_edges), color=colorPlot)
ax.plot([0,0], [0,np.nanmax(counts)], color='teal', linewidth=1)
ax.plot(mean_hist_data, np.max(counts)+2, 'v', color='r', markersize=1)

ax.set_xlim([bin_centers[0],bin_centers[-1]])
ax.set_xlabel('$cvISI_{spont}^{small} - cvISI_{spont}^{large}$')
ax.set_ylabel('cell count')
plt.savefig(fig_path + '%s.pdf' % (fig4ID), bbox_inches='tight', pad_inches=0, transparent=True)

