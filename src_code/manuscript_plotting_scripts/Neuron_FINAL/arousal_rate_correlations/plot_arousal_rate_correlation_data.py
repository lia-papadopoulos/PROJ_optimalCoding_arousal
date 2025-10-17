
"""
This script generates
    FigS2A
    FigS2B
    FigS2D
    FigS2E
    FigS2C
"""

#%% basic imports
import sys        
import numpy as np
from scipy.io import loadmat
import os

#%% global settings
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

funcpath = global_settings.path_to_src_code + 'functions/'
sys.path.append(funcpath)
from fcn_statistics import fcn_Wilcoxon

#%% settings

# paths
data_path = global_settings.path_to_data_analysis_output + 'rate_pupil_run_correlations/'
outpath = global_settings.path_to_manuscript_figs_final + 'arousal_rate_correlations_data/'

# sessions to plot
sessions_to_run = [\
                   'LA3_session3', \
                   'LA8_session1', 'LA8_session2', \
                   'LA9_session1', \
                   'LA9_session3', 'LA9_session4', 'LA9_session5', \
                   'LA11_session1', 'LA11_session2', 'LA11_session3', 'LA11_session4', \
                   'LA12_session1', 'LA12_session2', 'LA12_session3', 'LA12_session4' ]

# window length
window_length = 100e-3

# rate thresh
rate_thresh = 0.

# significant cells
sig_level = 0.05

# figure names
fig1ID = 'FigS2A'
fig2ID = 'FigS2B'
fig3ID = 'FigS2D'
fig4ID = 'FigS2E'
fig5ID = 'FigS2C'


#%% setup based on specified parameters

# for loading data
data_name = ''

# end of figure name
fig_name_end = ('_windLength%0.3fs_rateThresh%0.3fHz%s' % (window_length, rate_thresh, data_name))

# make output directory
if os.path.isdir(outpath) == False:
    os.makedirs(outpath)

#%% ANALYSIS


# fraction of cells positively or negatively correlated with pupil
frac_posCorr_pupil_rates = np.zeros(len(sessions_to_run))
frac_negCorr_pupil_rates = np.zeros(len(sessions_to_run))

# correlation between pupil size and spontaneous firing rate
corr_pupil_rates_pctileBins_allSessions = np.array([])
corr_pupil_rates_pctileBins_allSessions_sig = np.array([])
corr_pupil_rates_pctileBins_allSessions_goodPupilRange = np.array([])

# loop over sessions
for indSession in range(0, len(sessions_to_run)):
    
    # session data
    session_name = sessions_to_run[indSession]
    
    # load data
    data = loadmat('%s%s_pupil_run_rate_correlation_spont_windLength%0.3fs%s.mat' % (data_path, session_name, window_length,data_name), simplify_cells=True)

    # firing rates
    avgRate_pupilBins = data['avgRate_pupilBins']
    # pupil bin centers (avg pupil size in each percentile)
    pupil_binCenters = data['avg_pupilSize_inPercentiles']
    # correlations
    corr_pupil_rates_pctileBins = data['corr_pupil_rates_pctileBins'][:,0]
    # pvalues
    pval_corr_pupil_rates_pctileBins = data['corr_pupil_rates_pctileBins'][:,1]
    
    # average rate of each cell across all pupil bins
    avgRate = np.mean(avgRate_pupilBins, 1)

    # low firing cells
    lowRate_cells = np.nonzero(avgRate <= rate_thresh)[0]
    
    # nCells
    nCells = np.size(avgRate)
    
    # update average units
    avg_units = np.setdiff1d(np.arange(0,nCells), lowRate_cells)
        
    # fraction pos/neg correlations
    frac_posCorr_pupil_rates[indSession] = np.size(np.nonzero( (pval_corr_pupil_rates_pctileBins[avg_units] < sig_level) & (corr_pupil_rates_pctileBins[avg_units] > 0) )[0])/np.size(avg_units)
    frac_negCorr_pupil_rates[indSession] = np.size(np.nonzero( (pval_corr_pupil_rates_pctileBins[avg_units] < sig_level) & (corr_pupil_rates_pctileBins[avg_units] < 0) )[0])/np.size(avg_units)

    # correlation across all sessions
    corr_pupil_rates_pctileBins_allSessions = np.append(corr_pupil_rates_pctileBins_allSessions, corr_pupil_rates_pctileBins[avg_units])
    corr_pupil_rates_pctileBins_allSessions_sig = np.append(corr_pupil_rates_pctileBins_allSessions_sig, corr_pupil_rates_pctileBins[avg_units][pval_corr_pupil_rates_pctileBins[avg_units] < sig_level])
        

#%% STATISTICS

# COMPARE FRACTION EXCITED AND INHIBITED FOR ALL SESSIONS
all_session_stats = fcn_Wilcoxon(frac_posCorr_pupil_rates, frac_negCorr_pupil_rates)
print(all_session_stats)


#%% EXAMPLE SESSION: POSITIVELY + NEGATIVELY CORRELATED CELL

# session data
session_name = 'LA12_session1'
pos_cellPlot = 18
neg_cellPlot = 7

# data
data = loadmat('%s%s_pupil_run_rate_correlation_spont_windLength%0.3fs%s.mat' % (data_path, session_name, window_length,data_name), simplify_cells=True)

# unpack
avgRate_pupilBins = data['avgRate_pupilBins']
pupil_binCenters = data['avg_pupilSize_inPercentiles']
corr_pupil_rates_pctileBins = data['corr_pupil_rates_pctileBins'][:,0]
pval_corr_pupil_rates_pctileBins = data['corr_pupil_rates_pctileBins'][:,1]

# sort by correlation
pos_corr = np.nonzero( (pval_corr_pupil_rates_pctileBins < sig_level) & (corr_pupil_rates_pctileBins > 0) )[0]
neg_corr = np.nonzero( (pval_corr_pupil_rates_pctileBins < sig_level) & (corr_pupil_rates_pctileBins < 0) )[0]

# cells to plot
pos_cellPlot = pos_corr[pos_cellPlot]
neg_cellPlot = neg_corr[neg_cellPlot]

# plot postive cell
plt.rcParams.update({'font.size': 8})
fig = plt.figure(figsize=(1.75,1.75))  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 
corr_val = corr_pupil_rates_pctileBins[pos_cellPlot]
p_val = pval_corr_pupil_rates_pctileBins[pos_cellPlot]
ax.plot(pupil_binCenters*100, avgRate_pupilBins[pos_cellPlot,:], '-o', color='lightseagreen', markersize=3)
ax.text(10, 11, (r'$\rho = %0.2f$' % corr_val), fontsize=7)
ax.text(10, 10.2, ('p < 0.01'), fontsize=7)
ax.set_xlim([-2, 102])
ax.set_xlabel('pupil diameter\n[% max]')
ax.set_ylabel('avg. rate [spk/s]')
ax.set_title('positive correlation', fontsize=8)
plt.savefig( (outpath + '%s.pdf') % (fig1ID), bbox_inches='tight', pad_inches=0, transparent=True)

# plot negative cell
plt.rcParams.update({'font.size': 8})
fig = plt.figure(figsize=(1.75,1.75))  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 
corr_val = corr_pupil_rates_pctileBins[neg_cellPlot]
p_val = pval_corr_pupil_rates_pctileBins[neg_cellPlot]
ax.plot(pupil_binCenters*100, avgRate_pupilBins[neg_cellPlot,:], '-o', color='darkviolet', markersize=3)
ax.text(10, 3.8, (r'$\rho = %0.2f$' % corr_val), fontsize=7)
ax.text(10, 3.2, ('p < 0.01'), fontsize=7)
ax.set_xlim([-2, 102])
ax.set_xlabel('pupil diameter\n[% max]')
ax.set_ylabel('avg. rate [spk/s]')
ax.set_title('negative correlation', fontsize=8)
plt.savefig( (outpath + '%s.pdf') % (fig2ID), bbox_inches='tight', pad_inches=0, transparent=True)



#%% SESSION AVERAGE OF % OF CELLS POSITIVELY/NEGATIVELY CORRELATED WITH PUPIL [ALL SESSIONS]

plt.rcParams.update({'font.size': 8})
fig = plt.figure(figsize=(1.1,1.75))  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 

# pos
y = np.mean(frac_posCorr_pupil_rates*100)
yerr = np.std(frac_posCorr_pupil_rates*100)
ax.bar(0, y, yerr=yerr, color='lightseagreen')

# neg
y = np.mean(frac_negCorr_pupil_rates*100)
yerr = np.std(frac_negCorr_pupil_rates*100)
ax.bar(1, y, yerr=yerr, color='darkviolet')

# labels
ax.set_xlim([-0.5, 1.5])
ax.set_xticks([0,1])
ax.set_xticklabels(['pos. corr.', 'neg. corr.'], rotation=45)
ax.set_ylim([0, 50])
ax.set_yticks([0, 25, 50])
ax.set_ylabel('percent of cells')

# save
plt.savefig( (outpath + '%s.pdf') % (fig3ID), bbox_inches='tight', pad_inches=0, transparent=True)


#%% FRACTION OF POSITIVELY AND NEGATIVELY CORRELATION CELLS IN EACH SESSION

plt.rcParams.update({'font.size': 8})
fig = plt.figure(figsize=(4.5,1.75))  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 


for indSession in range(0, len(sessions_to_run)):

    y = frac_posCorr_pupil_rates[indSession]*100
    if indSession == len(sessions_to_run)-1:
        ax.bar(indSession, y, width=0.3, color='lightseagreen', label='pos. corr.')
    else:
        ax.bar(indSession, y, width=0.3, color='lightseagreen')

    y = frac_negCorr_pupil_rates[indSession]*100
    if indSession == len(sessions_to_run)-1:
        plt.bar(indSession+0.3, y, width=0.3, color='darkviolet', label='neg. corr.')
    else:
        plt.bar(indSession+0.3, y, width=0.3, color='darkviolet')


ax.set_ylim([0, 50])
ax.set_yticks([0, 25, 50])
ax.set_xticks(np.arange(0.16, len(sessions_to_run)+0.16))
ax.set_xticklabels(np.arange(1, len(sessions_to_run)+1))
ax.set_xlabel('session')
ax.set_ylabel('percent of cells')
ax.legend(fontsize=6, loc='upper left')
plt.savefig( (outpath + '%s.pdf') % (fig4ID), bbox_inches='tight', pad_inches=0, transparent=True)


#%% PLOT DISTRIBUTION OF CORRELATIONS BETWEEN FIRING RATE AND PUPIL ACROSS ALL SESSIONS

# all cells
plt.rcParams.update({'font.size': 8})
fig = plt.figure(figsize=(1.75,1.75))  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 
ax.hist(corr_pupil_rates_pctileBins_allSessions, facecolor='dimgrey')
ax.set_xlim([-1, 1])
ax.set_xlabel('Spearman correlation\nspont. rate vs. pupil diameter', multialignment='center')
ax.set_title('all cells')
ax.set_ylabel('cell count')    
plt.savefig( (outpath + '%s.pdf') % (fig5ID), bbox_inches='tight', pad_inches=0, transparent=True)

