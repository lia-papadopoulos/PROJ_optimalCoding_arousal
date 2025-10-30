
'''
This script generates different figure panels depending on the value of cellSelection

    cellSelection = ''
        Fig2C
        Fig2F
        Fig2G
        FigS1A
    
    cellSelection = '_spkTemplate_soundResp_cellSelection1'
        FigS8A
        FigS8B

'''


#%% basic imports
import sys        
import numpy as np
import scipy.stats
from scipy.io import loadmat
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

func_path0 = global_settings.path_to_src_code + 'functions/'
func_path1 = global_settings.path_to_src_code + 'data_analysis/'

sys.path.append(func_path0)
from fcn_statistics import fcn_pctChange_max
from fcn_statistics import fcn_Wilcoxon

sys.path.append(func_path1)
from fcn_SuData_analysis import fcn_sessionAvg_quantity_vs_pupilSize_bins_alt


#%% settings

# data sets
cellSelection = ''

# sessions to plot
sessions_to_run = np.array([\
                   'LA3_session3', \
                   'LA8_session1', \
                   'LA8_session2', \
                   'LA9_session1', \
                   'LA9_session3', \
                   'LA9_session4', \
                   'LA9_session5', \
                   'LA11_session1', 
                   'LA11_session2', \
                   'LA11_session3', \
                   'LA11_session4', \
                   'LA12_session1', \
                   'LA12_session2', \
                   'LA12_session3', \
                   'LA12_session4'
                    ])
        
# example session
example_session = 'LA12_session1'

# pupil diameter binning 
pupilSize_binStep = 0.1
minPupil = 0
maxPupil = 1

# min and max pupil thresh
lowPupil_thresh = 0.33
highPupil_thresh = 0.67

# dprime
window_length = 100e-3
nPercentile_bins = 10
rate_thresh = 0.
data_path = global_settings.path_to_data_analysis_output + 'singleCell_dPrime_pupil/'
outpath = global_settings.path_to_manuscript_figs_final + 'dprime_data/'
 
# update figure path
if cellSelection == '':
    outpath = outpath + 'original_cellSelection/'
elif cellSelection == '_spkTemplate_soundResp_cellSelection1':
    outpath = outpath + 'spkTemplate_soundResp_cellSelection1/'
    

# figure IDs

if cellSelection == '':
    fig1ID = 'Fig2C'
    fig2Id = 'Fig2F'
    fig3ID = 'Fig2G_L'
    fig4ID = 'Fig2G_R'
    fig5ID = 'FigS1A'
    savename = 'dprime_data_'
    
if cellSelection == '_spkTemplate_soundResp_cellSelection1':
    fig2Id = 'FigS8A'
    fig3ID = 'FigS8B_L'
    fig4ID = 'FigS8B_R'
    savename = 'dprime_data_alt_cellSelection'

    
#%% make output directory
if os.path.isdir(outpath) == False:
    os.makedirs(outpath)


#%% paths to data

data_name = '' + cellSelection
fig_name = ('windL%0.3f_rateThresh%0.1fHz%s' % (window_length, rate_thresh, data_name))


#%% initialize 

# pupil bin centers
avg_pupilSize_percentileBlocks = np.zeros((len(sessions_to_run), nPercentile_bins))

# dprime
max_dprime = np.zeros((len(sessions_to_run), nPercentile_bins))
max_dprime_sem = np.zeros((len(sessions_to_run), nPercentile_bins))
norm_max_dPrime = np.zeros((len(sessions_to_run), nPercentile_bins))

freqAvg_dPrime_mid_vs_low = np.ones((len(sessions_to_run), 2), dtype='object')*np.nan
freqAvg_dPrime_mid_vs_high = np.ones((len(sessions_to_run), 2), dtype='object')*np.nan

pupil_at_best_dprime_allSessions = np.zeros((len(sessions_to_run)))
pupil_at_worst_dprime_allSessions = np.zeros((len(sessions_to_run)))

count_lowPupil = 0
count_highPupil = 0


#%% load the data

 
for count, session_name in enumerate(sessions_to_run):

    ###### DPRIME ANALYSES ######   
    
    # load the dprime data 
    load_filename = ( (data_path + 'singleCell_dPrime_pupil_%s_windLength%0.3fs_%s.mat') % (session_name, window_length, data_name))        
    data = loadmat(load_filename, simplify_cells=True)
    
    # pupil size
    avg_pupilSize_percentileBlocks[count, :] = data['avg_pupilSize_dprimeTrials_pupilBlocks'].copy()

    # baseline firing rate
    baseRate = data['freqAvg_sampleAvg_base_rate'].copy()
    
    # cells that pass rate cut
    cells_pass_rate_cut = np.nonzero(np.all(baseRate > rate_thresh, 1))[0]

    # frequency average dprime
    freqAvg_sampleAvg_dprime = data['freqAvg_sampleAvg_dprime'][:, cells_pass_rate_cut, :].copy() # time, cells, pupil

    # average over units
    cellAvg_dprime = np.nanmean(freqAvg_sampleAvg_dprime, axis=1) # time, pupil
    cellSem_dprime = scipy.stats.sem(freqAvg_sampleAvg_dprime, axis=1, nan_policy='omit')
    ind_max_dprime = np.argmax(cellAvg_dprime, axis=0)

    # max across time
    max_dprime[count, :] = np.max(cellAvg_dprime, axis=0)
    for i in range(0, len(ind_max_dprime)):
        max_dprime_sem[count, i] = cellSem_dprime[ind_max_dprime[i], i]

    # normalized dprime
    norm_max_dPrime[count, :] = fcn_pctChange_max(max_dprime[count, :])   

    # pupil at peak accuracy
    pupilInd_at_best_dprime = np.argmax(max_dprime[count,:])
    pupilInd_at_worst_dprime = np.argmin(max_dprime[count,:])

    pupil_at_best_dprime_allSessions[count] = avg_pupilSize_percentileBlocks[count,pupilInd_at_best_dprime].copy()
    pupil_at_worst_dprime_allSessions[count] = avg_pupilSize_percentileBlocks[count,pupilInd_at_worst_dprime].copy()
    
    
    # save dprime for individual units
    if (np.min(avg_pupilSize_percentileBlocks[count, :]) <= lowPupil_thresh):
        
        count_lowPupil += 1
        
        ind_midPupil = np.argmin(np.abs(avg_pupilSize_percentileBlocks[count, :] - 0.5))

        freqAvg_dPrime_mid_vs_low[count,0] = freqAvg_sampleAvg_dprime[ind_max_dprime[ind_midPupil], :, ind_midPupil]
        freqAvg_dPrime_mid_vs_low[count,1] = freqAvg_sampleAvg_dprime[ind_max_dprime[0], :, 0]
                                            

    # save dprime for individual units
    if (np.max(avg_pupilSize_percentileBlocks[count, :]) >= highPupil_thresh):
        
        count_highPupil += 1
        
        ind_midPupil = np.argmin(np.abs(avg_pupilSize_percentileBlocks[count, :] - 0.5))

        freqAvg_dPrime_mid_vs_high[count,0] = freqAvg_sampleAvg_dprime[ind_max_dprime[ind_midPupil], :, ind_midPupil]
        freqAvg_dPrime_mid_vs_high[count,1] = freqAvg_sampleAvg_dprime[ind_max_dprime[-1], :, -1]
                                            

#%% STATISTICS

### compare dprime in middle vs low pupil bin

freqAvg_dPrime_mid_all = np.array([])
freqAvg_dPrime_low_all = np.array([])

for indSession in range(0,len(sessions_to_run)):
    
    freqAvg_dPrime_mid_all = np.append(freqAvg_dPrime_mid_all, freqAvg_dPrime_mid_vs_low[indSession, 0])
    freqAvg_dPrime_low_all = np.append(freqAvg_dPrime_low_all, freqAvg_dPrime_mid_vs_low[indSession, 1])

middlePupil_vs_lowPupil_dprime_stats = fcn_Wilcoxon(freqAvg_dPrime_mid_all, freqAvg_dPrime_low_all)


### compare dprime in middle vs high pupil bin        

freqAvg_dPrime_mid_all = np.array([])
freqAvg_dPrime_high_all = np.array([])

for indSession in range(0,len(sessions_to_run)):
    
    freqAvg_dPrime_mid_all = np.append(freqAvg_dPrime_mid_all, freqAvg_dPrime_mid_vs_high[indSession, 0])
    freqAvg_dPrime_high_all = np.append(freqAvg_dPrime_high_all, freqAvg_dPrime_mid_vs_high[indSession, 1])

middlePupil_vs_highPupil_dprime_stats = fcn_Wilcoxon(freqAvg_dPrime_mid_all, freqAvg_dPrime_high_all)



#%% print results

print('*'*100)
print('*'*100)

print('MIDDLE VS LOW PUPIL')
print(middlePupil_vs_lowPupil_dprime_stats)
print('# sessions:', count_lowPupil)

print('*'*100)
print('*'*100)

print('MIDDLE VS HIGH PUPIL')
print(middlePupil_vs_highPupil_dprime_stats)
print('# sessions:', count_highPupil)

print('*'*100)
print('*'*100)

                                            
#%% average across sessions -- normalized

pupilSize_binCenters, avg_norm_dPrime_vs_pupilSize_bins, std_norm_dPrime_vs_pupilSize_bins, sem_norm_dPrime_vs_pupilSize_bins, \
    allSessions_norm_dPrime_pupilBins, pupilSize_data_in_pupilBins_allSessions = \
    fcn_sessionAvg_quantity_vs_pupilSize_bins_alt(avg_pupilSize_percentileBlocks, minPupil, maxPupil, pupilSize_binStep, norm_max_dPrime, True)


#%% plotting


#%% EXAMPLE SESSION: PEAK DPRIME VS PUPIL SIZE

if cellSelection == '':

    # session index
    indSession = np.nonzero(sessions_to_run == example_session)[0][0]
    
    # plotting
    plt.rcParams.update({'font.size': 7})
    fig = plt.figure(figsize=(0.9, 1.))  
    ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 
        
    x = avg_pupilSize_percentileBlocks[indSession, :]*100
    y = max_dprime[indSession, :]
    ax.plot(x, y, '-o', color='dimgray', markersize=2., linewidth=1)
            
    ax.set_xlim([-2, 102])
    ax.set_yticks([0.2, 0.4])
    ax.set_xlabel('pupil diameter\n[% max]')
    ax.set_ylabel('cell avg. $D^{\prime}_{sc}$')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(('%s%s.pdf' % (outpath, fig1ID)), bbox_inches='tight', pad_inches=0, transparent=True)
    
    

#%% AVERAGE DPRIME MODULATION BOX PLOTS

plt.rcParams.update({'font.size': 7})
if ( (cellSelection != '') ):
    fig = plt.figure(figsize=(1.2,1.2))      
    markerSize1 = 2
    markerSize2 = 2.25
    lineWidth = 1
else:
    fig = plt.figure(figsize=(2.4,2.0))  
    markerSize1 = 3.5
    markerSize2 = 4
    lineWidth = 1.5
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8])

x = pupilSize_binCenters*100
y = allSessions_norm_dPrime_pupilBins.copy()
for binInd in range(0,len(x)):         
    boxPlot_data = y[binInd][~np.isnan(y[binInd])]
    ax.boxplot(boxPlot_data, positions=[round(x[binInd],2)], \
                widths=(pupilSize_binStep/1.1)*100, \
                showfliers=False, \
                patch_artist=False, \
                medianprops=dict(zorder=1,color='black'), \
                boxprops=dict(color='black'))
    x_dataPts = pupilSize_data_in_pupilBins_allSessions[binInd]*100
    y_dataPts = y[binInd].copy()
    ax.plot(x_dataPts, y_dataPts, 'o', markersize=markerSize1, color='black', alpha=0.4)  
x = pupilSize_binCenters*100
y = avg_norm_dPrime_vs_pupilSize_bins
ax.plot(x, y, '-o', markersize=markerSize2, linewidth=lineWidth, color='red', markerfacecolor='red', markeredgecolor='red', alpha=1)

plt.xticks(np.arange(0,120,20), labels=np.arange(0,120,20))
ax.set_xlim([-2, 102])
ax.set_xlabel('pupil diameter [% max]')
ax.set_ylabel('% change in cell avg. $D^{\prime}_{sc}$\n(relative to max)', multialignment='center')
plt.savefig(('%s%s.pdf' % (outpath, fig2Id)), bbox_inches='tight', pad_inches=0, transparent=True)
    

#%% change in Dprime between central and first decile

# with overflow
colorPlot = 'gray'
bin_width = 0.1
outlier_cutoff = 0.5

hist_data = np.array([])
for indSession in range(0,len(sessions_to_run)):
    dprime_mid_minus_low = freqAvg_dPrime_mid_vs_low[indSession,0] - freqAvg_dPrime_mid_vs_low[indSession,1]
    hist_data = np.append(hist_data, dprime_mid_minus_low)
hist_data_mean = np.nanmean(hist_data)
hist_data = hist_data[~np.isnan(hist_data)]
neg_outliersInds = np.nonzero(hist_data < -outlier_cutoff)[0]
neg_outliers = hist_data[neg_outliersInds]
pos_outliersInds =  np.nonzero(hist_data > outlier_cutoff)[0]
pos_outliers = hist_data[pos_outliersInds]
hist_data[neg_outliersInds] = np.nan
hist_data[pos_outliersInds] = np.nan
data_extreme = np.nanmax(np.abs(hist_data))
bins = np.arange( np.round(-outlier_cutoff - bin_width, 1), np.round(outlier_cutoff + 2*bin_width, 1), bin_width )
counts, bin_edges = np.histogram(hist_data, bins)
bin_centers = bin_edges[:-1] + bin_width/2


plt.rcParams.update({'font.size': 7})

if ( (cellSelection != '') ):
    fig = plt.figure(figsize=(0.9,0.9))      
    markerSize = 2
    lineWidth = 1
else:
    fig = plt.figure(figsize=(1.1,1.5))  
    markerSize = 3
    lineWidth = 1

ax = fig.add_axes([0.2, 0.2, 0.8, 0.8])

ax.bar(-outlier_cutoff - bin_width/2, np.size(neg_outliers), width = bin_width, color=colorPlot)
ax.bar(outlier_cutoff + bin_width/2, np.size(pos_outliers), width = bin_width, color=colorPlot)
ax.annotate("", xy=(-outlier_cutoff - bin_width/2, np.size(neg_outliers)+3), xytext=(-outlier_cutoff - bin_width/2, np.size(neg_outliers)+120),
            arrowprops=dict(arrowstyle="->", color="black", lw=0.5))
ax.text(-outlier_cutoff - 1.5*bin_width, np.size(neg_outliers)+140, '< -%0.1f' % outlier_cutoff, fontsize=6)
ax.annotate("", xy=(outlier_cutoff + bin_width/2, np.size(pos_outliers)+3), xytext=(outlier_cutoff + bin_width/2, np.size(pos_outliers)+120),
            arrowprops=dict(arrowstyle="->", color="black", lw=0.5))
ax.text(outlier_cutoff - 0.5*bin_width, np.size(neg_outliers)+140, '> %0.1f' % outlier_cutoff, fontsize=6)

ax.bar(bin_centers, counts, width=np.diff(bin_edges), color=colorPlot)
ax.plot([0,0], [0,np.nanmax(counts)], color='teal', linewidth=lineWidth)
ax.plot(hist_data_mean, np.max(counts)+2, 'v', color='r', markersize=markerSize)
ax.set_xlim([-outlier_cutoff-2*bin_width, outlier_cutoff+2*bin_width])
ax.set_xlabel('change in $D^{\prime}_{sc}$\n(central decile - first decile)', multialignment='center')
ax.set_ylabel('cell count')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig(outpath + fig3ID + '.pdf', bbox_inches='tight', pad_inches=0, transparent=True)


#%% change in Dprime between central and last decile

# with overflow
colorPlot = 'gray'
bin_width = 0.1
outlier_cutoff = 0.5

hist_data = np.array([])
for indSession in range(0,len(sessions_to_run)):
    dprime_mid_minus_low = freqAvg_dPrime_mid_vs_high[indSession,0] - freqAvg_dPrime_mid_vs_high[indSession,1]
    hist_data = np.append(hist_data, dprime_mid_minus_low)
    
hist_data_mean = np.nanmean(hist_data)
hist_data = hist_data[~np.isnan(hist_data)]
neg_outliersInds = np.nonzero(hist_data < -outlier_cutoff)[0]
neg_outliers = hist_data[neg_outliersInds]
pos_outliersInds =  np.nonzero(hist_data > outlier_cutoff)[0]
pos_outliers = hist_data[pos_outliersInds]
hist_data[neg_outliersInds] = np.nan
hist_data[pos_outliersInds] = np.nan
data_extreme = np.nanmax(np.abs(hist_data))
bins = np.arange( np.round(-outlier_cutoff - bin_width, 1), np.round(outlier_cutoff + 2*bin_width, 1), bin_width )
counts, bin_edges = np.histogram(hist_data, bins)
bin_centers = bin_edges[:-1] + bin_width/2

plt.rcParams.update({'font.size': 7})

if ( (cellSelection != '') ):
    fig = plt.figure(figsize=(0.9,0.9))      
    markerSize = 2
    lineWidth = 1
else:
    fig = plt.figure(figsize=(1.1,1.5))  
    markerSize = 3
    lineWidth = 1

ax = fig.add_axes([0.2, 0.2, 0.8, 0.8])

ax.bar(-outlier_cutoff - bin_width/2, np.size(neg_outliers), width = bin_width, color=colorPlot)
ax.bar(outlier_cutoff + bin_width/2, np.size(pos_outliers), width = bin_width, color=colorPlot)
ax.annotate("", xy=(-outlier_cutoff - bin_width/2, np.size(neg_outliers)+3), xytext=(-outlier_cutoff - bin_width/2, np.size(neg_outliers)+170),
            arrowprops=dict(arrowstyle="->", color="black", lw=0.5))
ax.text(-outlier_cutoff - 1.5*bin_width, np.size(neg_outliers)+190, '< -%0.1f' % outlier_cutoff, fontsize=6)
ax.annotate("", xy=(outlier_cutoff + bin_width/2, np.size(pos_outliers)+3), xytext=(outlier_cutoff + bin_width/2, np.size(pos_outliers)+170),
            arrowprops=dict(arrowstyle="->", color="black", lw=0.5))
ax.text(outlier_cutoff - 0.5*bin_width, np.size(neg_outliers)+190, '> %0.1f' % outlier_cutoff, fontsize=6)

ax.bar(bin_centers, counts, width=np.diff(bin_edges), color=colorPlot)
ax.plot([0,0], [0,np.nanmax(counts)], color='teal', linewidth=1)
ax.plot(hist_data_mean, np.max(counts)+2, 'v', color='r', markersize=3)
ax.set_xlim([-outlier_cutoff-2*bin_width, outlier_cutoff+2*bin_width])
ax.set_xlabel('change in $D^{\prime}_{sc}$\n(central decile - last decile)', multialignment='center')
ax.set_ylabel('cell count')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig(outpath + fig4ID + '.pdf', bbox_inches='tight', pad_inches=0, transparent=True)


#%% DISTRIBUTION OF BEST AND WORST ACCURACY ACROSS SESSIONS

if cellSelection == '':

    plt.rcParams.update({'font.size': 7})
    fig = plt.figure(figsize=(1.8,1.8))  
    ax = fig.add_axes([0.2, 0.2, 0.8, 0.8])
    
    
    ax.hist(pupil_at_best_dprime_allSessions*100, np.arange(0,110,10), color='teal', alpha=1, label='best')
    ax.hist(pupil_at_worst_dprime_allSessions*100, np.arange(0,110,10), color='grey', alpha=1, label='worst')
    
    ax.set_xlim([-2, 102])
    ax.set_xlabel('pupil diameter [% max]')
    ax.set_ylabel('number of sessions')
    ax.legend()
    plt.savefig(('%s%s.pdf'  % (outpath, fig5ID)), bbox_inches='tight', pad_inches=0, transparent=True)
