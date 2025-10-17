
'''
This script generates various figure panels depending on the values of certain settings

    cellSelection = ''
    global_pupilNorm = False
    rest_only = False
        FigS1B
        Fig2E
        Fig2H
        Fig2I
    
    cellSelection = ''
    global_pupilNorm = False
    rest_only = True
        FigS1C
        FigS1D
    
    cellSelection = ''
    global_pupilNorm = True
    rest_only = False
        FigS1F
        FigS1G
    
    cellSelection = '_spkTemplate_soundResp_cellSelection1'
    global_pupilNorm = False
    rest_only = False
        FigS8C
        FigS8D

'''

#%% basic imports
import sys        
import numpy as np
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
font_path = '/home/liap/fonts/Arial.ttf'  # Your font path goes here
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()
plt.rcParams["mathtext.default"]="regular"
plt.rcParams['axes.linewidth'] = 0.5

#%% load my functions

func_path0 = global_settings.path_to_src_code + 'functions/'
sys.path.append(func_path0)
from fcn_statistics import fcn_pctChange_max
from fcn_statistics import fcn_Wilcoxon
func_path1 = global_settings.path_to_src_code + 'data_analysis/'
sys.path.append(func_path1)
from fcn_SuData_analysis import fcn_sessionAvg_quantity_vs_pupilSize_bins_alt

#%% settings

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

# analysis information  
pupilSize_binStep = 0.1

# decoding
pupilSplit_method = 'percentile'
pupilBlock_size = 0.1
pupilBlock_step = 0.1
nFreqs = 5
n_decodeReps = 10
windSize = 100e-3
windStep = 10e-3
decoderType = 'LinearSVC'
crossvalType = 'repeated_stratified_kFold'

# min and max pupil thresh
lowPupil_thresh = 0.33
highPupil_thresh = 0.67

# set paths
decoding_path = global_settings.path_to_data_analysis_output + 'decoding_pupil/'
outpath = global_settings.path_to_manuscript_figs_final + 'decoding_data/'
 
# data params
cellSelection = '_spkTemplate_soundResp_cellSelection1'
global_pupilNorm = False
rest_only = False

# update figure path
if cellSelection == '':
    outpath = outpath + 'original_cellSelection/'
elif cellSelection == '_spkTemplate_soundResp_cellSelection1':
    outpath = outpath + 'spkTemplate_soundResp_cellSelection1/'
   
if global_pupilNorm:
    outpath = outpath + 'global_pupilNorm/'
    
if rest_only:
    outpath = outpath + 'rest_only/'


# figure ids
if ( (cellSelection == '') and  (global_pupilNorm == False) and (rest_only == False) ):
    fig1ID = 'Fig2E'
    fig2ID = 'Fig2H'
    fig3ID = 'Fig2I_L'
    fig4ID = 'Fig2I_R'
    fig5ID = 'FigS1B'


if ( (cellSelection == '') and  (global_pupilNorm == False) and (rest_only == True) ):
    fig2ID = 'FigS1C'
    fig3ID = 'FigS1D_L'
    fig4ID = 'FigS1D_R'


if ( (cellSelection == '') and  (global_pupilNorm == True) and (rest_only == False) ):
    fig2ID = 'FigS1F'
    fig3ID = 'FigS1G_L'
    fig4ID = 'FigS1G_R'


if ( (cellSelection == '_spkTemplate_soundResp_cellSelection1') and  (global_pupilNorm == False) and (rest_only == False) ):
    fig2ID = 'FigS8C'
    fig3ID = 'FigS8D_L'
    fig4ID = 'FigS8D_R'


#%% make output directory ###

if os.path.isdir(outpath) == False:
    os.makedirs(outpath)


#%% file names

# filename
global_pupilNorm_str = '' + cellSelection + '_globalPupilNorm'*global_pupilNorm

      
if rest_only:   
    
    fname_end = (('_sweep_pupilSize_pupilSplit%s_pupilBlock%0.3f_pupilStep%0.3f_windSize%0.3fs_windStep%0.3fs_decoder%s_crossVal%s_nFreqs%d_restOnly%s_') % \
                 (pupilSplit_method, pupilBlock_size, pupilBlock_step, windSize, windStep, decoderType, crossvalType, nFreqs, global_pupilNorm_str))                  
else:
      
    fname_end = (('_sweep_pupilSize_pupilSplit%s_pupilBlock%0.3f_pupilStep%0.3f_windSize%0.3fs_windStep%0.3fs_decoder%s_crossVal%s_nFreqs%d%s_') % \
                 (pupilSplit_method, pupilBlock_size, pupilBlock_step, windSize, windStep, decoderType, crossvalType, nFreqs, global_pupilNorm_str)) 
            

#%% figure names

if rest_only:
    
    fig_name = (('sweep_pupilSize_decoder%s_crossVal%s_nFreqs%d_restOnly%s') % (decoderType, crossvalType, nFreqs, global_pupilNorm_str))
    
else:
    
    fig_name = (('sweep_pupilSize_decoder%s_crossVal%s_nFreqs%d%s') % (decoderType, crossvalType, nFreqs, global_pupilNorm_str))
    
                
#%% initialize data
    
nSessions = len(sessions_to_run)
n_pupilBlocks = int((1 - pupilBlock_size)/pupilBlock_step + 1)

nBlocks = n_pupilBlocks

pupilSize_bins = np.arange(0, 1 + pupilSize_binStep, pupilSize_binStep)
n_pupilBins = len(pupilSize_bins)-1

avg_pupilSize_pupilBlocks = np.zeros((nSessions, n_pupilBlocks))
avg_pupilSize_allBlocks = np.zeros((nSessions, nBlocks))

t_peak_repAvg_accuracy_allSessions = np.zeros((nSessions, nBlocks))
peak_repAvg_accuracy_allSessions = np.zeros((nSessions, nBlocks))
peak_repAvg_accuracy_modulation = np.zeros((nSessions, nBlocks))

peak_repAvg_accuracy_mid_vs_low = np.ones((2), dtype='object')*np.nan
peak_repAvg_accuracy_mid_vs_high = np.ones((2), dtype='object')*np.nan

pupil_at_peak_accuracy_allSessions = np.zeros((nSessions))
pupil_at_worst_accuracy_allSessions = np.zeros((nSessions))

repAvg_accuracy_vs_time_allSessions = np.zeros((nSessions), dtype='object')


#%% loop over all sessions

for count, session_name in enumerate(sessions_to_run):
        
    # load decoding data
    decoding_filename = ((decoding_path + 'decode_toneFreq_session%s' + fname_end + 'rep%d.mat') % (session_name, 0)) 
    decoding_data = loadmat(decoding_filename, simplify_cells=True)
    
    # get relevant decoding info   
    t_decoding = decoding_data['t_decoding'][0]
    t_decoding_run = decoding_data['t_decoding_run']
    pupilSize_percentileBlocks = decoding_data['pupilSize_percentileBlocks']
    decode_freqs = decoding_data['params']['freqVals']

    # number of time points
    n_tPts = np.size(t_decoding)
    
    # initialize decoding quantities for each decoding rep
    accuracy_pupil = np.zeros((n_pupilBlocks, n_tPts, n_decodeReps))
    accuracyZscore_pupil = np.zeros((n_pupilBlocks, n_tPts, n_decodeReps))
    avg_pupilSize_decodingTrials_pupilBlocks = np.zeros((n_pupilBlocks, n_decodeReps))
    
    # loop over decoding repetitions
    for repInd in range(0, n_decodeReps):
        
        # load decoding data
        decoding_filename = ((decoding_path + 'decode_toneFreq_session%s' + fname_end + 'rep%d.mat') % (session_name, repInd)) 
        decoding_data = loadmat(decoding_filename, simplify_cells=True)
        
        # average pupil size used for decoding in each pupil bin
        if n_decodeReps == 1:
            avg_pupilSize_decodingTrials_pupilBlocks[:, repInd] = decoding_data['avg_pupilSize_decodingTrials_pupilBlocks']
        else:
            avg_pupilSize_decodingTrials_pupilBlocks[:, repInd] = decoding_data['avg_pupilSize_decodingTrials_pupilBlocks'][:, repInd]

        # loop over pupils
        for blockInd in range(0, n_pupilBlocks):
            accuracy_pupil[blockInd, :, repInd] = decoding_data['accuracy'][blockInd]
            
    # repetition average of accuracy
    repAvg_accuracy_pupil = np.mean(accuracy_pupil, 2)
    
    # peak of repetition average accuracy
    t_peak_repAvg_accuracy_allSessions[count, :] = t_decoding[np.argmax(np.mean(accuracy_pupil, 2), 1)]
    peak_repAvg_accuracy_pupil = np.max(np.mean(accuracy_pupil, 2), 1)   
            
    peak_repAvg_accuracy = peak_repAvg_accuracy_pupil.copy()
    repAvg_accuracy = repAvg_accuracy_pupil.copy()
    
    # repetition average of mean pupil size of decoding trials
    repAvg_avg_pupilSize_decodingTrials_pupilBlocks = np.mean(avg_pupilSize_decodingTrials_pupilBlocks, axis=1)
    avg_pupilSize_percentileBlocks = np.mean(pupilSize_percentileBlocks,0)

    # STORE DATA FOR ALL SESSIONS

    # rep average of accuracy vs time
    repAvg_accuracy_vs_time_allSessions[count] = repAvg_accuracy

    # peak rep average accuracy all sessions
    peak_repAvg_accuracy_allSessions[count, :] = peak_repAvg_accuracy
    
    # modulation of peak of rep avg accuracy        
    peak_repAvg_accuracy_modulation[count, :] = fcn_pctChange_max(peak_repAvg_accuracy)
    
    # pupil size (% max) corresponding to middle of pupil percentile bin
    avg_pupilSize_pupilBlocks[count, :] = repAvg_avg_pupilSize_decodingTrials_pupilBlocks

    # pupil at peak accuracy
    pupilInd_at_peak_accuracy = np.argmax(peak_repAvg_accuracy_allSessions[count,:])
    pupilInd_at_worst_accuracy = np.argmin(peak_repAvg_accuracy_allSessions[count,:])

    pupil_at_peak_accuracy_allSessions[count] = avg_pupilSize_pupilBlocks[count,pupilInd_at_peak_accuracy].copy()
    pupil_at_worst_accuracy_allSessions[count] = avg_pupilSize_pupilBlocks[count,pupilInd_at_worst_accuracy].copy()
    
    # store accuracy in middle vs lowest pupil bin
    if (np.min(avg_pupilSize_pupilBlocks[count, :]) <= lowPupil_thresh):
        
        ind_midPupil = np.argmin(np.abs(avg_pupilSize_pupilBlocks[count, :] - 0.5))
        
        peak_repAvg_accuracy_mid_vs_low[0] = np.append(peak_repAvg_accuracy_mid_vs_low[0], peak_repAvg_accuracy_allSessions[count, ind_midPupil])
        peak_repAvg_accuracy_mid_vs_low[1] = np.append(peak_repAvg_accuracy_mid_vs_low[1], peak_repAvg_accuracy_allSessions[count, 0])

    if (np.max(avg_pupilSize_pupilBlocks[count, :]) >= highPupil_thresh):
        
        ind_midPupil = np.argmin(np.abs(avg_pupilSize_pupilBlocks[count, :] - 0.5))
        
        peak_repAvg_accuracy_mid_vs_high[0] = np.append(peak_repAvg_accuracy_mid_vs_high[0], peak_repAvg_accuracy_allSessions[count, ind_midPupil])
        peak_repAvg_accuracy_mid_vs_high[1] = np.append(peak_repAvg_accuracy_mid_vs_high[1], peak_repAvg_accuracy_allSessions[count, -1])


    
# SESSION AVERAGE VS PUPIL SIZE BINS
pupilSize_binCenters, avg_peakAccuracy_modulation_vs_pupilSize_bins, std_peakAccuracy_modulation_vs_pupilSize_bins, sem_peakAccuracy_modulation_vs_pupilSize_bins, \
    allSessions_peakAccuracy_modulation_pupilBins, pupilSize_data_in_pupilBins_allSessions = \
    fcn_sessionAvg_quantity_vs_pupilSize_bins_alt(avg_pupilSize_pupilBlocks, 0, 1, pupilSize_binStep, peak_repAvg_accuracy_modulation[:,:n_pupilBlocks], 1)



#%% STATISTICS

### compare accuray in middle vs low pupil bin
middlePupil_vs_lowPupil_accuracy_stats = fcn_Wilcoxon(peak_repAvg_accuracy_mid_vs_low[0], peak_repAvg_accuracy_mid_vs_low[1])

### compare accuracy in middle vs high pupil bin        
middlePupil_vs_highPupil_accuracy_stats = fcn_Wilcoxon(peak_repAvg_accuracy_mid_vs_high[0], peak_repAvg_accuracy_mid_vs_high[1])



#%% PRINT

print('*'*100)
print('*'*100)

print('middle vs low')
print(middlePupil_vs_lowPupil_accuracy_stats['pVal_2sided'])
print('n_sessions: %d' % len(np.nonzero(~np.isnan(peak_repAvg_accuracy_mid_vs_low[0]))[0]))

print('*'*100)
print('*'*100)

print('middle vs high')
print(middlePupil_vs_highPupil_accuracy_stats['pVal_2sided'])
print('n_sessions: %d' % len(np.nonzero(~np.isnan(peak_repAvg_accuracy_mid_vs_high[0]))[0]))

print('*'*100)
print('*'*100)


#%% EXAMPLE SESSION: PEAK ACCURACY VS PUPIL SIZE

if ( (cellSelection == '') and (rest_only == False) and (global_pupilNorm == False) ):

    # session index
    indSession = np.nonzero(sessions_to_run == example_session)[0][0]
    
    # plotting
    plt.rcParams.update({'font.size': 7})
    fig = plt.figure(figsize=(0.9, 1.))  
    ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 
        
    x = avg_pupilSize_pupilBlocks[indSession, :]*100
    y = peak_repAvg_accuracy_allSessions[indSession, :n_pupilBlocks]
    ax.plot(x, y, '-o', color='dimgray', markersize=2., linewidth=1)
            
    ax.set_xlim([-2, 102])
    ax.set_yticks([0.7, 1])
    ax.set_xlabel('pupil diameter\n[% max]')
    ax.set_ylabel('accuracy')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(('%s%s.pdf' % (outpath, fig1ID)), bbox_inches='tight', pad_inches=0, transparent=True)
    


#%% AVERAGE PEAK ACCURACY MODULATION BOX PLOTS

plt.rcParams.update({'font.size': 7})
if ( (rest_only == True) or (global_pupilNorm == True) ):
    fig = plt.figure(figsize=(1.8,1.8))  
    markerSize1 = 3.5
    markerSize2 = 4
    lineWidth = 1.5
elif ( (cellSelection != '') ):
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
y = allSessions_peakAccuracy_modulation_pupilBins.copy()
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
y = avg_peakAccuracy_modulation_vs_pupilSize_bins
ax.plot(x, y, '-o', markersize=markerSize2, linewidth=lineWidth, color='red', markerfacecolor='red', markeredgecolor='red', alpha=1)

plt.xticks(np.arange(0,120,20), labels=np.arange(0,120,20))
ax.set_xlim([-2,102])
if global_pupilNorm:
    ax.set_xlabel('pupil diameter\n[% max across all sessions]')
else:
    ax.set_xlabel('pupil diameter [% max]')
ax.set_ylabel('% change in accuracy\n(relative to max)', multialignment='center')
plt.savefig(('%s%s.pdf' % (outpath, fig2ID)), bbox_inches='tight', pad_inches=0, transparent=True)
    

#%% COMPARE ACCURACY IN MIDDLE PUPIL BIN TO LOWEST PUPIL BIN

plt.rcParams.update({'font.size': 7})

if ( (cellSelection != '') ):
    fig = plt.figure(figsize=(0.7,1.2))      
    markerSize = 2
    lineWidth = 0.75
else:
    fig = plt.figure(figsize=(0.9,1.6))  
    markerSize = 3
    lineWidth = 1

    
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8])

# loop over relevant sessions
for i in range(0, len(peak_repAvg_accuracy_mid_vs_low[0])):
    
     data_mid_low = np.append(peak_repAvg_accuracy_mid_vs_low[0][i], peak_repAvg_accuracy_mid_vs_low[1][i])
     ax.plot([0,1], data_mid_low, '-o', markersize=markerSize, linewidth=lineWidth, markerfacecolor='r', markeredgecolor='r', color='dimgrey', alpha=0.8)
     

ax.set_xticks([0, 1])
ax.set_xticklabels(['central\ndecile', 'first\ndecile'])
ax.set_ylabel('accuracy')
plt.savefig(('%s%s.pdf' % (outpath, fig3ID)), bbox_inches='tight', pad_inches=0, transparent=True) 


#%% COMPARE ACCURACY IN MIDDLE PUPIL BIN TO HIGHEST PUPIL BIN

plt.rcParams.update({'font.size': 7})

if ( (cellSelection != '') ):
    fig = plt.figure(figsize=(0.7,1.2))      
    markerSize = 2
    lineWidth = 0.75
else:
    fig = plt.figure(figsize=(0.9,1.6))  
    markerSize = 3
    lineWidth = 1

ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 

# loop over relevant sessions
for i in range(0, len(peak_repAvg_accuracy_mid_vs_high[0])):
        
     data_mid_high = np.append(peak_repAvg_accuracy_mid_vs_high[0][i], peak_repAvg_accuracy_mid_vs_high[1][i])
     ax.plot([0,1], data_mid_high, '-o', markersize=markerSize, linewidth=lineWidth, markerfacecolor='r', markeredgecolor='r', color='dimgrey',alpha=0.8)
     

ax.set_xticks([0, 1])
ax.set_xticklabels(['central\ndecile', 'last\ndecile'])
ax.set_ylabel('accuracy')
plt.savefig(('%s%s.pdf' % (outpath, fig4ID)), bbox_inches='tight', pad_inches=0, transparent=True) 

    
#%% DISTRIBUTION OF BEST AND WORST ACCURACY ACROSS SESSIONS


if ( (cellSelection == '') and (rest_only == False) and (global_pupilNorm == False) ):
    
    plt.rcParams.update({'font.size': 7})
    fig = plt.figure(figsize=(1.8,1.8))  
    ax = fig.add_axes([0.2, 0.2, 0.8, 0.8])
    
    ax.hist(pupil_at_peak_accuracy_allSessions*100, np.arange(0,110,10), color='teal', alpha=1, label='best')
    ax.hist(pupil_at_worst_accuracy_allSessions*100, np.arange(0,110,10), color='grey', alpha=1, label='worst')
    ax.set_xlim([-2, 102])
    ax.set_xlabel('pupil diameter [% max]')
    ax.set_ylabel('number of sessions')
    ax.legend()
    plt.savefig(('%s%s.pdf' % (outpath, fig5ID)), bbox_inches='tight', pad_inches=0, transparent=True)