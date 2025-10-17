
'''
This script generates
    FigS1E
''' 

#%% basic imports
import numpy as np
from scipy.io import loadmat
import sys
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

#%% settings


# sessions to plot
sessions_to_run = np.array([\
                   'LA3_session3', \
                   'LA8_session1', \
                   'LA8_session2', \
                   'LA9_session1', \
                   'LA9_session3', 'LA9_session4', 'LA9_session5', \
                   'LA11_session1', 
                   'LA11_session2', \
                   'LA11_session3', \
                   'LA11_session4', \
                   'LA12_session1', 'LA12_session2', 'LA12_session3', 'LA12_session4'
                    ])
                   
    

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

# set paths
decoding_path = global_settings.path_to_data_analysis_output + 'decoding_pupil/'
outpath = global_settings.path_to_manuscript_figs_final + 'decoding_data/original_cellSelection/'

# figure id
figID = 'FigS1E'


#%% make output directory
if os.path.isdir(outpath) == False:
    os.makedirs(outpath)


#%% set up based on analysis parameters

# filenames          
fname_end_rest = (('_sweep_pupilSize_pupilSplit%s_pupilBlock%0.3f_pupilStep%0.3f_windSize%0.3fs_windStep%0.3fs_decoder%s_crossVal%s_nFreqs%d_restOnly') % \
             (pupilSplit_method, pupilBlock_size, pupilBlock_step, windSize, windStep, decoderType, crossvalType, nFreqs))          

fname_end_all = (('_sweep_pupilSize_pupilSplit%s_pupilBlock%0.3f_pupilStep%0.3f_windSize%0.3fs_windStep%0.3fs_decoder%s_crossVal%s_nFreqs%d') % \
             (pupilSplit_method, pupilBlock_size, pupilBlock_step, windSize, windStep, decoderType, crossvalType, nFreqs))         
        
# figure name 
fig_name = (('sweep_pupilSize_decoder%s_crossVal%s_nFreqs%d') % (decoderType, crossvalType, nFreqs))


#%% initialize data
    
nSessions = len(sessions_to_run)
nBlocks = int((1 - pupilBlock_size)/pupilBlock_step + 1)

avg_pupilDiameter_in_decileBins_rest = np.zeros((nSessions, nBlocks, n_decodeReps)) 
avg_pupilDiameter_in_decileBins_all = np.zeros((nSessions, nBlocks, n_decodeReps)) 


#%% loop over sessions

for count, session_name in enumerate(sessions_to_run):
        
    for repInd in range(0, n_decodeReps):
    
        decoding_filename_rest = ((decoding_path + 'decode_toneFreq_session%s' + fname_end_rest + '_rep%d.mat') % (session_name, repInd)) 
        decoding_data_rest = loadmat(decoding_filename_rest, simplify_cells=True)
    
        decoding_filename_all = ((decoding_path + 'decode_toneFreq_session%s' + fname_end_all + '_rep%d.mat') % (session_name, repInd)) 
        decoding_data_all = loadmat(decoding_filename_all, simplify_cells=True)
    
        avg_pupilDiameter_in_decileBins_rest[count, :, repInd] = decoding_data_rest['avg_pupilSize_decodingTrials_pupilBlocks'][:, repInd].copy()
        avg_pupilDiameter_in_decileBins_all[count, :, repInd] = decoding_data_all['avg_pupilSize_decodingTrials_pupilBlocks'][:, repInd].copy()

#%% average across repetitions

repAvg_avg_pupilDiameter_in_decileBins_rest = np.mean(avg_pupilDiameter_in_decileBins_rest, 2)
repAvg_avg_pupilDiameter_in_decileBins_all = np.mean(avg_pupilDiameter_in_decileBins_all, 2)


#%% plotting


#%% average pupil size in last decile bin: all data and without locomotion trials

plt.rcParams.update({'font.size': 7})
fig = plt.figure(figsize=(1.8,1.8))  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 

min_data = np.min(np.append(repAvg_avg_pupilDiameter_in_decileBins_all[:,-1], repAvg_avg_pupilDiameter_in_decileBins_rest[:,-1]))
max_data = np.max(np.append(repAvg_avg_pupilDiameter_in_decileBins_all[:,-1], repAvg_avg_pupilDiameter_in_decileBins_rest[:,-1]))

ax.plot(repAvg_avg_pupilDiameter_in_decileBins_all[:,-1], repAvg_avg_pupilDiameter_in_decileBins_rest[:,-1], 'o', color='black', markersize=4)
ax.plot([min_data, max_data], [min_data, max_data], color='r')
ax.set_xlabel('all data')
ax.set_ylabel('w/o locomotion trials')
plt.savefig(('%s%s.pdf' % (outpath, figID)), bbox_inches='tight', pad_inches=0, transparent=True)
