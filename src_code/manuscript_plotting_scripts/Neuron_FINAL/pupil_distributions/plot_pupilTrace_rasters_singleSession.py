
'''
This script generates
    Fig1A colorbar
    Fig1C
    Fig1D
    Fig1E
'''


#%% basic imports
import sys        
import numpy as np
import numpy.matlib
from scipy.io import savemat
import os

#%% global settings
sys.path.append('../../../')
import global_settings

#%% plotting
import matplotlib
matplotlib.use('agg')
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
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
funcpath = global_settings.path_to_src_code + 'data_analysis/'
sys.path.append(funcpath)
from fcn_processedh5data_to_dict import fcn_processedh5data_to_dict

#%% settings

# paths
data_path = global_settings.path_to_processed_data
outpath = global_settings.path_to_manuscript_figs_final + 'pupil_distributions/'

# session name
session_name = 'LA12_session1'

# smoothing window
smoothingWindow = 50e-3

# figure IDs
fig1ID = 'Fig1A_colorbar'
fig2ID = 'Fig1C_lowQual'
fig3ID = 'Fig1C_smooth'
fig4ID = 'Fig1D'
fig5ID = 'Fig1E'

#%% make output directory
if os.path.isdir(outpath) == False:
    os.makedirs(outpath)

#%% LOAD IN SESSION INFO
session_info = fcn_processedh5data_to_dict(session_name, data_path)

#%% UNPACK SESSION INFO
tPts = session_info['time_stamp']
runSpeed = session_info['walk_trace']
pupil = session_info['norm_pupilTrace']*100
cell_spk_times = session_info['cell_spk_times'].copy()
spont_blocks = session_info['spont_blocks']
stim_freq = session_info['stim_freq']
stim_on_time = session_info['stim_on_time']

#%% SPONTANEOUS BLOCKS
spontBlock_start = spont_blocks[0,:].copy()
spontBlock_end = spont_blocks[1,:].copy()
n_spontBlocks = np.size(spontBlock_start)

#%% NUMBER OF CELLS
nCells = np.size(cell_spk_times)

#%% NUMBER OF TRIALS
nTrials = np.size(stim_freq)

#%% UNIQUE FREQUENCIES
uniqueFreq = np.unique(stim_freq)

#%% ABSOLUTE VALUE OF RUNNING SPEED
runSpeed = np.abs(runSpeed)

#%% SMOOTH AND DOWNSAMPLE TRACES FOR PLOTTING

bins = np.arange(tPts[0], tPts[-1], smoothingWindow)
bin_centers = (bins[:-1]+bins[1:])/2
binIDs = np.digitize(tPts, bins)
pupil_smooth = np.zeros(len(bins)-1)
run_smooth = np.zeros(len(bins)-1)

for indBin in range(0, len(bins)-1):
    
    pupil_smooth[indBin] = np.mean(pupil[np.nonzero(binIDs == indBin+1)[0]])
    run_smooth[indBin] = np.mean(runSpeed[np.nonzero(binIDs == indBin+1)[0]])


#%% SAVE PARAMETERS

params_dict = {}

params_dict['session_name'] = session_name
params_dict['data_path'] = data_path
params_dict['smoothingWindow'] = smoothingWindow

savemat(('%spupilTrace_rasters_params.mat' % (outpath)), params_dict)

print('done')




#%% PLOT TONE FREQUENCIES COLORBAR2

cmap = cm.get_cmap('plasma', 5)
cmap = cmap(range(5))
cmap_im = ListedColormap(cmap)
fig, ax = plt.subplots(figsize=(3.75,2.5))
freq_array = [np.arange(0,5)]
ax.imshow(freq_array, cmap=cmap_im)
ax.set_xticks([])
ax.set_yticks([])
plt.savefig(outpath + fig1ID + '.pdf', transparent=True)


#%% PLOT PUPIL AND RUNNING TRACE


# start and end times 
t0 = tPts[0]
tf = tPts[-1]

plt.rcParams.update({'font.size': 8})
fig = plt.figure(figsize=(5.1, 2.1))  

ax2 = fig.add_axes([0.2, 0.61, 0.79, 0.35]) 
ax1 = fig.add_axes([0.2, 0.19, 0.79, 0.35]) 


ax1.plot(tPts, pupil, color = [0.2,0.2,0.2], linewidth = 0.25)
for ind_spontBlock in range(0, n_spontBlocks):
    ax1.fill_betweenx([-2,102], spontBlock_start[ind_spontBlock], spontBlock_end[ind_spontBlock], color='lightseagreen', alpha=0.2)
ax1.set_ylabel('pupil diameter \n[% max]', multialignment='center')
ax1.set_xlim([t0, tf])
ax1.set_ylim([-2,102])
ax1.set_xlabel('time [seconds]')
ax1.set_xlim([t0, tf])


ax2.plot(tPts, runSpeed, color = [0.7,0.7,0.7], linewidth=0.25)
for ind_spontBlock in range(0, n_spontBlocks):
    ax2.fill_betweenx([0,16], spontBlock_start[ind_spontBlock], spontBlock_end[ind_spontBlock], color='lightseagreen', alpha=0.2)
ax2.set_ylabel('run speed\n[cm/s]', multialignment='center')
ax2.set_yticks([0, 8, 16])
ax2.set_ylim([0, 16])
ax2.set_xlim([t0, tf])
ax2.set_xticklabels([])

plt.savefig(outpath + '%s.png'  % fig2ID, bbox_inches='tight', pad_inches=0.01, transparent=False)



#%% PLOT PUPIL AND RUNNING TRACE SMOOTHED

    
# start and end times 
t0 = tPts[0]
tf = tPts[-1]

plt.rcParams.update({'font.size': 8})
fig = plt.figure(figsize=(5.8, 2.1))  

ax2 = fig.add_axes([0.2, 0.61, 0.79, 0.35]) 
ax1 = fig.add_axes([0.2, 0.19, 0.79, 0.35]) 


ax1.plot(bin_centers, pupil_smooth, color = [0.15,0.15,0.15], linewidth = 0.5)
for ind_spontBlock in range(0, n_spontBlocks):
    ax1.fill_betweenx([-2,102], spontBlock_start[ind_spontBlock], spontBlock_end[ind_spontBlock], color='lightseagreen', alpha=0.2)
ax1.set_ylabel('pupil diameter \n[% max]', multialignment='center')
ax1.set_xlim([t0, tf])
ax1.set_ylim([-2,102])
ax1.set_xlabel('time [seconds]')
ax1.set_xlim([t0, tf])


ax2.plot(bin_centers, run_smooth, color = [0.6,0.6,0.6], linewidth=0.5)
for ind_spontBlock in range(0, n_spontBlocks):
    ax2.fill_betweenx([0,16], spontBlock_start[ind_spontBlock], spontBlock_end[ind_spontBlock], color='lightseagreen', alpha=0.2)
ax2.set_ylabel('run speed\n[cm/s]', multialignment='center')
ax2.set_yticks([0, 8, 16])
ax2.set_ylim([0, 16])
ax2.set_xlim([t0, tf])
ax2.set_xticklabels([])

plt.savefig(outpath + '%s.pdf'  % fig3ID, bbox_inches='tight', pad_inches=0.01, transparent=False)


#%% PLOT EVOKED RASTER

# time that you want to start & end plotting
t0_evoked = 1625
tf_evoked = t0_evoked + 5


# figure
plt.rcParams.update({'font.size': 8})
fig = plt.figure(figsize=(2.5, 2.3))  
ax1 = fig.add_axes([0.25, 0.7, 0.7, 0.2]) 
ax2 = fig.add_axes([0.25, 0.1, 0.7, 0.55]) 


ind_t0 = np.argmin(np.abs(bins - t0_evoked))
ind_tf = np.argmin(np.abs(bins - tf_evoked))


# plot pupil trace
x = bins[ind_t0:ind_tf+1]
y = pupil_smooth[ind_t0:ind_tf+1]
ax1.plot(x, y, color = [0.15,0.15,0.15], linewidth = 1)
ax1.set_yticks([0,100])
ax1.set_ylim([0, 100])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_ylabel('pupil diam.\n[% max]', multialignment='center')
ax1.set_xticks([])
ax1.set_xticklabels([])
ax1.set_xlim([t0_evoked, tf_evoked])


# plot rasters
for ind, cell in enumerate(np.arange(0, nCells)):
    
    x = cell_spk_times[cell]
    x = x[ (x >= t0_evoked) & (x <= tf_evoked) ]
    y = np.ones(len(x))*ind
    ax2.plot(x, y, '.', markersize=0.6, color='k')

for indStim in np.arange(0, nTrials):

    x = stim_on_time[indStim]
    y = nCells
    
    if ( (x >= t0_evoked) and (x <= tf_evoked) ):
    
        indFreq = np.nonzero(uniqueFreq == stim_freq[indStim])[0][0]
        ax2.plot([x,x], [nCells+1,nCells+5], color=cmap[indFreq,:], linewidth=2)
    
ax2.set_ylabel('single units')
ax2.set_xlabel('time [seconds]')
ax2.set_xlim([t0_evoked, tf_evoked])
ax2.set_ylim([-2,nCells+7])
ax2.set_xticks([t0_evoked, tf_evoked])
ax2.set_xticklabels([t0_evoked, tf_evoked])
ax2.set_yticks([0, nCells-1])
ax2.set_yticklabels(['1', ('%d' % nCells)])
plt.savefig(outpath + '%s.pdf' % fig4ID, bbox_inches='tight', pad_inches=0.01, transparent=True)


#%% PLOT SPONTANEOUS RASTER

# which spontaneous block to plot
blockPlot = 3

t0_spont = np.round(spontBlock_start[blockPlot]) + 100
tf_spont = t0_spont + 5

if tf_spont >= spontBlock_end[blockPlot]:
    sys.exit('plotting outside of spontaneous block')


# figure
plt.rcParams.update({'font.size': 8})
fig = plt.figure(figsize=(2.5, 2.3))  
ax1 = fig.add_axes([0.25, 0.7, 0.7, 0.2]) 
ax2 = fig.add_axes([0.25, 0.1, 0.7, 0.55]) 


ind_t0 = np.argmin(np.abs(bins - t0_spont))
ind_tf = np.argmin(np.abs(bins - tf_spont))

# plot pupil trace
x = bins[ind_t0:ind_tf+1]
y = pupil_smooth[ind_t0:ind_tf+1]
ax1.plot(x, y, color = [0.15,0.15,0.15], linewidth = 1)
ax1.set_yticks([0,100])
ax1.set_ylim([0, 100])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_ylabel('pupil diam.\n[% max]', multialignment='center')
ax1.set_xticks([])
ax1.set_xticklabels([])
ax1.set_xlim([t0_spont, tf_spont])


# plot raster
for ind, cell in enumerate(np.arange(0, nCells)):
    
    x = cell_spk_times[cell]
    x = x[ (x >= t0_spont) & (x <= tf_spont) ]
    y = np.ones(len(x))*ind
    ax2.plot(x, y, '.', markersize=0.6, color='k')

ax2.set_xlabel('time [seconds]')
ax2.set_ylabel('single units')
ax2.set_xlim([t0_spont, tf_spont])
ax2.set_xticks([t0_spont, tf_spont])
ax2.set_xticklabels([ ('%d' % t0_spont), ('%d' % tf_spont)])
ax2.set_yticks([0, nCells-1])
ax2.set_yticklabels(['1', ('%d' % nCells)])
ax2.set_ylim([-2,nCells+7])
plt.savefig(outpath + '%s.pdf'  % fig5ID, bbox_inches='tight', pad_inches=0.01, transparent=True)


