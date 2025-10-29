

"""
This script generates
    Fig2A
"""

#%%
# basic imports
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
from fcn_SuData import fcn_makeTrials
from fcn_SuData import fcn_compute_pupilMeasure_eachTrial
from fcn_SuData import fcn_pupilPercentile_to_pupilSize
from fcn_processedh5data_to_dict import fcn_processedh5data_to_dict
from fcn_processedNWBdata_to_dict import fcn_processedNWBdata_to_dict

#%% settings

# paths
data_path = global_settings.path_to_processed_data
outpath = global_settings.path_to_manuscript_figs_final + 'pupil_distributions/'

# window for trials
trial_window = [-100e-3, 600e-3]

# pupil blocks
pupilBlock_size = 0.1
pupilBlock_step = 0.1

# bins
bins = np.arange(0, 102, 2)

# pupil size method
pupilSize_method = 'avgSize_beforeStim'

# session to run
session_name = 'LA12_session1'

# data filetype
data_filetype = 'nwb'

# figure id
figID = 'Fig2A'


#%% make output directory
if os.path.isdir(outpath) == False:
    os.makedirs(outpath)

#%% LOAD IN SESSION INFO

# make session dictionary
if data_filetype == 'h5':
    session_info = fcn_processedh5data_to_dict(session_name, data_path)
elif data_filetype == 'nwb':
    session_info = fcn_processedNWBdata_to_dict(session_name, data_path)
else:
    sys.exit('unknown data_filetype')

#%% UPDATE SESSION INFO

session_info['pupilSize_method'] = pupilSize_method
session_info['trial_window'] = trial_window
session_info['pupilBlock_size'] = pupilBlock_size
session_info['pupilBlock_step'] = pupilBlock_step

print('session updated')


#%% make trials
session_info = fcn_makeTrials(session_info)
trial_start = session_info['trial_start']
trial_end = session_info['trial_end']


#%% compute average pupil size in trials
session = fcn_compute_pupilMeasure_eachTrial(session_info)
avg_pupilSize = session['trial_pupilMeasure']

#%% percentile bins

# number of blocks
nBlocks = int((1 - pupilBlock_size)/(pupilBlock_step) + 1)

# initialize
pupilSize_percentileBlocks = fcn_pupilPercentile_to_pupilSize(session)

    
#%% plotting



#%% plot pupil size distribution

plt.rcParams.update({'font.size': 7})
fig = plt.figure(figsize=(1.,1.))  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 

counts, _, _ = ax.hist(avg_pupilSize*100, bins, color='gray')
for blockInd in range(0,nBlocks):
    x = pupilSize_percentileBlocks[1, blockInd]*100
    y = np.max(counts)
    ax.plot([x,x], [0,y], color='black', linewidth=0.5)
    
ax.set_xlim([-2,102])
ax.set_ylabel('trial count')
ax.set_xlabel('pupil diameter\n[% max]')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig(((outpath + '%s.pdf') % figID), bbox_inches='tight', pad_inches=0, transparent=True)


#%% save parameters

params_dict = {}

params_dict['session_name'] = session_name
params_dict['trial_window'] = trial_window
params_dict['pupilBlock_size'] = pupilBlock_size
params_dict['pupilBlock_step'] = pupilBlock_step
params_dict['pupilSize_method'] = pupilSize_method
params_dict['data_path'] = data_path

savemat(('%spupilDiameter_in_trials_params.mat' % (outpath)), params_dict)

print('done')

