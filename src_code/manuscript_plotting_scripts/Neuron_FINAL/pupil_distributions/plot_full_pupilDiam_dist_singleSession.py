
'''
This script generates
    Fig1B
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

# session
session_name = 'LA12_session1'

# running threshold
runThresh = 2.0

# figure id
figID = 'Fig1B'

#%% make output directory
if os.path.isdir(outpath) == False:
    os.makedirs(outpath)

#%% load session
session_info = fcn_processedh5data_to_dict(session_name, data_path)

#%% unpack session
tPts = session_info['time_stamp']
runSpeed = session_info['walk_trace']
pupil = session_info['norm_pupilTrace']*100

#%% run and rest data
run_data = np.nonzero(np.abs(runSpeed) >= runThresh)[0]
rest_data = np.nonzero(np.abs(runSpeed) < runThresh)[0]

#%% plot pupil size distribution

plt.rcParams.update({'font.size': 8})
fig = plt.figure(figsize=(2.35,1.7))  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 

ax.hist(pupil[rest_data], bins=np.arange(0,1,0.025)*100, color=[0.1,0.1,0.1], label='rest')
ax.hist(pupil[run_data], bins=np.arange(0,1,0.025)*100, color=[0.7,0.7,0.7], label='run')
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.xlabel('pupil diameter [% max]')
plt.ylabel('count')
plt.legend(loc = 'upper right', frameon=False)
plt.xlim([-2, 102])
plt.savefig(outpath + '%s.pdf' % figID, bbox_inches='tight', pad_inches=0, transparent=True)

#%% save parameters

params_dict = {}
params_dict['data_path'] = data_path
params_dict['session_name'] = session_name
params_dict['runThresh'] = runThresh
savemat(('%spupilDiameter_hist_params.mat' % (outpath)), params_dict)

print('done')



