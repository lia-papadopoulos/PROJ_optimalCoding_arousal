'''
example of how to load an .h5 file for one session in order to access data 
'''

#%% all user inputs in this block

import h5py
import sys

# get path to processed data
sys.path.append('../../')
import global_settings

# path to processed data
data_path = global_settings.path_to_processed_data

# session to run
session_to_run = 'LA3_session3'

#%% load and extract data
  
########## load data ############################################################ 

filename = (('%s%s_processed_data.h5') % (data_path, session_to_run))
data = h5py.File(filename, 'r')


########## behavioral data ############################################################ 

# time array; shape = (n_tPts,)
time_pts = data['behavioral_data']['time'][:]
# pupil trace array; shape = (n_tPts,)
pupil_trace = data['behavioral_data']['pupil_trace'][:]
# run trace array; shape = (n_tPts,)
run_trace = data['behavioral_data']['run_trace'][:]

# behavioral data units (strings)
pupil_trace_units = data['behavioral_data'].attrs['pupil_trace_units']
run_trace_units = data['behavioral_data'].attrs['run_trace_units']


########## cell spike times ############################################################ 

# object array of cell spike times; shape = (n_cells, )
# ith element of cell_spike_times contains a 1d array of spike times for ith cell
cell_spike_times = data['cell_spikeTimes'][:]


########## stimulus data ############################################################

# stimulus onset time array; shape = (n_stim, )
stim_onset = data['stim_data']['stim_on_time'][:]
# stimulus frequency array; shape = (n_stim, )
stim_Hz = data['stim_data']['stim_Hz'][:]
# array containing the start and end times of each spontaneous block in the session
# shape = (2, n_spontBlocks); row0 = start time of each block; row1 = end time of each block
spont_blocks = data['stim_data']['spont_blocks'][:]

# other stimulus info (strings)
inter_stim_interval = data['stim_data'].attrs['interStim_interval']
stim_amp = data['stim_data'].attrs['stim_amp']
stim_duration = data['stim_data'].attrs['stim_duration']


########## close file ############################################################ 

data.close()

