'''
function to convert the neural and behavioral data from a single session
(stored in a .nwb file) to a dictionary that contains all relevant
information for downstream analyses

inputs: 
    session_name:       string that specifies name of session to analyze
                        e.g.: 'LA9_session1'
    data_path:          string that specifies the path to the .nwb file containing
                        the session data
    fname_end:          only required if you aren't analyzing the default 
                        dataset for a given session; in that case, fname_end
                        is a string that specifies the name of the alternate 
                        version of the data that you wish to analyze
'''

# --- basic imports ---
import pynwb
import numpy as np

# --- function definition ---
def fcn_processedNWBdata_to_dict(session_name, data_path, fname_end = ''):

    # --- full path to data --- 
    full_path = (('%s%s%s_processed_data.nwb') % (data_path, session_name, fname_end))

    # --- data dictionary ---
    data_dict = {}

    # --- session name ---
    data_dict['session_name'] = session_name

    # --- open nwb file ---
    with pynwb.NWBHDF5IO(full_path, mode="r") as io2:

        # nwbfile
        nwbfile = io2.read()

        # get trial start times
        data_dict['stim_on_time'] = nwbfile.trials.start_time[:]
        # get trial frequency
        data_dict['stim_freq'] = nwbfile.trials.stim_frequency[:]
        # get stimulus duration
        stim_duration_seconds = nwbfile.trials.stim_duration[:][0]
        data_dict['stim_duration'] = ( ('%d ms') % (stim_duration_seconds*1000) )
        # get spontaneous blocks
        spont_block_start_time = nwbfile.get_time_intervals('spontaneous_blocks').start_time[:]
        spont_block_end_time = nwbfile.get_time_intervals('spontaneous_blocks').stop_time[:]
        data_dict['spont_blocks'] = np.vstack((spont_block_start_time, spont_block_end_time))
        # get behavioral data and timestamps
        data_dict['time_stamp'] = nwbfile.processing['behavior']['PupilTracking']['pupil_diameter'].timestamps[:]
        data_dict['norm_pupilTrace'] = nwbfile.processing['behavior']['PupilTracking']['pupil_diameter'].data[:]
        data_dict['walk_trace'] = nwbfile.processing['behavior']['running_speed'].data[:]
        # get cell spike times
        data_dict['cell_spk_times'] = nwbfile.units.to_dataframe().to_numpy(dtype=object)[:,0]

        # compute a couple of other quantities
        data_dict['nTrials'] = np.size(data_dict['stim_freq'])
        data_dict['nCells'] = np.size(data_dict['cell_spk_times'])

    # --- delete ---
    del nwbfile
    del io2

    # --- return data dict ---
    return data_dict