"""
This script converts .h5 files with preprocessed neural and behavioral data to NWB format.

Usage:
python convert_h5_to_nwb.py ./<nwb_metadata_file> /<output_path>/  # Convert all sessions
python convert_h5_to_nwb.py ./<nwb_metadata_file> /<output_path>/ LA3_session3  # Convert single session

where <nwb_metadata_file> is a file that defines the metadata for all recorded sessions,
/<output_path>/ is the path to the directory where the NWB files will be saved,
and LA3_session3 is the name of a single session to convert (optional). Note that the 
metadata file contains the path to the .h5 files that should be loaded in.
"""

import os
import sys
import pynwb
from datetime import datetime
import numpy as np
import h5py
import importlib

# -- Select what to process (useful for debugging) --
ADD_TRIAL_INFO = 1     # Add trial information to NWB file
ADD_SPONT_BLOCKS = 1   # Add start and stop times for spontaneous blocks
ADD_SPIKES = 1         # Add spike times to NWB file
ADD_BEHAVIOR = 1       # Add pupil, running, and whisking data to NWB file
SAVE_NWB = 1           # Save NWB file

# -- Parse command line arguments --
if len(sys.argv) > 3:
    sessions = [sys.argv[3]]
else:
    sessions = None    
if len(sys.argv) > 2:
    metadata_filename = sys.argv[1]
    output_path = sys.argv[2]
else:
    raise ValueError('Please provide a metadata file.')

# -- Read metadata file --
def read_metadata(metadata_filename):
    spec = importlib.util.spec_from_file_location('metadata_module', metadata_filename)
    metamodule = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metamodule)
    return metamodule

metamodule = read_metadata(metadata_filename)
metadata = metamodule.metadata
if sessions is None:
    sessions = metadata.keys()

# -- For each session, read data from HDF5 file and create an NWB file --
for session in sessions:
    print(f'Converting session {session}')

    h5file_fullpath = os.path.join(metadata[session]['path_to_data'],
                                   metadata[session]['filename'])
    
    # -- Add sesion information --
    start_time = datetime.fromisoformat(metadata[session]['session_start_time'])
    nwbfile = pynwb.NWBFile(
        experiment_description = metadata[session]['experiment_description'],
        keywords = metadata[session]['keywords'],
        session_description = metadata[session]['session_description'],
        identifier = metadata[session]['identifier'],
        session_start_time = start_time,
        experimenter = metadata[session]['experimenter'],
        lab = metadata[session]['lab'],
        institution = metadata[session]['institution'],
        session_id = metadata[session]['session_id'],
        related_publications = metadata[session]['related_publications']
    )

    # -- Add subject information --
    nwbfile.subject = pynwb.file.Subject(
        subject_id = metadata[session]['subject_id'],
        age = metadata[session]['subject_age'],
        description = metadata[session]['subject_description'],
        species = metadata[session]['subject_species'],
        sex = metadata[session]['subject_sex']
    )

    # -- Add trial information --
    if ADD_TRIAL_INFO:
        with h5py.File(h5file_fullpath, 'r') as datafile:
            print('Storing trial information...')
            trial_start_time = datafile['stim_data']['stim_on_time'][()]
            stim_frequency = datafile['stim_data']['stim_Hz'][()]
            stim_amplitude_str = datafile['stim_data'].attrs['stim_amp']
            if ' dB' in stim_amplitude_str:
                stim_amplitude = float(stim_amplitude_str.replace(' dB','')) # In seconds
            stim_duration_str = datafile['stim_data'].attrs['stim_duration']
            if ' ms' in stim_duration_str:
                stim_duration = 1e-3*float(stim_duration_str.replace(' ms','')) # In seconds
            elif ' s' in stim_duration_str:
                stim_duration = float(stim_duration_str.replace(' s','')) # In seconds
            else:
                raise ValueError('Units of stimulus duration are not valid.')
            # Append an estimated trial_stop_time for the last trial
            trial_duration_estimate = np.median(np.diff(trial_start_time))
            trial_stop_time = np.append(trial_start_time[1:], trial_start_time[-1]+trial_duration_estimate)
            
            nwbfile.add_trial_column(name='stim_frequency',
                                     description='Stimulus frequency (Hz)')
            nwbfile.add_trial_column(name='stim_duration',
                                     description='Stimulus duration (s)')
            nwbfile.add_trial_column(name='stim_amplitude',
                                     description='Stimulus amplitude (dB)')
            # NOTE: Looping through >5000 trials takes a long time,
            #       but pynwb doesn't seem to have a faster way.
            for indt, start_time in enumerate(trial_start_time):
                nwbfile.add_trial(start_time = start_time,
                                  stop_time = trial_stop_time[indt],
                                  stim_frequency = stim_frequency[indt],
                                  stim_duration = stim_duration,
                                  stim_amplitude = stim_amplitude)
                
                
    # -- Add information about spontaneous blocks --
    if ADD_SPONT_BLOCKS:
        with h5py.File(h5file_fullpath, 'r') as datafile:
            print('Storing spontaneous block information...')   
            # start and end of spontaneous blocks
            spontBlock_start_time = datafile['stim_data']['spont_blocks'][0,:] 
            spontBlock_end_time = datafile['stim_data']['spont_blocks'][1,:]
            n_spontBlocks = np.size(spontBlock_start_time)
            # set up custom TimeIntervals table
            spontaneous_blocks = pynwb.epoch.TimeIntervals(
            name="spontaneous_blocks",
            description="start and stop times of spontaneous blocks (s)",
            )
            # fill up table with spontaneous block start and stop times
            for ind_spontBlock in range(0, n_spontBlocks):
                spontaneous_blocks.add_row(start_time=spontBlock_start_time[ind_spontBlock], stop_time=spontBlock_end_time[ind_spontBlock])
            # add to nwbfile
            _ = nwbfile.add_time_intervals(spontaneous_blocks)


    # -- Add spike-times information --
    if ADD_SPIKES:
        with h5py.File(h5file_fullpath, 'r') as datafile:
            print('Storing spike times information...')
            spike_times_all = datafile['cell_spikeTimes'][()]
            for indu, spike_times in enumerate(spike_times_all):
                nwbfile.add_unit(spike_times=spike_times)


    # -- Add pupil and running data --
    if ADD_BEHAVIOR:
        print('Storing behavior (pupil/running) information...')
        behavior_module = nwbfile.create_processing_module(
            name = 'behavior', 
            description = 'Pre-processed behavioral data'
        )

        with h5py.File(h5file_fullpath, 'r') as datafile:
            timestamps = datafile['behavioral_data']['time'][()]
            pupil = datafile['behavioral_data']['pupil_trace'][()]
            running = datafile['behavioral_data']['run_trace'][()]
            pupil_units = datafile['behavioral_data'].attrs['pupil_trace_units']
            running_units = datafile['behavioral_data'].attrs['run_trace_units']

        # Based on https://pynwb.readthedocs.io/en/stable/tutorials/domain/plot_behavior.html
        #                #pupiltracking-storing-continuous-eye-tracking-data-of-pupil-size
        pupil_diameter = pynwb.base.TimeSeries(
            name = 'pupil_diameter',
            description = metamodule.pupil_description,
            timestamps = timestamps,
            data = pupil,
            unit = pupil_units
        )
        pupil_tracking = pynwb.behavior.PupilTracking(time_series=pupil_diameter, name='PupilTracking')
        behavior_module.add(pupil_tracking)
        
        running_speed = pynwb.base.TimeSeries(
            name = 'running_speed',
            description = metamodule.running_description,
            timestamps = timestamps,
            data = running,
            unit = running_units
        )
        behavior_module.add(running_speed)

        
    # -- Save NWB file --
    if SAVE_NWB:
        nwb_filename = metadata[session]['filename'].replace('.h5', '.nwb')
        output_fullpath = os.path.join(output_path, nwb_filename)                           
        print(f'Saving {output_fullpath}')
        with pynwb.NWBHDF5IO(output_fullpath, 'w') as io:
            io.write(nwbfile)
            print('Done.\n')

'''
# -- Check the contents of the NWB object --
nwbfile.trials.to_dataframe()
nwbfile.units.to_dataframe()
nwbfile.processing['behavior']['PupilTracking']['pupil_diameter']
nwbfile.processing['behavior']['running_speed']
'''

# -- Useful for debugging --            
# with pynwb.NWBHDF5IO('/tmp/test.nwb', 'w') as io: io.write(nwbfile)

