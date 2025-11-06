"""
Example showing how to read an NWB file created by 'convert_h5_to_nwb.py'

Usage:
python example_read_nwb.py ./<path_to_NWB_file.nwb>
"""
#%%
import pynwb
import sys
#%%
if len(sys.argv) > 1:
    nwb_path = sys.argv[1]
else:
    raise ValueError('Please provide an NWB file to load.')

#%%
ioObj = pynwb.NWBHDF5IO(nwb_path, 'r', load_namespaces=True)
nwbfile = ioObj.read()

input('Press ENTER to show stimulus data...')
print('--- Stimulus data (as a dataframe) ---')
print(nwbfile.trials.to_dataframe())

input('Press ENTER to show spontaneous blocks...')
print('--- Stimulus data (as a dataframe) ---')
print(nwbfile.get_time_intervals('spontaneous_blocks').to_dataframe())

input('Press ENTER to show spike times...')
print('--- Spike times (as a dataframe) ---')
print(nwbfile.units.to_dataframe())

input('Press ENTER to show pupil data...')
print('--- Pupil diameter (data and timestamps)---')
print(nwbfile.processing['behavior']['PupilTracking']['pupil_diameter'].data[()])
print(nwbfile.processing['behavior']['PupilTracking']['pupil_diameter'].timestamps[()])

input('Press ENTER to show running data...')
print('--- Running speed (data and timestamps)---')
print(nwbfile.processing['behavior']['running_speed'].data[()])
print(nwbfile.processing['behavior']['running_speed'].timestamps[()])

