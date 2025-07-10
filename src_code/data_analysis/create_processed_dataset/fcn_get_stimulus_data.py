#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract relevant data from preprocessed dataset from Su
"""

import h5py

def fcn_get_stimulus_data(session_name, data_path):

    # load the data
    f = h5py.File(data_path + session_name + '.mat','r')

    # get relevant data
    stim_on_time = f['stim_on_time'][0,:]
    stim_Hz = f['stimHz'][0,:]
    spont_blocks = f['spn_time'][:,:]
    
    stim_duration = ''.join(chr(i[0]) for i in f[f['stimInfo'][1,0]][:])
    interStim_interval = ''.join(chr(i[0]) for i in f[f['stimInfo'][1,1]][:])
    stim_amp = ''.join(chr(i[0]) for i in f[f['stimInfo'][1,2]][:])
    
    # close file
    f.close()
    
    # results
    results = {}
    results['stim_on_time'] = stim_on_time
    results['stim_Hz'] = stim_Hz
    results['stim_duration'] = stim_duration
    results['interStim_interval'] = interStim_interval
    results['stim_amp'] = stim_amp
    results['spont_blocks'] = spont_blocks
    
    return results
    