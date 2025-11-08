# Overview

This directory contains the code used to process the electrophysiology and behavioral data from Su.

# Creating preprocessed version of Su's data

We created preprocessed versions of the neuropixels and behavioral data from Su. This allowed for streamlining all analyses and simplifed the public distribution of the data. Below, we document the steps followed to create preprocessed data for each session given the original data that Su shared with us.

## Original data

The original data that we received from Su is located in:  

`\ion-nas2\home\Brain_Initiative\Neuropixels\Su_NP\ToLiaLuca\`

The usable sessions (i.e., the ones that we analyze) are:  

- LA3_session3
- LA8_session1
- LA8_session2
- LA9_session1
- LA9_session3
- LA9_session4
- LA9_session5
- LA11_session1
- LA11_session2
- LA11_session3
- LA11_session4
- LA12_session1
- LA12_session2
- LA12_session3
- LA12_session4

## Preprocessing

The goal of the preprocessing is to clean up the original data and package it into a format that is easy to use for subsequent analyses.

The directory containing all scripts for the preprocessing is:
`/src_code/data_analysis/create_processed_dataset/`

We explored different types of preprocessing that differ in their implementation of the cell selection criteria. However, all versions of the preprocessing follow the same general schema:

- Load in single session data from Su
- Run cell selection procedure and output cell spike times of all good units 
- Run processing of behavioral data
- Get stimulus information
- Save the processed data as an .h5 file


### Main scripts

Each version of the preprocessing is associated with a different main script that runs the preprocessing pipeline. The main difference between the different versions of the preprocessing is the cell selection criteria. 

- `run_data_processing.py`
    - This script runs the default preprocessing. It implements cell selection based on an analysis of the spike template amplitudes for each unit marked as            "good" by Su. In the script, the user can also specify if they want to use global pupil normalization or not. The default is non-global-normalization, wherein the pupil diameter of each session is normalized by its own maximum value (rather than by the maximum value across all sessions). This is the preprocessing version that was used to generate the main text figures in the manuscript.

- `run_data_processing_spkTemplates_soundResp_cellSelection.py`
    - This script implements cell selection based on analysis of the spike template amplitudes AND based on tone-responsiveness of each unit marked as "good" by Su. This is the preprocessing version that was used to check the robustness of the main results to cell-selection criteria (Fig. S8) in the manuscript.

- `run_data_processing_soundResp_cellSelection.py`
    - This script impelements cell selection based only the tone-responsiveness of each unit marked as "good" by Su.
        
- `run_data_processing_no_cellSelection.py`
    - This script does not implement any additional cell selection beyond Su's manual curation.
        
- `run_data_processing_rateDrift_cellSelection.py`
    - This script implements cell selection criteria based on an analysis of firing rate drift over time for the units marked as "good" by Su.
    
- `fcn_rateDrift_soundResp_cellSelection.py`
    - This script implements cell selection criteria based on an analysis of firing rate drift over time AND based on tone-responsiveness of each unit marked as "good" by Su.

All of the preprocessing scripts generate an hdf5 file for each session that contains the spike times of each unit (after implementing cell selection criteria), preprocessed behavioral data (i.e., cleaned and smoothed pupil, run, and whisk traces), and all stimulus information needed for subsequent analyses.


### Parameter files

- `behavioral_preprocessing_info.py`: stores parameters required for preprocessing the behavioral data (pupil, run, whisk traces).

- `cell_selection_info.py`: cell selection parameters for `run_data_processing.py`
- `params_spkTemplate_soundResp.py`: cell selection parameters for `run_data_processing_spkTemplates_soundResp_cellSelection.py`
- `soundResp_params.py`: cell selection parameters for `run_data_processing_soundResp_cellSelection.py`
- `rateDrift_params.py`: cell selection parameters for `run_data_processing_rateDrift_cellSelection.py`
- `rateDrift_soundResp_params.py`: cell selection parameters for `rateDrift_soundResp_params.py`


### Helper functions

- `fcn_behavioral_preprocessing.py`: functions for running preprocessing of behavioral data; use parameters in `behavioral_preprocessing_info.py`
- `fcn_get_stimulus_data.py`: function used to extract stimulus information from the data Su shared with us
- `fcn_save_processed_data.py`: function that saves preprocessed data as hd5f file

- `fcn_spkTemplate_cellSelection.py`: function that implements cell selection for `run_data_processing.py`
- `fcn_spkTemplate_soundResp_cellSelection.py`: function that implements cell selection for `run_data_processing_spkTemplates_soundResp_cellSelection.py`
- `fcn_soundResp_cellSelection.py`: function that implements cell selection for `run_data_processing_soundResp_cellSelection.py`
- `fcn_no_cellSelection.py`: function that implements cell selection for `run_data_processing_no_cellSelection.py` (just extracts "good" units from Su's data and stores spike times)
- `fcn_rateDrift_cellSelection.py`: function that implements cell selection for `run_data_processing_rateDrift_cellSelection.py`
- `fcn_rateDrift_soundResp_cellSelection.py`: function that implements cell selection for `fcn_rateDrift_soundResp_cellSelection.py`


### Other scripts

- `load_h5files_example.py`: example showing how to load an .h5 file for one session in order to access data neural and behavioral data
- `find_max_smoothedPupil.py`: script that computes the maximum of the smoothed pupil trace for each session; used to determine global pupil normalization factor.

