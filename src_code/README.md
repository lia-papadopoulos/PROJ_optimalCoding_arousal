# Overview

This code runs the data analysis, simulations, and plotting associated with the paper "Modulation of metastable ensemble dynamics explains the inverted-U relationship between tone discriminability and arousal in auditory cortex" by L. Papadopoulos, S. Jo, K. Zumwalt, M. Wehr, S. Jaramillo, D.A. McCormick, and L. Mazzucato. A preprint of the manuscript is available here: https://doi.org/10.1101/2024.04.04.588209.

## Software Versions

Python version 3.9.9 
NumPy version 1.26  
SciPy version 1.12  
pyNWB version 3.1.2
scikit-learn version 0.24  
nitime version 0.10.2

## Usage


### Accessing the neural and behavioral data

#### Downloading the data

The preprocessed neural and behavioral data analyzed in the main text of the manuscript has been deposited on DANDI and can be accessed at:

--------------------------------------------------------------------------------------------------------------------------------------------------------------------  
**FILL THIS IN WITH LINK TO FINAL DANDISET**
--------------------------------------------------------------------------------------------------------------------------------------------------------------------

To download the dataset in a format that is compatible with the provided data analysis code:

1. Follow the instructions at https://docs.dandiarchive.org/user-guide-using/accessing-data/downloading/ to install the DANDI Python Client on your computer.
2. Once you have installed the DANDI Python Client, navigate to the `Download` button on the right-hand side of the Dandiset webpage (see link above for the url).
3. Click `Download` and copy the DANDI CLI command.
4. Run the copied command in your terminal to download the dataset.
5. In `src_code/global_settings.py` (see below) set `path_to_processed_data` to the directory containing the downloaded dataset (note that this path should point to the directory that houses the subdirectories for each subject, not to a specific subject directory). 

#### Organization of the data

The dataset consists of 5 subjects (LA3, LA8, LA9, LA11, LA12). Within each of the subject directories (e.g. `sub-LA9/`), there is one .nwb file per session associated with that subject (e.g., `sub-LA9/sub-LA9_ses-3_behavior.nwb`). The .nwb files contain the preprocessed neural and behavioral data analyzed in the manuscript. In all data analysis scripts, the .nwb files are loaded and unpacked using the `pynwb` python package (see, e.g., `src_code/data_analysis/fcn_processedNWBdata_to_dict.py`).


### Setting up 

In the top-level `src_code/` directory, you will find a file named `global_settings.py`. This file specifies absolute paths to the `src_code/` directory, the directory containing the neural and behavioral data, and various output directories that will contain all files and figures generated from running simulation, analysis, and plotting scripts. `global_settings.py` is imported by nearly all simulation, analysis, and plotting scripts using relative paths from the working directory. In order for the code to run properly, the user must update the absolute paths in `global_settings.py` based on the directory structure of the computer they are working on. The user should also pre-generate the output directories in `global_settings.py`, as most of them will not be generated adaptively.

### Batch job submission

All code is setup to submit batch jobs using task-spooler.

### Code organization

The `src_code/` directory contains all code required to run the simulations, analysis, and plotting for the manuscript. It is organized in several subdirectories that are responsible for different aspects of the analysis, as explained briefly below.

#### functions

`src_code/functions/` contains various modules that are used to run model simulations, analyze spike trains, run decoding and clustering analyses, perform statistical tests, etc. These modules are imported by scripts in `src_code/data_analysis/`, `src_code/run_simulations/`, `src_code/simulations_analysis/`, and `src_code/manuscript_plotting_scripts/`.

#### MFT

`src_code/MFT/` contains modules that are used for the mean-field analyses of the clustered network model.

#### data_analysis

`src_code/data_analysis/` contains code to analyze the neural data.

#### run_simulations

`src_code/run_simulations/` contains code to run model simulations.

#### simulations analysis

`src_code/simulations analysis/` contains code to analyze model simulations.

#### manuscript_plotting_scripts

`src_code/manuscript_plotting_scripts/` contains code to plot figures in the manuscript.