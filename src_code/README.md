# Overview

This code runs the data analysis, simulations, and plotting associated with the paper "Modulation of metastable ensemble dynamics explains the inverted-U relationship between tone discriminability and arousal in auditory cortex" by L. Papadopoulos, S. Jo, K. Zumwalt, M. Wehr, S. Jaramillo, D.A. McCormick, and L. Mazzucato.

## Software Versions

Python version 3.9
NumPy version 1.26
SciPy version 1.12
scikit-learn version 0.24

## Usage

### Downloading the neural data

The neural data can be downloaded from...

**FILL IN***

### Setting up 

In the top-level `src_code/` directory, you will find a file named `global_settings.py`. This file specifies absolute paths to the `src_code/` directory, the directory containing the neural data, and various output directories that will contain all files and figures generated from running simulation, analysis, and plotting scripts. `global_settings.py` is imported by nearly all simulation, analysis, and plotting scripts using relative paths from the working directory. In order for the code to run properly, the user must update the absolute paths in `global_settings.py` based on the directory structure of the computer they are working on.

### Code organization

The `src_code/` directory contains all code required to run the simulations, analysis, and plotting for the manuscript. It is organized in several subdirectories that are responsible for different aspects of the analysis, as explained briefly below.

#### functions

#### data_analysis

#### run_simulations

