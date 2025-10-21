# Overview

`src_code/MFT/` contains various modules for running mean-field analyses of LIF network models. These modules are imported by scripts in `src_code/simulations_analysis/run_MFT/`

## Subdirectories

The mean-field modules are organized into two subdirectories: `basicEI_networks/` and `clusteredNets_fullTheory/`.

### basicEI_networks/

This directory contains modules to run mean-field analyses of 2-cluster LIF networks.

1. `fcn_MFT_fixedInDeg_clusNet.py`: Set of functions used to find the mean-field solutions of a 2-cluster network with specified model parameters.
2. `effectiveMFT/fcn_effectiveMFT_fixedInDeg_clusNet.py`: Set of functions for running the effective mean-field theory of a 2-cluster network with specified model parameters.

### clusteredNets_fullTheory/

This directory contains modules to run mean-field analyses on more general clustered networks (i.e., networks with more than 2 clusters).

1. `fcn_MFT_general_tools.py`: Set of functions that are used in MFT calculations of LIF networks.
2. `fcn_MFT_clusteredEINetworks_tools.py`: Set of functions used to setup various quantities for MFT calculations of clustered LIF networks; also uses `fcn_MFT_general_tools.py`. 
3. `fcn_MFT_fixedInDeg_generalNet.py`: Set of functions used to find the mean-field solution of an LIF network; assumes that there is no quenched variability and that population-level weight and degree matrices have already been computed; also uses `fcn_MFT_general_tools.py`.
4. `master_MFT_fixedInDeg_EIclusters.py`: Set of functions to run mean-field analysis of an LIF network (potentially clustered) with specified model parameters; calls functions in `fcn_MFT_clusteredEINetworks_tools.py` and `fcn_MFT_fixedInDeg_generalNet.py`.
5. `settings.py`: Sets path to the function that generates network architecture given connectivity parameters.