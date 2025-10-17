# Overview

`src_code/functions/` contains various modules that are used to run model simulations, analyze spike trains, run decoding and clustering analyses, perform statistical tests, etc. These modules are imported by scripts in `src_code/data_analysis/`, `src_code/run_simulations/`, `src_code/simulations_analysis/`, and `src_code/manuscript_plotting_scripts/`.


## modules

`fcn_analyze_corr.py`: set of functions to analyze results of clustering correlation matrices  
`fcn_compute_firing_stats.py`: set of functions to analyze spike trains from model simulations and compute various measures of cluster dynamics  
`fcn_decoding.py`: set of functions used to run population decoding analyses  
`fcn_hierarchical_clustering.py`: set of functions used to run hierarchical clustering analyses on spike-count correlation matrices  
`fcn_make_network_2cluster.py`: set of functions to make 2-cluster networks  
`fcn_make_network_cluster.py`: set of functions to make full clustered networks  
`fcn_simulation_EIextInput.py`: main function for running model simulations  
`fcn_simulation_loading.py`: function that helps load model simulations
`fcn_statistics.py`: functions to perform statistical analyses and data normalization  
`fcn_stimulation.py`: functions to setup stimulation input for model simulations



