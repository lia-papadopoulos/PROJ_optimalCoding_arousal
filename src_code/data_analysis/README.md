
# Overview

`src_code/data_analysis/` contains code to analyze the neural data. The code is organized into different subdirectories, where each subdirectory corresponds to a different analysis. `data_analysis/` also contains a few modules whose functions are imported and used in many of the analyses and/or corresponding plotting scripts. Below, we provide a brief description of the modules and each subdirectory in `data_analysis/`. We then explain how to run one analysis in detail. Note that all code is setup to submit batch jobs using task-spooler.

## Modules used to aid the data analysis

1. `fcn_processedh5data_to_dict.py`: Loads in single-session neural and behavioral data from an .h5 file and outputs a dictionary for the session that contains all relevant information for downstream analyses (e.g., pupil trace, cell spike times, stimulus onset times, etc). This function is called at the beginning of every analysis script, and the resulting "data dictionary" is the starting point for subsequent analyses. 
2. `fcn_SuData.py`: Set of functions that aid in the analysis of the neural and behavioral data from a particular recording session. Many of the functions in `fcn_SuData.py` take a session's data dictionary as input (i.e., the output of `fcn_processedh5data_to_dict.py`), perform some computation, and then store the results as a new key-value pair in the dictionary.
3. `fcn_SuData_analysis.py`: The functions in this module take in the results of various analyses and perform additional computations (e.g., perform session-averaging).

## Subdirectories

Each subdirectory in `data_analysis/` performs a different analysis on the neural and/or behavioral data and saves the results to a specified output directory. 


### `cv_isi_vs_pupil/`

Contains code to compute the coefficient of variation of interspike intervals (during spontaneous activity) as a function of arousal (associated with Fig. S6D-G).

1. `isiCV_vs_pupilPercentile_settings.py`: File that specifies which sessions to analyze, the analysis parameters, and all paths required to load data and functions and save results.  This is the only file that needs to be changed by the user.
2. `isiCV_vs_pupilPercentile.py`:  Main analysis script; loads the settings file and then runs and saves the analysis for a specified session.
3. `isiCV_vs_pupilPercentile_launchJobs.py`: Loads in settings file and for each session, runs `isiCV_vs_pupilPercentile.py` using task spooler. This enables the user to run parallel jobs on a computing cluster (one job for each session). The user must specify ahead of time the number of cores to use for each job and how many cores can be used simultaneously; these are then used to set the number of simultaneous jobs.


### `decoding/`

Contains code to run decoding analyses as a function of arousal and for different ensemble sizes (associated with Fig. 2E,H,I; Fig. S1B-D,F,G; Fig. S8C,D)

1. `decoding_params.py`: File that specifies which sessions to analyze, the analysis parameters, and all paths required to load data and functions and save results.  This is the only file that needs to be changed by the user.
2. `decode_pupil.py`: Main analysis script; loads the params file and then runs and saves the analysis for a specified session.
3. `decode_pupil_launchJobs.py`: Loads in params file and for each session, runs `decode_pupil.py` using task spooler. This enables the user to run parallel jobs on a computing cluster (one job for each session). The user must specify ahead of time the number of cores to use for each job and how many cores can be used simultaneously; these are then used to set the number of simultaneous jobs.


### `fanofactor_vs_pupil/`

Contains code to run the Fano factor analyses as a function of arousal (associated with Fig. 8D-I; Fig. S7A-G; Fig. S8G-L)

1. `fano_factor_settings.py`: File that specifies which sessions to analyze, the analysis parameters, and all paths required to load data and functions and save results.  This is the only file that needs to be changed by the user.
2. `fanofactor_all_pupilPercentile_rawSpontEvoked.py`: Main analysis script for computing the spontaneous and evoked Fano factor using data combined across pupil bins; loads the settings file and then runs and saves the analysis for a specified session.
3. `fanofactor_vs_pupilPercentile_rawSpontEvoked.py`: Main analysis script for computing the spontaneous and evoked Fano factor as a function of pupil diameter/arousal; loads the settings file and then runs and saves the analysis for a specified session.
4. `fanofactor_vs_pupilPercentile_rawSpontEvoked_varyWindowSize.py`: Main analysis script for computing the spontaneous and evoked Fano factor as a function of pupil diameter/arousal and for different window sizes; loads the settings file and then runs and saves the analysis for a specified session. The same number of trials are used for all window sizes.
5. `fanofactor_all_pupilPercentile_rawSpontEvoked_launchJobs.py`: Loads in settings file and for each session, runs `fanofactor_all_pupilPercentile_rawSpontEvoked.py` using task spooler. This enables the user to run parallel jobs on a computing cluster (one job for each session). The user must specify ahead of time the number of cores to use for each job and how many cores can be used simultaneously; these are then used to set the number of simultaneous jobs.
6. `fanofactor_vs_pupilPercentile_rawSpontEvoked_launchJobs.py`: Loads in params file and for each session, runs `fanofactor_vs_pupilPercentile_rawSpontEvoked.py` using task spooler. This enables the user to run parallel jobs on a computing cluster (one job for each session). The user must specify ahead of time the number of cores to use for each job and how many cores can be used simultaneously; these are then used to set the number of simultaneous jobs.
7. `fanofactor_vs_pupilPercentile_rawSpontEvoked_varyWindowSize_launchJobs.py`: Loads in params file and for each session, runs `fanofactor_vs_pupilPercentile_rawSpontEvoked_varyWindowSize.py` using task spooler. This enables the user to run parallel jobs on a computing cluster (one job for each session). The user must specify ahead of time the number of cores to use for each job and how many cores can be used simultaneously; these are then used to set the number of simultaneous jobs.
8. `fcn_plot_fanofactor.py`: Set of functions to help with plotting Fano factor (used in `src_code/manuscript_plotting_scripts/`)






### `cellRates_vs_perturbation/`

Contains code to compute relationships between single-cell firing rates and arousal level (associated with Fig. S2F,G; Fig. S3C).

1. `settings.py`: File that specifies which simulations to analyze, the analysis parameters, and all paths required to load functions and simulations and save results. 
2. `singelCell_tuning_to_perturbation.py`: Runs and saves the analysis for the parameters in `settings.py`



### `dprime/`

Contains code to compute the single-cell neural discriminability index as a function of arousal (associated with Fig. 5C,D; Fig. S3D)

1. `dPrime_settings.py`: File that specifies which simulations to analyze, the analysis parameters, and all paths required to load functions and simulations and save results.  This is the only file that needs to be changed by the user.
2. `singleCell_dPrime.py`:  Main analysis script; loads the settings file and then runs and saves the analysis for a specified arousal level.
3. `singleCell_dPrime_launchJobs.py`: Loads in simulation info based on the settings file and for each arousal level, runs `singleCell_dPrime.py` using task spooler. This enables the user to run parallel jobs on a computing cluster (one job for each arousal level). The user must specify ahead of time the number of cores to use for each job and how many cores can be used simultaneously; these are then used to set the number of simultaneous jobs.


### `psth/`

Contains code to compute the amplitude and significance of stimulus-evoked responses using trials combined across all arousal levels. The results of this analysis are used to compute the pairwise tuning similarity and to determine the set of the cells that respond significantly to at least one stimulus for the clustering analyses.

1. `psth_settings.py`: File that specifies which simulations to analyze, the analysis parameters, and all paths required to load functions and simulations and save results.  This is the only file that needs to be changed by the user.
2. `compute_psth.py`: Main analysis script; loads the settings file and then runs and saves the analysis for a specified network and stimulus realization.
3. `compute_psth_launchJobs.py`:  Loads in simulation info based on the settings file, then runs `compute_psth.py` (using task spooler) for each network and stimulus realization. This enables the user to run parallel jobs on a computing cluster (one job for each combination of network and stimulus realization). The user must specify ahead of time the number of cores to use for each job and how many cores can be used simultaneously; these are then used to set the number of simultaneous jobs.

### `spikeCount_correlations/`

Contains code to compute spike count correlations and run the hierarchical clustering analysis (associated with Fig. 4A-D).

1. `evoked_corr_settings.py`: File that specifies which simulations to analyze, the analysis parameters, and all paths required to load functions and simulations and save results. This is the only file that needs to be changed by the user.
2. `evoked_corr.py.`:  Main analysis script for computing spike-count correlations; loads the settings file and then runs and saves the analysis for a specified network realization.
3.  `evoked_corr_launchJobs.py`: Loads in simulation info based on the settings file, then runs `evoked_corr.py` (using task spooler) for each network realization. This enables the user to run parallel jobs on a computing cluster (one job per network realization). The user must specify ahead of time the number of cores to use for each job and how many cores can be used simultaneously; these are then used to set the number of simultaneous jobs.
4. `run_hClustering_evoked.py`: Script to run hierarchical clustering; loads the settings file and then runs hierarchical clustering on the correlation matrices generated by `evoked_corr.py`.


### `spont_spikeSpectra/`

Contains code to compute the spike spectrum of individual cells as a function of arousal (associated with Fig. S6B,C).

1. `spont_spikeSpectra_settings.py`: File that specifies which simulations to analyze, the analysis parameters, and all paths required to load functions and simulations and save results.  This is the only file that needs to be changed by the user.
2. `spont_spikeSpectra.py`: Main analysis script; loads the settings file and then runs and saves the analysis for specified arousal level.
3. `spont_spikeSpectra_launchJobs.py`: Loads in simulation info based on the settings file and for each arousal level, runs `spont_spikeSpectra.py` using task spooler. This enables the user to run parallel jobs on a computing cluster (one job for each arousal level). The user must specify ahead of time the number of cores to use for each job and how many cores can be used simultaneously; these are then used to set the number of simultaneous jobs.

## Example

### Computing the single-cell neural discriminability index as a function of arousal (associated with Fig. 5C,D; Fig. S3D)

1. Open `src_code/global_settings.py` and set global absolute paths for your project 
2. Make output directory `singlCell_dprime/` inside `global_settings.path_to_sim_output/`
3. Configure computing cluster to use desired number of cores/job
4. Navigate to `src_code/simulations_analysis/dprime/` and open `dPrime_settings.py`
5. Set required paths to functions, simulations, etc. Asumming default directory structure, these should be:

```
    sim_params_path = global_settings.path_to_src_code + 'run_simulations/'
    func_path = global_settings.path_to_src_code + 'functions/'
    func_path0 = global_settings.path_to_src_code + 'run_simulations/'
    load_path = global_settings.path_to_sim_output + ''
    save_path = global_settings.path_to_sim_output + 'singleCell_dPrime/'
```

6. Specify parameters that determine which set of simulations to analyze.

&nbsp; &nbsp; &nbsp; &nbsp; a. For Fig. 5C:
    
```
    simParams_fname = 'simParams_051325_clu'
    sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'
    net_type = 'baseEIclu'
    nNetworks = 10   
```

&nbsp; &nbsp; &nbsp; &nbsp; b. For Fig. 5D:

```
    simParams_fname = 'simParams_051325_hom'
    sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'
    net_type = 'baseHOM'
    nNetworks = 10   
```    

&nbsp; &nbsp; &nbsp; &nbsp; c. For Fig. S3D:
    
```
    simParams_fname = 'simParams_050925_clu'
    sweep_param_name = 'zeroMean_sd_nu_ext_ee'
    net_type = 'baseEIclu'
    nNetworks = 5   
```    

7. Set analysis parameters.

```
    windL = 100e-3
    windStep = 20e-3
```

8. Set `cores_per_job` (number of cores to use/job) and `maxCores` (total number of cores that can be used simultaneously) according to desired cluster usage

9. Run:

```
    $ python singleCell_dPrime_launchJobs.py
``` 
