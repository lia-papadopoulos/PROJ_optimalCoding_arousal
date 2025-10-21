
# Overview

`src_code/simulations_analysis/` contains code to analyze the model simulations. The code is organized into different subdirectories, where each subdirectory corresponds to a different analysis. Below, we provide a brief description of each subdirectory in `simulations_analysis/`. We then explain how to run one analysis in detail. Note that all code is setup to submit batch jobs using task-spooler.

## Subdirectories

Each subdirectory in `simulations_analysis/` corresponds to a different analysis. 

### cellRates_vs_perturbation/

Contains code to compute relationships between single-cell firing rates and arousal level (associated with Fig. S2F,G; Fig. S3C).

1. `settings.py`: File that specifies which simulations to analyze, the analysis parameters, and all paths required to load functions and simulations and save results. 
2. `singelCell_tuning_to_perturbation.py`: Runs and saves the analysis for the parameters in `settings.py`

### clusterRates_numActiveClusters_vs_JeePlus/

Contains code to compute cluster firing rates and the number of active clusters as a function of the intracluster E-to-E connection strength, JeePlus (associated with Fig. S5B).

1. `clusterRates_numActiveClusters_vs_JeePlus_noStim_gainBased_settings.py`: File that specifies which simulations to analyze, the analysis parameters, and all paths required to load functions and simulations and save results. This is the only file that needs to be changed by the user.
2. `clusterRates_numActiveClusters_vs_JeePlus_noStim_gainBased.py`: Main analysis script; loads the settings file and then runs and saves the analysis for a specified value of JeePlus.
3. `clusterRates_numActiveClusters_vs_JeePlus_noStim_gainBased_launchJobs.py`: Loads in simulation info based on the settings file, and for each value of JeePlus in the parameter sweep, runs `clusterRates_numActiveClusters_vs_JeePlus_noStim_gainBased.py` using task spooler. This enables the user to run parallel jobs on a computing cluster (one job for each value of JeePlus). The user must specify ahead of time the number of cores to use for each job and how many cores can be used simultaneously; these are then used to set the number of simultaneous jobs.

### clusterRates_numActiveClusters_vs_perturbation/

Contains code to compute cluster firing rates and the number of active clusters as a function of arousal (associated with Fig. 6B; Fig. S3F; Fig. S5C; ).

1. `clusterRates_numActiveClusters_vs_perturbation_gainBased_settings.py`: File that specifies which simulations to analyze, the analysis parameters, and all paths required to load functions and simulations and save results. This is the only file that needs to be changed by the user.
2. `clusterRates_numActiveClusters_vs_perturbation_gainBased.py`: Main analysis script; loads the settings file and then runs and saves the analysis for a specified arousal level. 
3. `clusterRates_numActiveClusters_vs_perturbation_gainBased_launchJobs.py`: Loads in simulation info based on the settings file, and for each arousal level in the parameter sweep, runs `clusterRates_numActiveClusters_vs_perturbation_gainBased.py` using task spooler. This enables the user to run parallel jobs on a computing cluster (one job for each arousal level). The user must specify ahead of time the number of cores to use for each job and how many cores can be used simultaneously; these are then used to set the number of simultaneous jobs.

### clusterTimescale/

Contains code to compute cluster activation and interactive timescales as a function of arousal (associated with Fig. 6G; Fig. S36)

1. `clusterTimescale_vs_perturbation_settings.py`: File that specifies which simulations to analyze, the analysis parameters, and all paths required to load functions and simulations and save results. This is the only file that needs to be changed by the user.
2. `clusterTimescale_vs_perturbation.py`: Main analysis script; loads the settings file and then runs and saves the analysis for a specified arousal level.  
3. `clusterTimescale_vs_perturbation_launchJobs.py`: Loads in simulation info based on the settings file, and for each arousal level in the parameter sweep, runs `clusterTimescale_vs_perturbation.py` using task spooler. This enables the user to run parallel jobs on a computing cluster (one job for each arousal level). The user must specify ahead of time the number of cores to use for each job and how many cores can be used simultaneously; these are then used to set the number of simultaneous jobs.


### decoding/

Contains code to run decoding analyses as a function of arousal and for different ensemble sizes (associated with Fig. 5E,F Fig. S3E; Fig. S4A,B)

1. `decode_settings.py`: File that specifies which simulations to analyze, the analysis parameters, and all paths required to load functions and simulations and save results.  This is the only file that needs to be changed by the user.
2. `decode_varyParam_master.py`: Main analysis script; loads the settings file and then runs and saves the analysis for a specified arousal level, network realization, and ensemble size. 
3. `decode_varyParam_launchJobs.py`: Loads in simulation info based on the settings file, then loops over arousal level, network realizations, and ensemble size and runs `decode_varyParam_master.py` using task spooler. This enables the user to run parallel jobs on a computing cluster (one job for each combination of the arousal level, network realization, and ensemble size). The user must specify ahead of time the number of cores to use for each job and how many cores can be used simultaneously; these are then used to set the number of simultaneous jobs.

### deltaRate_selective_nonSelective/

Contains code to run the cluster signal analysis (associated with Fig. 7B)

`deltaRate_selective_nonSelective.py`: Script that specifies all simulation and analysis parameters, and that runs and saves the analysis. 

### dprime/

Contains code to compute the single-cell neural discriminability index as a function of arousal (associated with Fig. 5C,D; Fig. S3D)

1. `dPrime_settings.py`: File that specifies which simulations to analyze, the analysis parameters, and all paths required to load functions and simulations and save results.  This is the only file that needs to be changed by the user.
2. `singleCell_dPrime.py`:  Main analysis script; loads the settings file and then runs and saves the analysis for a specified arousal level.
3. `singleCell_dPrime_launchJobs.py`: Loads in simulation info based on the settings file and for each arousal level, runs `singleCell_dPrime.py` using task spooler. This enables the user to run parallel jobs on a computing cluster (one job for each arousal level). The user must specify ahead of time the number of cores to use for each job and how many cores can be used simultaneously; these are then used to set the number of simultaneous jobs.

### fano_factor/

Contains code to run the Fano factor analyses as a function of arousal (associated with Fig. 8A-C; Fig. S3H; Fig.S7H-J)

1. `fcn_plot_fanofactor.py`: Set of functions to help with plotting Fano factor (used in `src_code/manuscript_plotting_scripts/`)
2. `FF_settings.py`: File that specifies which simulations to analyze, the analysis parameters, and all paths required to load functions and simulations and save results.  This is the only file that needs to be changed by the user.
3. `FF_vs_arousal.py`: Main analysis script; loads the settings file and then runs and saves the Fano factor analysis for a specified arousal level.  
4. `FF_vs_arousal_launchJobs.py`: Loads in simulation info based on the settings file, then loops over arousal levels and runs `FF_vs_arousal.py` using task spooler. This enables the user to run parallel jobs on a computing cluster (one job for each arousal level). The user must specify ahead of time the number of cores to use for each job and how many cores can be used simultaneously; these are then used to set the number of simultaneous jobs.

### numActive_targeted_nontargeted_clusters/

Contains code to run the cluster reliability analysis (associated with Fig. 7C), as well as several supplementary analyses that are not in the manuscript.

1. `settings.py`: File that specifies which simulations to analyze, the analysis parameters, and all paths required to load functions and simulations and save results.  This is the only file that needs to be changed by the user.
2.  `numActive_targeted_nontargeted_clusters_vs_perturbation_gainBased.py`: Main analysis script; loads the settings file and then runs and saves the analysis for a specified arousal level.
3. `numActive_targeted_nontargeted_clusters_vs_perturbation_gainBased_launchJobs.py`: Loads in simulation info based on the settings file and for each arousal level, runs `numActive_targeted_nontargeted_clusters_vs_perturbation_gainBased_launchJobs.py` using task spooler. This enables the user to run parallel jobs on a computing cluster (one job for each arousal level). The user must specify ahead of time the number of cores to use for each job and how many cores can be used simultaneously; these are then used to set the number of simultaneous jobs.

### psth/

Contains code to compute the amplitude and significance of stimulus-evoked responses using trials combined across all arousal levels. The results of this analysis are used to compute the pairwise tuning similarity and to determine the set of the cells that respond significantly to at least one stimulus for the clustering analyses.

1. `psth_settings.py`: File that specifies which simulations to analyze, the analysis parameters, and all paths required to load functions and simulations and save results.  This is the only file that needs to be changed by the user.
2. `compute_psth.py`: Main analysis script; loads the settings file and then runs and saves the analysis for a specified network and stimulus realization.
3. `compute_psth_launchJobs.py`:  Loads in simulation info based on the settings file, then runs `compute_psth.py` (using task spooler) for each network and stimulus realization. This enables the user to run parallel jobs on a computing cluster (one job for each combination of network and stimulus realization). The user must specify ahead of time the number of cores to use for each job and how many cores can be used simultaneously; these are then used to set the number of simultaneous jobs.


### spikeCount_correlations/

Contains code to compute spike count correlations and run the hierarchical clustering analysis (associated with Fig. 4A-D).

1. `evoked_corr_settings.py`: File that specifies which simulations to analyze, the analysis parameters, and all paths required to load functions and simulations and save results. This is the only file that needs to be changed by the user.
2. `evoked_corr.py.`:  Main analysis script for computing spike-count correlations; loads the settings file and then runs and saves the analysis for a specified network realization.
3.  `evoked_corr_launchJobs.py`: Loads in simulation info based on the settings file, then runs `evoked_corr.py` (using task spooler) for each network realization. This enables the user to run parallel jobs on a computing cluster (one job per network realization). The user must specify ahead of time the number of cores to use for each job and how many cores can be used simultaneously; these are then used to set the number of simultaneous jobs.
4. `run_hClustering_evoked.py`: Script to run hierarchical clustering; loads the settings file and then runs hierarchical clustering on the correlation matrices generated by `evoked_corr.py`.


### spont_cvISI/

Contains code to compute the coefficient of variation of interspike intervals (during spontaneous activity) as a function of arousal (associated with Fig. S6A).

1. `settings.py`: File that specifies which simulations to analyze, the analysis parameters, and all paths required to load functions and simulations and save results.  This is the only file that needs to be changed by the user.
2. `spont_cvISI.py`:  Main analysis script; loads the settings file and then runs and saves the analysis for a specified arousal level.
3. `spont_cvISI_launchJobs.py`: Loads in simulation info based on the settings file; for each arousal level, then runs `spont_cvISI.py` using task spooler. This enables the user to run parallel jobs on a computing cluster (one job for each arousal level).The user must specify ahead of time the number of cores to use for each job and how many cores can be used simultaneously; these are then used to set the number of simultaneous jobs.


### spont_spikeSpectra/

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

    1. For Fig. 5C:
    
```
    simParams_fname = 'simParams_051325_clu'
    sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'
    net_type = 'baseEIclu'
    nNetworks = 10   
```

    2. For Fig. 5D:
    
```
    simParams_fname = 'simParams_051325_hom'
    sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'
    net_type = 'baseHOM'
    nNetworks = 10   
```    

    3. For Fig. S3D:
    
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
