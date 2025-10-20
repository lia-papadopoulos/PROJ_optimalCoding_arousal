
# Overview

`src_code/simulations_analysis/` contains code to analyze the model simulations. The code is organized into different subdirectories, where each subdirectory corresponds to a different analysis. Below, we provide a brief description of each subdirectory in `simulations_analysis/`. We then explain how to run one analysis in detail.

## Subdirectories

Each subdirectory in `simulations_analysis/` corresponds to a different analysis. 

### cellRates_vs_perturbation/

Contains code to compute relationships between single-cell firing rates and arousal level (associated with Fig. S2F,G; Fig. S3C).

1. `settings.py`: File that specifies which simulations to analyze, the analysis parameters, and all paths required to load functions and simulations and save results. 
2. `singelCell_tuning_to_perturbation.py`: Runs and saves the analysis for the parameters in `settings.py`

### clusterRates_numActiveClusters_vs_JeePlus/

Contains code to compute cluster firing rates and the number of active clusters as a function of the intracluster E-to-E connection strength, JeePlus (associated with Fig. S5B).

1. `clusterRates_numActiveClusters_vs_JeePlus_noStim_gainBased_settings.py`: File that specifies which simulations to analyze, the analysis parameters, and all paths required to load functions and simulations and save results. 
2. `clusterRates_numActiveClusters_vs_JeePlus_noStim_gainBased.py`: Main analysis script; loads the settings file and then runs and saves the analysis for a particular value of JeePlus.
3. `clusterRates_numActiveClusters_vs_JeePlus_noStim_gainBased_launchJobs.py`: Loads in simulation info based on the settings file, and for each value of JeePlus in the parameter sweep, runs `clusterRates_numActiveClusters_vs_JeePlus_noStim_gainBased.py` using task spooler. This enables the user to run parallel jobs on a computing cluster (one job for each value of JeePlus). The user must specify ahead of time the number of cores to use for each job and how many jobs to run simultaneously.

### clusterRates_numActiveClusters_vs_perturbation/

Contains code to compute cluster firing rates and the number of active clusters as a function of arousal (associated with Fig. 6B; Fig. S3F; Fig. S5C; ).

1. `clusterRates_numActiveClusters_vs_perturbation_gainBased_settings.py`: File that specifies which simulations to analyze, the analysis parameters, and all paths required to load functions and simulations and save results.  
2. `clusterRates_numActiveClusters_vs_perturbation_gainBased.py`: Main analysis script; loads the settings file and then runs and saves the analysis for a particular arousal level. 
3. `clusterRates_numActiveClusters_vs_perturbation_gainBased_launchJobs.py`: Loads in simulation info based on the settings file, and for each arousal level in the parameter sweep, runs `clusterRates_numActiveClusters_vs_perturbation_gainBased.py` using task spooler. This enables the user to run parallel jobs on a computing cluster (one job for each arousal level). The user must specify ahead of time the number of cores to use for each job and how many jobs to run simultaneously.

### clusterTimescale/

Contains code to compute cluster activation and interactive timescales as a function of arousal (associated with Fig. 6G; Fig. S36)

1. `clusterTimescale_vs_perturbation_settings.py`: File that specifies which simulations to analyze, the analysis parameters, and all paths required to load functions and simulations and save results.  
2. `clusterTimescale_vs_perturbation.py`: Main analysis script; loads the settings file and then runs and saves the analysis for a particular arousal level.  
3. `clusterTimescale_vs_perturbation_launchJobs.py`: Loads in simulation info based on the settings file, and for each arousal level in the parameter sweep, runs `clusterTimescale_vs_perturbation.py` using task spooler. This enables the user to run parallel jobs on a computing cluster (one job for each arousal level). The user must specify ahead of time the number of cores to use for each job and how many jobs to run simultaneously.


### decoding/

Contains code to run decoding analyses as a function of arousal and for different ensemble sizes (associated with Fig. 5E,F Fig. S3E; Fig. S4A,B)

1. `decode_settings.py`: File that specifies which simulations to analyze, the analysis parameters, and all paths required to load functions and simulations and save results.  
2. `decode_varyParam_master.py`: Main analysis script; loads the settings file and then runs and saves the analysis for a particular arousal level, network realization, and ensemble size. 
3. `decode_varyParam_launchJobs.py`: Loads in simulation info based on the settings file, then loops over arousal level, network realizations, and ensemble size and runs `decode_varyParam_master.py` using task spooler. This enables the user to run parallel jobs on a computing cluster (one job for each combination of the arousal level, network realization, and ensemble size). The user must specify ahead of time the number of cores to use for each job and how many jobs to run simultaneously.

### deltaRate_selective_nonSelective/

Contains code to run the cluster signal analysis (associated with Fig. 7B)

`deltaRate_selective_nonSelective.py`: Script that specifies all simulation and analysis parameters, and that runs and saves the analysis. 

### dprime/

Contains code to compute the single-cell neural discriminability index as a function of arousal (associated with Fig. 5C,D; Fig. S3D)

1. `dPrime_settings.py`: File that specifies which simulations to analyze, the analysis parameters, and all paths required to load functions and simulations and save results.  
2. `singleCell_dPrime.py`:  Main analysis script; loads the settings file and then runs and saves the analysis for particular arousal level.
3. `singleCell_dPrime_launchJobs.py`: Loads in simulation info based on the settings file and for each arousal level, runs `singleCell_dPrime.py` using task spooler. This enables the user to run parallel jobs on a computing cluster (one job for each arousal level). The user must specify ahead of time the number of cores to use for each job and how many jobs to run simultaneously.

### fano_factor/

Contains code to run the Fano factor analyses as a function of arousal (associated with Fig. 8A-C; Fig. S3H; Fig.S7H-J)

1. `fcn_plot_fanofactor.py`: Set of functions to help with plotting Fano factor (used in `src_code/manuscript_plotting_scripts/`)
2. `FF_settings.py`: File that specifies which simulations to analyze, the analysis parameters, and all paths required to load functions and simulations and save results.  
3. `FF_vs_arousal.py`: Main analysis script; loads the settings file and then runs and saves the Fano factor analysis for a particular arousal level.  
4. `FF_vs_arousal_launchJobs.py`: Loads in simulation info based on the settings file, then loops over arousal levels and runs `FF_vs_arousal.py` using task spooler. This enables the user to run parallel jobs on a computing cluster (one job for each arousal level). The user must specify ahead of time the number of cores to use for each job and how many jobs to run simultaneously.

### numActive_targeted_nontargeted_clusters/

Contains code to run the cluster reliability analysis (associated with Fig. 7C), as well as several supplementary analyses that are not in the manuscript.

1. `settings.py`: File that specifies which simulations to analyze, the analysis parameters, and all paths required to load functions and simulations and save results.  
2.  `numActive_targeted_nontargeted_clusters_vs_perturbation_gainBased.py`: Main analysis script; loads the settings file and then runs and saves the analysis for particular arousal level.
3. `numActive_targeted_nontargeted_clusters_vs_perturbation_gainBased_launchJobs.py`: Loads in simulation info based on the settings file and for each arousal level, runs `numActive_targeted_nontargeted_clusters_vs_perturbation_gainBased_launchJobs.py` using task spooler. This enables the user to run parallel jobs on a computing cluster (one job for each arousal level). The user must specify ahead of time the number of cores to use for each job and how many jobs to run simultaneously.

### psth/

Contains code to compute the amplitude and significance of stimulus-evoked responses using trials combined across all arousal levels. The results of this analysis are used to compute the pairwise tuning similarity and to determine the set of the cells that respond significantly to at least one stimulus for the clustering analyses.

1. `psth_settings.py`: File that specifies which simulations to analyze, the analysis parameters, and all paths required to load functions and simulations and save results.  
2. `compute_psth.py`: Main analysis script; loads the settings file and then runs and saves the analysis for a particular network and stimulus realization.
3. `compute_psth_launchJobs.py`:  Loads in simulation info based on the settings file, then runs `compute_psth.py` (using task spooler) for each network and stimulus realization. This enables the user to run parallel jobs on a computing cluster (one job for each combination of network and stimulus realization). The user must specify ahead of time the number of cores to use for each job and how many jobs to run simultaneously.


## Usage


### Running batch simulations

`masterSim.py` and `launchJobs.py` are used to run a complete set of batch simulations associated with one of the parameter sets described above. A given parameters file (e.g., `simParams_051325_clu.py`) contains all model and simulation parameters required for running the batch simulations; this includes specifying which model parameters should be swept over (e.g., to simulate different levels of arousal), as well how many network realizations, initial conditions, and stimulus realizations to run for a fixed set of parameters. `launchJobs.py` takes in the specified parameters file, and for each combination of the swept parameter values, network realization, and initial conditions, it runs `masterSim.py` using task spooler. `masterSim.py` then runs the network simulation for the given parameters and saves the results in a specified output directory (note that the loop over the stimulus realization is done within `masterSim.py`, so this part is not parallelized). The user must specify ahead of time the number of cores to use for each job (i.e., for each call of `masterSim.py`), as well as how many jobs to run simultaneously. The basic steps for running batch simulations are:  


1. Open `../global_settings.py` and set global absolute paths for your project  
2. Open `paths_file.py` and set:  
	a. 'sim_params_name' (name of simulation parameters file that you want to run)    
	b. 'save_path' (where to save simulation output)  
3. Specify the number of cores/job and the number of jobs to run simultaneously using task spooler  
4. Run:
```
$ python launchJobs.py
``` 
