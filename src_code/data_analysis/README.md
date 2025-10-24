
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
6. `fanofactor_vs_pupilPercentile_rawSpontEvoked_launchJobs.py`: Loads in settings file and for each session, runs `fanofactor_vs_pupilPercentile_rawSpontEvoked.py` using task spooler. This enables the user to run parallel jobs on a computing cluster (one job for each session). The user must specify ahead of time the number of cores to use for each job and how many cores can be used simultaneously; these are then used to set the number of simultaneous jobs.
7. `fanofactor_vs_pupilPercentile_rawSpontEvoked_varyWindowSize_launchJobs.py`: Loads in settings file and for each session, runs `fanofactor_vs_pupilPercentile_rawSpontEvoked_varyWindowSize.py` using task spooler. This enables the user to run parallel jobs on a computing cluster (one job for each session). The user must specify ahead of time the number of cores to use for each job and how many cores can be used simultaneously; these are then used to set the number of simultaneous jobs.
8. `fcn_plot_fanofactor.py`: Set of functions to help with plotting Fano factor (used in `src_code/manuscript_plotting_scripts/`)

### `n_goodUnits_eachSession/`

1. `n_goodUnits_eachSession.py`: Code to compute the number of cells in each session.

### `psth_allTrials/`

Contains code to compute the amplitude and significance of stimulus-evoked responses using trials combined across all arousal levels. The results of this analysis are used to to determine the set of the cells that respond significantly to at least one stimulus and to compute the pairwise tuning similarity for the clustering analysis.

1. `psth_allTrials_settings.py`: File that specifies which sessions to analyze, the analysis parameters, and all paths required to load data and functions and save results. This is the only file that needs to be changed by the user.
2. `psth_allTrials.py`: Main analysis script; loads the settings file and then runs and saves the analysis for a specified session.
3. `psth_allTrials_launchJobs.py`:  Loads in settings file and for each session, runs `psth_allTrials.py` using task spooler. This enables the user to run parallel jobs on a computing cluster (one job for each session). The user must specify ahead of time the number of cores to use for each job and how many cores can be used simultaneously; these are then used to set the number of simultaneous jobs.

### `pupil_rate_correlation/`

Contains code to assess relationship between pupil diameter and spontaneous single-cell firing rates (associated with Fig. S2A-E).

1. `pupil_rate_correlation_settings.py`: ile that specifies which sessions to analyze, the analysis parameters, and all paths required to load data and functions and save results. This is the only file that needs to be changed by the user.
2. `pupil_rate_correlation.py`: Main analysis script; loads the settings file and then runs and saves the analysis for a specified session.
3. `pupil_rate_correlation_launchJobs.py`: Loads in settings file and for each session, runs `pupil_rate_correlation.py` using task spooler. This enables the user to run parallel jobs on a computing cluster (one job for each session). The user must specify ahead of time the number of cores to use for each job and how many cores can be used simultaneously; these are then used to set the number of simultaneous jobs.

### `singleCell_dPrime/`

Contains code to compute the single-cell neural discriminability index as a function of arousal (associated with Fig. 2C,F,G; Fig. S1A; Fig. S8A,B)

1. `singleCell_dPrime_settings.py`: File that specifies which sessions to analyze, the analysis parameters, and all paths required to load data and functions and save results. This is the only file that needs to be changed by the user.
2. `singleCell_dPrime_vs_pupil.py`:  Main analysis script; loads the settings file and then runs and saves the analysis for a specified session.
3. `singleCell_dPrime_vs_pupil_launchJobs.py`: Loads in settings file and for each session, runs `singleCell_dPrime_vs_pupil.py` using task spooler. This enables the user to run parallel jobs on a computing cluster (one job for each session). The user must specify ahead of time the number of cores to use for each job and how many cores can be used simultaneously; these are then used to set the number of simultaneous jobs.


### `spikeCount_correlations/`

Contains code to compute spike count correlations and run the hierarchical clustering analysis (associated with Fig. 4E-K; Fig. S8E,F).

1. `evoked_corr_settings.py`: File that specifies which sessions to analyze, the analysis parameters, and all paths required to load data and functions and save results. This is the only file that needs to be changed by the user.
2. `evoked_corr.py.`:  Main analysis script for computing spike-count correlations; loads the settings file and then runs and saves the analysis for a specified session.
3.  `evoked_corr_launchJobs.py`: Loads in settings file and for each session, runs `evoked_corr.py` using task spooler. This enables the user to run parallel jobs on a computing cluster (one job for each session). The user must specify ahead of time the number of cores to use for each job and how many cores can be used simultaneously; these are then used to set the number of simultaneous jobs.
4. `run_hClustering_evoked.py`: Script to run hierarchical clustering; loads the settings file and then runs hierarchical clustering on the correlation matrices generated by `evoked_corr.py`. Also requires that you have already run `psth_allTrials.py` (see above).


### `spont_spikeSpectra_pupil/`

Contains code to compute the spike spectrum of individual cells as a function of arousal (associated with Fig. S6H-L).

1. `spont_spikeSpectra_settings.py`: File that specifies which sessions to analyze, the analysis parameters, and all paths required to load data and functions and save results. This is the only file that needs to be changed by the user.
2. `spont_spikeSpectra_pupilPercentile.py`: Main analysis script; loads the settings file and then runs and saves the analysis for a specified session.
3. `spont_spikeSpectra_pupilPercentile_launchJobs.py`: Loads in settings file and for each session, runs `spont_spikeSpectra_pupilPercentile.py` using task spooler. This enables the user to run parallel jobs on a computing cluster (one job for each session). The user must specify ahead of time the number of cores to use for each job and how many cores can be used simultaneously; these are then used to set the number of simultaneous jobs.


## Example

### Computing the single-cell neural discriminability index as a function of arousal (associated with Fig. 2C,F,G; Fig. S1A; Fig. S8A,B)

1. Open `src_code/global_settings.py` and set global absolute paths for your project 
2. Make output directory `singleCell_dPrime/` inside `global_settings.path_to_data_analysis_output/`
3. Configure computing cluster to use desired number of cores/job
4. Navigate to `src_code/data_analysis/singleCell_dprime/` and open `dPrime_settings.py`

5. Set required paths to functions, data, etc. Asumming default directory structure, these should be:

```
    data_path = global_settings.path_to_processed_data
    outpath = global_settings.path_to_data_analysis_output + 'singleCell_dPrime/'
    func_path1 = global_settings.path_to_src_code + 'data_analysis/'      
    func_path2 = global_settings.path_to_src_code + 'functions/' 
```

6. Specify which sessions to run. To run all sessions:

```
sessions_to_run = ['LA3_session3', \
                   'LA8_session1', 'LA8_session2', \
                   'LA9_session1', 'LA9_session3', 'LA9_session4', 'LA9_session5', \
                   'LA11_session1', 'LA11_session2', 'LA11_session3', 'LA11_session4', \
                   'LA12_session1', 'LA12_session2', 'LA12_session3', 'LA12_session4']
```

7. Set `maxCores` (total number of cores that can be used simultaneously) and `cores_per_job` (number of cores to use/job) according to desired cluster usage. Make sure that you have configured your computing cluster to use the number of cores per job that you set here.

8. Specify analysis parameters

&nbsp; &nbsp; &nbsp; &nbsp; For Fig. S2C,F,G and S1A:
    
```
    stim_duration = 25e-3
    trial_window = [-100e-3, 450e-3]
    window_length = 100e-3
    window_step = 10e-3
    pupilBlock_size = 0.1
    pupilBlock_step = 0.1
    pupilSplit_method = 'percentile'
    pupilSize_method = 'avgSize_beforeStim'
    n_subsamples = 100
    nTrials_thresh = 20
    restOnly = False
    trialMatch = False
    runThresh = 2.
    runSpeed_method = 'avgSize_beforeStim'
    runBlock_size = 1.
    runBlock_step = 1.
    runSplit_method = 'percentile'
    global_pupilNorm = False
    cellSelection = ''
    highDownSample = False
```

9. Run:

```
    $ python singleCell_dPrime_vs_pupil_launchJobs.py
``` 
