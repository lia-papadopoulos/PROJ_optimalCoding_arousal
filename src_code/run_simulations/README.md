
# Overview

`src_code/run_simulations/` contains code and parameter sets for running the model simulations in the paper

## Files

Below is a brief description of the files in `run_simulations/`

### Files that specify parameter sets associated with different analyses

1. `simParams_041725_clu_varyJEEplus.py`: simulation parameters associated with Fig. S5A,B  
2. `simParams_050925_clu.py`: simulation parameters associated with Fig. S3  
3. `simParams_051325_clu_spont.py`: simulation parameters associated with Fig. S6A-C  
4. `simParams_051325_clu_spontLong.py`: simulation parameters associated with Fig. S6G  
5. `simParams_051325_clu.py`: simulation parameters associated with Fig. 3A,C; Fig. 4A-D, Fig. 5C,E, Fig. 6A,B,F, Fig. 7A-C, Fig. 8A-C, Fig. S2F, Fig. S4A, Fig. S5C, Fig. S7H-J
6. `simParams_051325_hom.py`: simulation parameters associated with Fig. 3B,C; Fig. 4A, Fig. 5D,F, Fig. S2G, Fig. S4B  

### Files for simulation setup  
    
`fcn_simulation_setup.py`: set of functions that are used to set up simulations given model parameters
   
`paths_file.py`: specifies paths and which set of parameters to use for simulations  

### Files for running simulations

`run_testSimulation.py`: script for running a single simulation (see below for usage)  

`masterSim.py`: main script for running batch simulations (see below for usage)

`launchJobs.py`: calls `masterSim.py` and submits jobs to a computing cluster using task spooler (see below for usage)

## Usage

### Running a test simulation 

`run_testSimulation.py` is used to run a single simulation associated with one of the parameter sets described above. The basic steps to do this are:  

1. Open `../global_settings.py` and set global absolute paths for your project  
2. Open `paths_file.py` and set:  
	a. 'sim_params_name' (name of simulation parameters file that you want to run)  
	b. 'save_path' (where to save simulation output)  
3. Open `run_testSimulation.py` and specify:  
    'externalInput_seed': random seed for setting up external inputs  
    'stimClusters_seed' and 'stimNeurons_seed': random seeds for setting up stimulation  
    'networkSeed': random seed for setting up the network connectivity  
    'arousalLevel': level of arousal for the simulation (between 0 and 1  )
3. Run:  
```
$ python run_testSimulation.py
``` 
    

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
