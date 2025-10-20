#%% IMPORTS
import os
import numpy as np
import sys
import importlib

#%% LOAD REQUIRED FUNCTIONS FROM SETTINGS FILE
import clusterRates_numActiveClusters_vs_JeePlus_noStim_gainBased_settings as settings
func_path1 = settings.func_path1
func_path2 = settings.func_path2
sim_params_path = settings.sim_params_path
sys.path.append(func_path1)
from fcn_simulation_loading import fcn_set_sweepParam_string
sys.path.append(func_path2)
from fcn_simulation_setup import fcn_define_arousalSweep

#%% GET RELEVANT SETTINGS
simParams_fname = settings.simParams_fname
sweep_param_name = settings.sweep_param_name
maxCores = settings.maxCores
cores_per_job = settings.cores_per_job

#%% LOAD IN SIMULATION PARAMETERS
sys.path.append(sim_params_path)
params = importlib.import_module(simParams_fname) 
s_params = params.sim_params

#%% SETUP THE PARAMETER SWEEP BASED ON LOADED PARAMETERS
s_params = fcn_define_arousalSweep(s_params)
    
#%% UNPACK PARAMETER SWEEP
n_sweepParams = s_params['nParams_sweep']
swept_params_dict = s_params['swept_params_dict']

del s_params
del params


#%% SET UP FOR TASK SPOOLER

# tell task-spooler how many jobs it can run simultaneously
simul_jobs = round(maxCores/cores_per_job)
os.system("tsp -S %s" % simul_jobs)

# number of parameter values
nParam_vals = np.size(swept_params_dict['param_vals1'])
    
#%% LAUNCH JOBS

# loop over swept parameter and launch jobs
for param_indx in range(0, nParam_vals):
    
    sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, param_indx) 
            
    command = "tsp python clusterRates_numActiveClusters_vs_JeePlus_noStim_gainBased.py --sweep_param_str_val %s --param_indx %d " % (sweep_param_str, param_indx)

    os.system(command) 