
#%% imports
import os
import numpy as np
import sys
import importlib

#%% settings
import settings as params

#%% functions
func_path = params.func_path
func_path1 = params.func_path1
sys.path.append(func_path)
sys.path.append(func_path1)
from fcn_simulation_loading import fcn_set_sweepParam_string
from fcn_simulation_setup import fcn_define_arousalSweep

#%% get settings
sim_params_path = params.sim_params_path
simParams_fname = params.simParams_fname
sweep_param_name = params.sweep_param_name
maxCores = params.maxCores
cores_per_job = params.cores_per_job

#%% load sim parameters
sys.path.append(sim_params_path)
sim_params = importlib.import_module(simParams_fname) 
s_params = sim_params.sim_params

#%% arousal sweep
s_params = fcn_define_arousalSweep(s_params)

#%% unpack simulation parameters
n_sweepParams = s_params['nParams_sweep']
swept_params_dict = s_params['swept_params_dict']

#%% LAUNCH JOBS

# tell task-spooler how many jobs it can run simultaneously
simul_jobs = round(maxCores/cores_per_job)
os.system("tsp -S %s" % simul_jobs)

# number of parameter values
nParam_vals = np.size(swept_params_dict['param_vals1'])

# loop over arousal and launch jobs
for param_indx in range(0, nParam_vals):
            
    sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, param_indx)
  
    command = "tsp python numActive_targeted_nontargeted_clusters_vs_perturbation_gainBased.py --sweep_param_str_val %s " % (sweep_param_str)

    os.system(command) 
