
#%% imports
import os
import numpy as np
import sys
import importlib

#%% from settings
import settings as settings

#%% functions
func_path0 = settings.func_path0
func_path = settings.func_path
sim_params_path = settings.sim_params_path
sys.path.append(func_path)
sys.path.append(func_path0)
sys.path.append(sim_params_path)
from fcn_simulation_loading import fcn_set_sweepParam_string
from fcn_simulation_setup import fcn_define_arousalSweep

#%% get settings
simParams_fname = settings.simParams_fname
sweep_param_name = settings.sweep_param_name
maxCores = settings.maxCores
cores_per_job = settings.cores_per_job

#%% load sim parameters
params = importlib.import_module(simParams_fname) 
s_params = params.sim_params

#%% arousal sweep
s_params = fcn_define_arousalSweep(s_params)

#%% unpack simulation parameters
simID = s_params['simID']
n_sweepParams = s_params['nParams_sweep']
swept_params_dict = s_params['swept_params_dict']

#%% LAUNCH JOBS

# tell task-spooler how many jobs it can run simultaneously
simul_jobs = round(maxCores/cores_per_job)
os.system("tsp -S %s" % simul_jobs)

# number of parameter values
nParam_vals = np.size(swept_params_dict['param_vals1'])

# loop over arousal and launch jobs
for indParam in range(0, nParam_vals):
                
    sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, indParam) 
        
    command = " tsp python spont_cvISI.py --sweep_param_str_val %s " % (sweep_param_str)

    os.system(command) 

