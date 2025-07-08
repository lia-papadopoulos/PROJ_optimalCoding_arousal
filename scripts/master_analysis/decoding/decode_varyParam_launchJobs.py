

#%% imports
import os
import numpy as np
import sys
import importlib

#%% from settings
import decode_settings as settings
func_path0 = settings.func_path0
func_path = settings.func_path
sim_params_path = settings.sim_params_path
sys.path.append(func_path)
sys.path.append(func_path0)
sys.path.append(sim_params_path)

from fcn_simulation_loading import fcn_set_sweepParam_string
from fcn_simulation_setup import fcn_define_arousalSweep

#%% get settings
load_from_simParams = settings.load_from_simParams
nNetworks = settings.nNetworks
sweep_param_name = settings.sweep_param_name
maxCores = settings.maxCores
cores_per_job = settings.cores_per_job
windL_vals = settings.windL_vals
ensembleSize_vals = settings.ensembleSize_vals
indNet_start = settings.indNet_start

if load_from_simParams == True:
    simParams_fname = settings.simParams_fname
else:
    n_sweepParams = settings.n_sweepParams
    swept_params_dict = settings.swept_params_dict
    

#%% load sim parameters
if load_from_simParams == True:
    # load sim parameters
    params = importlib.import_module(simParams_fname) 
    s_params = params.sim_params
    # arousal sweep
    s_params = fcn_define_arousalSweep(s_params)
    # unpack arousal
    n_sweepParams = s_params['nParams_sweep']
    swept_params_dict = s_params['swept_params_dict']

  
#%% LAUNCH JOBS

# tell task-spooler how many jobs it can run simultaneously
simul_jobs = round(maxCores/cores_per_job)
os.system("tsp -S %s" % simul_jobs)


# number of parameter values
nParam_vals = np.size(swept_params_dict['param_vals1'])

# loop over swept parameter, networks and launch jobs
for ind_ensembleSize in range(0,len(ensembleSize_vals)):
                
    ensembleSize = ensembleSize_vals[ind_ensembleSize]
                
    for indParam in range(0, nParam_vals):
            
        sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, indParam) 
            
        
        for ind_network in range(indNet_start,nNetworks,1):
            
            
            for ind_windL in range(0,len(windL_vals)):
                
                windL = windL_vals[ind_windL]
                    
                    
                # COMMAND TO RUN
                command = "tsp python decode_varyParam_master.py " \
                          "--sweep_param_name %s '--sweep_param_str_val' %s --ind_network %d --windL %f --ensembleSize %d " \
                          % (sweep_param_name, sweep_param_str, ind_network, windL, ensembleSize)
    
                # SUBMIT JOBS
                os.system(command) 

    
