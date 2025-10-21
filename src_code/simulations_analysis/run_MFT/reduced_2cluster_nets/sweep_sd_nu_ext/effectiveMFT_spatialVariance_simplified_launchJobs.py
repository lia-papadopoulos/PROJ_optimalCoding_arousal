# LAUNCH JOBS FOR PARAMETER SWEEP

#-------------------- basic imports -------------------------------------------#

import os
import effectiveMFT_spatialVariance_simplified_settings as settings

#-------------------- unpack settings -------------------------------------------#
maxCores = settings.maxCores
cores_per_job = settings.cores_per_job

param_vals = settings.param_vals
sweep_param_name = settings.sweep_param_name

#-------------------- cluster usage -------------------------------------------#

simul_jobs = round(maxCores/cores_per_job)

#%% LAUNCH JOBS

# tell task-spooler how many jobs it can run simultaneously
os.system("tsp -S %s" % simul_jobs)

# loop over swept parameter, networks and launch jobs
for param_indx, param in enumerate(param_vals):
            
    # swept param label
    sweep_param_str_val = sweep_param_name + '%0.3f' % param
    
    # COMMAND TO RUN
    command = " tsp python run_effectiveMFT_spatialVariance_simplified.py  --%s %f --sweep_param_str_val %s " % \
                (sweep_param_name, param, sweep_param_str_val)

    # SUBMIT JOBS
    os.system(command) 

