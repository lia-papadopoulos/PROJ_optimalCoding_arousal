# LAUNCH JOBS FOR PARAMETER SWEEP

#-------------------- basic imports -------------------------------------------#

import os
import numpy as np

#-------------------- cluster usage -------------------------------------------#

maxCores = 48
cores_per_job = 4 # needs to be set ahead of time using OMP_NUM_THREADS
simul_jobs = round(maxCores/cores_per_job)

#-------------------- parameters ----------------------------------------------#

# name of parameter swept over
sweep_param_name = "sd_nu_ext_e_pert"

# swept parameter values
param_vals = np.arange(0.005, 0.45, 0.01)


#%% LAUNCH JOBS

# tell task-spooler how many jobs it can run simultaneously
os.system("tsp -S %s" % simul_jobs)

# loop over swept parameter, networks and launch jobs
for param in param_vals:
    
    # swept param label
    sweep_param_str_val = sweep_param_name + '%0.3f' % param
    
        
    # COMMAND TO RUN
    command = "tsp python run_JeePlus_sweep_MFT_sd_nu_ext_pEIclusters.py " \
              "--sweep_param_name %s --%s %f '--sweep_param_str_val' %s " \
              % (sweep_param_name, sweep_param_name, param, sweep_param_str_val)

    # SUBMIT JOBS
    os.system(command) 