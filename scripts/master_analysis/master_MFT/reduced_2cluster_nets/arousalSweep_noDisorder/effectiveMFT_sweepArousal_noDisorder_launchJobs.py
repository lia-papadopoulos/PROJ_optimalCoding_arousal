# LAUNCH JOBS FOR PARAMETER SWEEP

#%% basic imports
import os
import effectiveMFT_sweepArousal_noDisorder_settings as settings

#%% unpack settings
nArousal_samples = settings.nArousal_samples
maxCores = settings.maxCores
cores_per_job = settings.cores_per_job

#%% cluster usage
simul_jobs = round(maxCores/cores_per_job)

#%% LAUNCH JOBS

# tell task-spooler how many jobs it can run simultaneously
os.system("tsp -S %s" % simul_jobs)

# loop over swept parameter, networks and launch jobs
for param_indx in range(0, nArousal_samples+1):
                
    # COMMAND TO RUN
    command = " tsp python run_effectiveMFT_sweepArousal_noDisorder_ALT2.py --param_indx %d " % (param_indx)

    # SUBMIT JOBS
    os.system(command) 

