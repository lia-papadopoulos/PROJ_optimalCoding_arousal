

#%% imports
import os
import psth_allTrials_settings as settings

#%% settings
maxCores = settings.maxCores
cores_per_job = settings.cores_per_job # needs to be set ahead of time using OMP_NUM_THREADS
sessions_to_run = settings.sessions_to_run

#%% cluster usage
simul_jobs = round(maxCores/cores_per_job)

#%% LAUNCH JOBS

# tell task-spooler how many jobs it can run simultaneously
os.system("tsp -S %s" % simul_jobs)

# loop over sessions and launch jobs
for session_name in sessions_to_run:
    
    command = "tsp python psth_allTrials.py --session_name %s "  % (session_name)
    os.system(command) 
    
