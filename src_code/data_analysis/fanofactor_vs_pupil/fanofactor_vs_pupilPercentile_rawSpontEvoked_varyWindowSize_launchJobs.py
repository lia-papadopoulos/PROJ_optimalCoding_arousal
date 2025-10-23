
#%% imports
import os
import fano_factor_settings as settings

#%% settings
maxCores = settings.maxCores
cores_per_job = settings.cores_per_job # needs to be set ahead of time using OMP_NUM_THREADS
all_sessions_to_run = settings.all_sessions_to_run

#%% cluster usage
simul_jobs = round(maxCores/cores_per_job)
      
#%% LAUNCH JOBS

# tell task-spooler how many jobs it can run simultaneously
os.system("tsp -S %s" % simul_jobs)

# loop over sessions and launch jobs
for session_name in all_sessions_to_run:
    
    command = "tsp python fanofactor_vs_pupilPercentile_rawSpontEvoked_varyWindowSize.py --session_name %s "  % (session_name)
    os.system(command) 