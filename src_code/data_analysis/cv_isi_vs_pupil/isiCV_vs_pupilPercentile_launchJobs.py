
#%% imports
import os
import isiCV_vs_pupilPercentile_settings as settings

#%% unpack settings
maxCores = settings.maxCores
cores_per_job = settings.cores_per_job
all_sessions_to_run = settings.all_sessions_to_run


#%% LAUNCH JOBS

# tell task-spooler how many jobs it can run simultaneously
simul_jobs = round(maxCores/cores_per_job)
os.system("tsp -S %s" % simul_jobs)

# loop over sessions and launch jobs
for session_name in all_sessions_to_run:
    
    # run      
    command = "tsp python isiCV_vs_pupilPercentile.py --session_name %s "  % (session_name)
    os.system(command) 