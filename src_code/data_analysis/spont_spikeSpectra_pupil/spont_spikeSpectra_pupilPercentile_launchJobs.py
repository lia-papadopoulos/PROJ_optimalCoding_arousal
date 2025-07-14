
# imports
import os
import spont_spikeSpectra_settings as settings

# from settings
maxCores = settings.maxCores
cores_per_job = settings.cores_per_job # needs to be set ahead of time using OMP_NUM_THREADS
all_sessions_to_run = settings.sessions_to_run

# cluster usage
simul_jobs = round(maxCores/cores_per_job)


#%% LAUNCH JOBS

# tell task-spooler how many jobs it can run simultaneously
os.system("tsp -S %s" % simul_jobs)

# loop over sessions and launch jobs
for session_name in all_sessions_to_run:
    
    # run analysis        
    command = "tsp python spont_spikeSpectra_pupilPercentile.py --session_name %s "  % (session_name)

    # SUBMIT JOBS
    os.system(command) 