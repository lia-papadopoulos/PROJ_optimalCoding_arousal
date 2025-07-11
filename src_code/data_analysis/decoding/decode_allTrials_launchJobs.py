

#-------------------- basic imports -------------------------------------------#
import os

import decoding_params as decode_params

#-------------------- cluster usage -------------------------------------------#

maxCores = decode_params.maxCores
cores_per_job = decode_params.cores_per_job
sessions_to_run = decode_params.sessions_to_run

simul_jobs = round(maxCores/cores_per_job)


  
#%% LAUNCH JOBS

# tell task-spooler how many jobs it can run simultaneously
os.system("tsp -S %s" % simul_jobs)

# loop over sessions and launch jobs
for session_name in sessions_to_run:
    
    # run decoding        
    command = "tsp python decode_allTrials.py --session_name %s "  % (session_name)

    # SUBMIT JOBS
    os.system(command) 
    
    
    