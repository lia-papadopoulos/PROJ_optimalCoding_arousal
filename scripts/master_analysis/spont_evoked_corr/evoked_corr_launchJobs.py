# LAUNCH JOBS FOR PARAMETER SWEEP

#-------------------- basic imports -------------------------------------------#

import os
import evoked_corr_vs_perturbation_settings as settings

#-------------------- unpack settings -------------------------------------------#
maxCores = settings.maxCores
cores_per_job = settings.cores_per_job
nNetworks = settings.nNetworks


#%% LAUNCH JOBS

# tell task-spooler how many jobs it can run simultaneously
simul_jobs = round(maxCores/cores_per_job)
os.system("tsp -S %s" % simul_jobs)

# loop over swept parameter, networks and launch jobs
for net_indx in range(0, nNetworks):
            
    
    # COMMAND TO RUN
    command = " tsp python evoked_corr.py --net_indx %d " % (net_indx)

    # SUBMIT JOBS
    os.system(command) 

