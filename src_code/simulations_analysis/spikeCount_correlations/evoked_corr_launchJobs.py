
#%% imports
import os
import evoked_corr_settings as settings

#%% settings
maxCores = settings.maxCores
cores_per_job = settings.cores_per_job
nNetworks = settings.nNetworks

#%% LAUNCH JOBS

# tell task-spooler how many jobs it can run simultaneously
simul_jobs = round(maxCores/cores_per_job)
os.system("tsp -S %s" % simul_jobs)

# loop over networks and launch jobs
for net_indx in range(0, nNetworks):
            
    command = " tsp python evoked_corr.py --net_indx %d " % (net_indx)

    os.system(command) 

