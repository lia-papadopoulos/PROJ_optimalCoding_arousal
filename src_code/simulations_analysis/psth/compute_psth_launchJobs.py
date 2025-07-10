

#-------------------- basic imports -------------------------------------------#

import os
import psth_settings as settings

#-------------------- unpack settings -------------------------------------------#
maxCores = settings.maxCores
cores_per_job = settings.cores_per_job
nNetworks = settings.nNetworks
nStim = settings.nStim

#%% LAUNCH JOBS

# tell task-spooler how many jobs it can run simultaneously
simul_jobs = round(maxCores/cores_per_job)
os.system("tsp -S %s" % simul_jobs)

# loop over swept parameter, networks and launch jobs
for indNet in range(0,nNetworks):
    
    for indStim in range(0, nStim):
    
        # COMMAND TO RUN
        command = ( ("tsp python compute_psth.py --ind_network %d --ind_stim %d") % (indNet, indStim))

        # SUBMIT JOBS
        os.system(command) 
        
