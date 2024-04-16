# LAUNCH JOBS FOR PARAMETER SWEEP

import os
import numpy as np
import sys
from datetime import datetime

#-------------------- cluster usage -------------------------------------------#

maxCores = 48
cores_per_job = 4 # needs to be set ahead of time using OMP_NUM_THREADS
simul_jobs = round(maxCores/cores_per_job)

# start time
now = datetime.now()

#date_time = now.strftime("%m%d%Y%H%M%S")
date_time = '031120241042'
#date_time = '021120241253'
#date_time = '122020231145' #cluster, sd pert (same each clusters), 1 stim,  poisson, external inputs, Jplus=15.75
#date_time = '113020232105' #cluster, sd pert (same each clusters), 5 stim,  poisson, external inputs, Jplus=15.75
#date_time = '072520231624' #hom, mean pert, 5 stim,  poisson, external inputs
#date_time = '111520221344' #cluster, mean pert, 5 stim, poisson external inputs, Jeeplus = 15.75
#date_time = '110920220959' #hom, sd pert, 5 stim,  poisson, external inputs
#date_time = '102320221109' #cluster, sd pert, 5 stim,  poisson, external inputs, Jplus=15.75

#date_time = '04052023_f_Ecells_target_1.0'

#-------------------- path for saving data ------------------------------------#

path_data = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/'\
             'PROJ_VariabilityGainMod/data_files/test_stim_expSyn/')
    
#-------------------- network and initial conditions --------------------------#
   
# type of network (cluster or hom)
net_type = 'cluster' 
# number of different networks per swept value
indNet_start = 0
n_networks = 10
# number of initial conditions per network
n_ICs = 30

#-------------------- swept parameter -----------------------------------------#

# name of parameter to sweep over
# use NONE if you only want to sweep over different initial conditions/networks/stimulation

sweep_param_name = "sd_nu_ext_e_pert"
#sweep_param_name = "sd_nu_ext_e_white_pert"
#sweep_param_name = "mean_nu_ext_e_offset"
#sweep_param_name = "JplusEE"

sweep_param_name2 = ''
#sweep_param_name2 = "mean_nu_ext_i_offset"

param_vals = np.arange(0.0, 0.45, 0.05)
#param_vals =  np.arange(0.0, 0.45, 0.05)
#param_vals = np.arange(1.05, 1.3, 0.15)

param_vals2 = np.array([])
#param_vals2 = np.arange(1.05, 1.3, 0.15)


#-------------------- other quantities -----------------------------------------#
sd_nu_ext_e_type = 'same_eachCluster'
sd_nu_ext_i_type = 'same_eachCluster'


#-------------------- filename setting ----------------------------------------#
if sweep_param_name == "NONE":
    param_vals = [0]
    if net_type == 'cluster':
        fname = (date_time + "_baseEIclu_sweep" + sweep_param_name)
    elif net_type == 'hom':
        fname = (date_time + "_baseHOM_sweep" + sweep_param_name)
    else:
        sys.exit('INCOMPATIBLE NETWORK TYPE SPECIFIED')

else:
    if net_type == 'cluster': 
        if ( (len(sweep_param_name) != 0) & (len(sweep_param_name2) != 0) ):
            fname = (date_time + "_baseEIclu_sweep_" + sweep_param_name + '_' + sweep_param_name2) 
        else:
            if sweep_param_name == 'sd_nu_ext_e_pert':
                fname = (date_time + "_baseEIclu_sweep_" + sd_nu_ext_e_type + sweep_param_name)
            else:
                fname = (date_time + "_baseEIclu_sweep_" + sweep_param_name) 
    elif net_type == 'hom':
        if sweep_param_name == 'sd_nu_ext_e_pert':
            fname = (date_time + "_baseHOM_sweep_" + sd_nu_ext_e_type + sweep_param_name)
        else:
            fname = (date_time + "_baseHOM_sweep_" + sweep_param_name)
    else:
        sys.exit('INCOMPATIBLE NETWORK TYPE SPECIFIED')
    
        
    
# function that runs jobs
def masterSim_launchJobs():
      
    ###------------------------------------------------------------------------
    ### TASK SPOOLER PARAMETERS
    ###------------------------------------------------------------------------
    # set core usage
    
    # tell task-spooler how many jobs it can run simultaneously
    os.system("tsp -S %s" % simul_jobs)
    
    ###------------------------------------------------------------------------
    ### LOOP OVER SWEPT PARAMETERS/JOB SUBMISSION
    ###------------------------------------------------------------------------
    for indParam in np.arange(0, len(param_vals)):
        
        param = param_vals[indParam]
        
        if len(sweep_param_name2) != 0:
            param2 = param_vals2[indParam]
                
        for ind_network in range(indNet_start, indNet_start + n_networks):
            
            for ind_IC in range(0,n_ICs):
                        
                # ID FOR SAVING DATA
                if sweep_param_name == "NONE":
                    
                    ID = fname + '_network' + str(ind_network) + '_IC' + str(ind_IC)
                    
                    # COMMAND TO RUN
                    command = "tsp python masterSim.py " \
                              "--ID %s --path_data %s --ind_network %d --ind_IC %d " \
                              % (ID, path_data, ind_network, ind_IC)
                else:
                    
                    if ( (len(sweep_param_name) != 0) & (len(sweep_param_name2) != 0) ):
                        
                        param_str = ( ('%0.3f_%0.3f') % (param, param2))
                        
                        ID = ( fname + param_str + '_network' + str(ind_network) + '_IC' + str(ind_IC) )
                        
                        # COMMAND TO RUN
                        command = "tsp python masterSim.py " \
                              "--%s %f --%s %f --sd_nu_ext_e_type %s --ID %s --path_data %s --ind_network %d --ind_IC %d " \
                                  % (sweep_param_name, param, sweep_param_name2, param2, sd_nu_ext_e_type, ID, path_data, ind_network, ind_IC)
                        
                    else:
                        
                        param_str = ('%0.3f' % param)
                    
                        ID = ( fname + param_str + '_network' + str(ind_network) + '_IC' + str(ind_IC) )
    
                        # COMMAND TO RUN
                        command = "tsp python masterSim.py " \
                                  "--%s %f --sd_nu_ext_e_type %s --ID %s --path_data %s --ind_network %d --ind_IC %d " \
                                  % (sweep_param_name, param, sd_nu_ext_e_type, ID, path_data, ind_network, ind_IC)
    
                # SUBMIT JOBS
                os.system(command) 

    
# CALL JOB LAUNCHER
masterSim_launchJobs()
