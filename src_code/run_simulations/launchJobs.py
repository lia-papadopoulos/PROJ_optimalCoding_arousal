# LAUNCH JOBS FOR PARAMETER SWEEP

import os
import numpy as np
import sys
import importlib

import paths_file
from fcn_simulation_setup import fcn_define_arousalSweep

# unpack paths
sim_params_path = paths_file.sim_params_path
sim_params_name = paths_file.sim_params_name
functions_path1 = paths_file.functions_path1
path_data = paths_file.save_path

sys.path.append(sim_params_path) 
params = importlib.import_module(sim_params_name) 
s_params = params.sim_params
del params

sys.path.append(functions_path1)       
from fcn_simulation_loading import fcn_set_sweepParam_string

#-------------------- parameters -------------------------------------------#


# get arousal parameters for sweeping over
s_params = fcn_define_arousalSweep(s_params)

# unpack parameters
simID = s_params['simID']
net_type = s_params['net_type']
nParams_sweep = s_params['nParams_sweep']
swept_param_name_dict = s_params['swept_param_name_dict']
swept_params_dict = s_params['swept_params_dict']

maxCores = s_params['maxCores']
cores_per_job = s_params['cores_per_job']
indNet_start = s_params['indNet_start']
n_networks = s_params['n_networks']
n_ICs = s_params['n_ICs']


#-------------------- swept parameter -----------------------------------------#


# name of parameter to sweep over
sweep_param_name = swept_param_name_dict['param_vals1']
for i in range(2, nParams_sweep+1):
    key_name = 'param_vals%d' % i
    sweep_param_name = sweep_param_name + '_' + swept_param_name_dict[key_name]



if net_type == 'cluster': 
    
    net_name = 'baseEIclu'
    
elif net_type == 'hom':
    
    net_name = 'baseHOM'

else:
    sys.exit('INCOMPATIBLE NETWORK TYPE SPECIFIED')
        

simul_jobs = round(maxCores/cores_per_job)

    
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
    # number of parameter values
    nParam_vals = np.size(swept_params_dict['param_vals1'])

    # loop over swept parameter, networks and launch jobs
    for param_indx in range(0, nParam_vals):
        
        sweep_param_str = fcn_set_sweepParam_string(nParams_sweep, sweep_param_name, swept_params_dict, param_indx) 
    
        for ind_network in range(indNet_start, indNet_start + n_networks):
            
            for ind_IC in range(0,n_ICs):
                
                ID = ('%s_%s_sweep_%s_network%d_IC%d' % (simID, net_name, sweep_param_str, ind_network, ind_IC))

                str_pass = ' --%s %f '
                tuple_pass = (swept_param_name_dict['param_vals1'], swept_params_dict['param_vals1'][param_indx])

                for i in range(2, nParams_sweep+1):
                    
                    key_name = 'param_vals%d' % i
                    tuple_pass = tuple_pass + (swept_param_name_dict[key_name], swept_params_dict[key_name][param_indx])
                    str_pass = str_pass + '--%s %f '


                tuple_pass = tuple_pass + (ID, path_data, ind_network, ind_IC)
                str_pass = str_pass + '--ID %s --path_data %s --ind_network %d --ind_IC %d '
                
                command = 'tsp python masterSim.py' + ( str_pass % tuple_pass )
                
                # SUBMIT JOBS
                os.system(command) 

    
# CALL JOB LAUNCHER
masterSim_launchJobs()
