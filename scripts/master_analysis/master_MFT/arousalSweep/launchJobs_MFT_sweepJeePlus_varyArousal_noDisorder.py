# LAUNCH JOBS FOR PARAMETER SWEEP

#-------------------- basic imports -------------------------------------------#

import os
from scipy.io import loadmat

#-------------------- cluster usage -------------------------------------------#

maxCores = 48
cores_per_job = 4 # needs to be set ahead of time using OMP_NUM_THREADS
simul_jobs = round(maxCores/cores_per_job)

#-------------------- parameters ----------------------------------------------#

# simulation ID
simID = '051300002025_clu' 

# network name
net_type = 'baseEIclu'

# stim shape
stim_shape = 'diff2exp'

# relative stimulation amplitude
stim_rel_amp = 0.05

# name of parameter swept over
sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'

# path for loading data
simParams_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/test_stim_expSyn/simParams_mft/')


# load in mft parameter values
filename = ('%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f_simParams_mft_noDisorder.mat' % \
            ( simID, net_type, sweep_param_name, stim_shape, stim_rel_amp) )

simParams_mft = loadmat(simParams_path + filename, simplify_cells=True)   

n_paramVals_mft = simParams_mft['n_paramVals_mft']


#%% LAUNCH JOBS

# tell task-spooler how many jobs it can run simultaneously
os.system("tsp -S %s" % simul_jobs)

# loop over swept parameter, networks and launch jobs
for param in range(0, n_paramVals_mft):
    
    # swept param label
    sweep_param_str_val = sweep_param_name + '%0.3f' % param
    
        
    # COMMAND TO RUN
    command = "tsp python run_MFT_sweepJeePlus_varyArousal_noDisorder.py " \
              "--simID %s --net_type %s --stim_shape %s --stim_rel_amp %s --simParams_path %s --sweep_param_name %s --indParamSweep %d" \
              % (simID, net_type, stim_shape, stim_rel_amp, simParams_path, sweep_param_name, param)

    # SUBMIT JOBS
    os.system(command) 