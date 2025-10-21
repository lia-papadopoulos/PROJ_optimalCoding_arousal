
"""
RUN MFT FOR FIXED IN-DEGREE NETWORKS
POTENTIALLY CLUSTERED ARCHITECTURE
ARBITRARY NUMBER OF CLUSTERS
WITH QUENCHED VARIABILITY IN EXTERNAL INPUTS
USES PARAMETERS FROM A SET OF OF SIMULATIONS
"""


#%% BASIC IMPORTS

import sys
import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
import argparse


sys.path.append('/home/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/scripts/')    
sys.path.append('/home/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/scripts/functions/')  
sys.path.append('/home/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/scripts/MFT/funcs_MFT/clusteredNets_fullTheory/') 
 

from fcn_compute_firing_stats import Dict2Class
import master_MFT_fixedInDeg_EIclusters
import master_MFT_spatialVariance_EIclusters


#%% INFO FOR LOADING IN SIMULATION PARAMETERS

# simulation ID
simID = '102320221109' #[cluster, sd pert, 5 stim, poisson external inputs, Jeeplus = 15.75]

# network name
net_type = 'baseEIclu'
#net_type = 'baseHOM'

# stim shape
stim_shape = 'diff2exp'

# stim type
stim_type = ''

# relative stimulation amplitude
stim_rel_amp = 0.05


#%% ARGPARSER    

parser = argparse.ArgumentParser() 


# swept parameter name
parser.add_argument('-sweep_param_name', '--sweep_param_name', \
                    type=str, default = 'sd_nu_ext_e_pert')
    
# swept parameter name + value as string
parser.add_argument('-sweep_param_str_val', '--sweep_param_str_val', \
                    type=str, default = 'sd_nu_ext_e_pert0.000')
    
    
# possible swept parameters used in launch jobs
parser.add_argument('-sd_nu_ext_e_pert', '--sd_nu_ext_e_pert', \
                     type=float, default = 0.1)
        
    
# arguments of parser
args = parser.parse_args()


#-------------------- argparser values for later use -------------------------#
    

# name of swept parameter
sweep_param_name = args.sweep_param_name
# name of swept parameter with value as a string
sweep_param_str_val = args.sweep_param_str_val
# sd_nu_ext_e_pert
sd_nu_ext_e_pert = args.sd_nu_ext_e_pert



#%% SET PATHS

# path for loading data
load_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/test_stim_expSyn/')

# path for saving data
save_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/test_stim_expSyn/MFT_sweep_JeePlus_sweep_%s/' % sweep_param_name)


#%% MFT PARAMETERS

mft_params_dict = {}

mft_params_dict['solve_reduced'] = True
mft_params_dict['nSteps_MFT_DynEqs'] = 5000
mft_params_dict['dt_MFT_DynEqs'] = 1e-4
mft_params_dict['tau_e_MFT_DynEqs'] = 1e-3
mft_params_dict['tau_i_MFT_DynEqs'] = 1e-3
mft_params_dict['stopThresh_MFT_DynEqs'] = 1e-6
mft_params_dict['plot_MFT_DynEqs'] = False

mft_params_dict['stability_tau_e'] = 20e-3
mft_params_dict['stability_tau_i'] = 20e-3

mft_params_dict['low_w_lim'] = -15
mft_params_dict['high_w_lim'] = 15
mft_params_dict['w_vals'] = np.arange(-15, 15, 0.01)

mft_params_dict['min_JplusEE'] = 12
mft_params_dict['max_JplusEE'] = 25
mft_params_dict['delta_JplusEE'] = 0.075
mft_params_dict['nu_high_E'] = 80
mft_params_dict['nu_high_I'] = 60
mft_params_dict['nu_low_E'] = 1
mft_params_dict['nu_low_I'] = 1
mft_params_dict['popVar_nu_high_E'] = 10
mft_params_dict['popVar_nu_high_I'] = 1
mft_params_dict['popVar_nu_low_E'] = 5
mft_params_dict['popVar_nu_low_I'] = 1
mft_params_dict['n_active_clusters_sweep'] = np.arange(1,12,1)
    

    
#%% SET FILENAMES

filename = ( '%s_%s_sweep_%s%0.3f_network%d_IC%d_stim%d_stimType_%s_stim_rel_amp%0.3f_simulationData.mat' % \
           (simID, net_type, sweep_param_name, 0, 0, 0, 0, stim_shape, stim_rel_amp ) )


#%% LOAD ONE SIMULATION TO GET SIMULATION PARAMETERS

# load data
data = loadmat(load_path + filename, simplify_cells=True)   


#%% GET SIMULATION AND MFT PARAMETERS IN CLASS FORMAT
   
# sim_params         
s_params = Dict2Class(data['sim_params'])

# update with new value of sd_nu_ext_e_pert
s_params.sd_nu_ext_e_pert = sd_nu_ext_e_pert

# mft_params
mft_params = Dict2Class(mft_params_dict)


#%% FORWARDS AND BACKWARDS SWEEPS OVER JEEPLUS FOR DIFFERENT NUMBERS OF ACTIVE CLUSTERS



# perform the backwards sweep over JeePlus
if sd_nu_ext_e_pert == 0.0:
    
    JeePlus_backSweep_results_full = master_MFT_fixedInDeg_EIclusters.fcn_JeePlus_sweep_backwards(s_params, mft_params)

    # only save active, inactive, background rates    
    JeePlus_backSweep_results = {}
    JeePlus_backSweep_results['JplusEE_back'] = JeePlus_backSweep_results_full['JplusEE_back']
    JeePlus_backSweep_results['nu_e_backSweep'] = JeePlus_backSweep_results_full['nu_e_backSweep'][np.array([0,-2,-1])]
    JeePlus_backSweep_results['nu_i_backSweep'] = JeePlus_backSweep_results_full['nu_i_backSweep'][np.array([0,-2,-1])]


else:
    
    JeePlus_backSweep_results = master_MFT_spatialVariance_EIclusters.fcn_JeePlus_sweep_backwards(s_params, mft_params)



# perform the forwards sweep over JeePlus
if sd_nu_ext_e_pert == 0.0:
    
    JeePlus_forSweep_results_full = master_MFT_fixedInDeg_EIclusters.fcn_JeePlus_sweep_forwards(s_params, mft_params)
    
    # only save active, inactive, background rates    
    JeePlus_forSweep_results = {}
    JeePlus_forSweep_results['JplusEE_for'] = JeePlus_forSweep_results_full['JplusEE_for']    
    JeePlus_forSweep_results['nu_e_forSweep'] = JeePlus_forSweep_results_full['nu_e_forSweep'][np.array([0,-2,-1])]
    JeePlus_forSweep_results['nu_i_forSweep'] = JeePlus_forSweep_results_full['nu_i_forSweep'][np.array([0,-2,-1])]

    
else:      
    
    JeePlus_forSweep_results = master_MFT_spatialVariance_EIclusters.fcn_JeePlus_sweep_forwards(s_params, mft_params)
    
    
    
    
#%% SAVE THE DATA


results = {}
results['sim_params'] = s_params
results['mft_params'] = mft_params
results['JeePlus_backSweep_results'] = JeePlus_backSweep_results
results['JeePlus_forSweep_results'] = JeePlus_forSweep_results

save_name = ( '%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f_reducedMFT_sweepJeePlus.mat' % \
            ( simID, net_type, sweep_param_str_val, stim_shape, stim_rel_amp ) )
        
 
savemat(save_path + save_name, results)


        
    

