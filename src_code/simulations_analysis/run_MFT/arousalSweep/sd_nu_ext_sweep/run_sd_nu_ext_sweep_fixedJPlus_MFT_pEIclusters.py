
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


sys.path.append('/home/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/scripts/')    
sys.path.append('/home/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/scripts/functions/')  
sys.path.append('/home/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/scripts/MFT/funcs_MFT/clusteredNets_fullTheory/') 
 

from fcn_compute_firing_stats import Dict2Class
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

# set JeePlus values
JplusEE = 16.725

# name of parameter swept over
sweep_param_name = "sd_nu_ext_e_pert"

# swept parameter values
param_vals = np.arange(0.001, 0.4, 0.001)

fname_end = ''


#%% SET PATHS

# path for loading data
load_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/test_stim_expSyn/')

# path for saving data
save_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/test_stim_expSyn/MFT_sweep_sd_nu_ext_e_pert/')


#%% MFT PARAMETERS

mft_params_dict = {}

mft_params_dict['solve_reduced'] = True
mft_params_dict['nSteps_MFT_DynEqs'] = 4000
mft_params_dict['dt_MFT_DynEqs'] = 1e-4
mft_params_dict['tau_e_MFT_DynEqs'] = 1e-3
mft_params_dict['tau_i_MFT_DynEqs'] = 1e-3
mft_params_dict['stopThresh_MFT_DynEqs'] = 1e-6
mft_params_dict['plot_MFT_DynEqs'] = False

mft_params_dict['stability_tau_e'] = 20e-3
mft_params_dict['stability_tau_i'] = 20e-3

mft_params_dict['low_w_lim'] = -5.5
mft_params_dict['high_w_lim'] = 5.5
mft_params_dict['w_vals'] = np.arange(-5.5, 5.5, 0.01)
mft_params_dict['name_sweepParam'] = sweep_param_name
mft_params_dict['min_sweepParam'] = param_vals[0]
mft_params_dict['max_sweepParam'] = param_vals[-1]
mft_params_dict['delta_sweepParam'] = param_vals[1]-param_vals[0]
mft_params_dict['nu_high_E'] = 80
mft_params_dict['nu_high_I'] = 60
mft_params_dict['nu_low_E'] = 5
mft_params_dict['nu_low_I'] = 10
mft_params_dict['popVar_nu_high_E'] = 10
mft_params_dict['popVar_nu_high_I'] = 1
mft_params_dict['popVar_nu_low_E'] = 100
mft_params_dict['popVar_nu_low_I'] = 25
mft_params_dict['n_active_clusters_sweep'] = np.arange(3,4,1)
    

    
#%% SET FILENAMES

filename = ( '%s_%s_sweep_%s%0.3f_network%d_IC%d_stim%d_stimType_%s_stim_rel_amp%0.3f_simulationData.mat' % \
           (simID, net_type, sweep_param_name, 0, 0, 0, 0, stim_shape, stim_rel_amp ) )


#%% LOAD ONE SIMULATION TO GET SIMULATION PARAMETERS

# load data
data = loadmat(load_path + filename, simplify_cells=True)   


#%% GET SIMULATION AND MFT PARAMETERS IN CLASS FORMAT
   
# sim_params         
s_params = Dict2Class(data['sim_params'])

# update with new value of JplusEE
s_params.JplusEE = JplusEE

if fname_end == 'noInterClusterDepression':
    s_params.depress_interCluster = False
else:
    s_params.depress_interCluster = True
    

# mft_params
mft_params = Dict2Class(mft_params_dict)


#%% FORWARDS AND BACKWARDS SWEEPS OVER SD_NU_EXT_E_PERT FOR DIFFERENT NUMBERS OF ACTIVE CLUSTERS

    
# backwards sweep
print('backwards')
backSweep_results = master_MFT_spatialVariance_EIclusters.fcn_param_sweep_high_to_low_rate(s_params, mft_params)
     
# forwards sweep
print('forwards')    
forSweep_results = master_MFT_spatialVariance_EIclusters.fcn_param_sweep_low_to_high_rate(s_params, mft_params)
        
    
#%% SAVE THE DATA


results = {}
results['sim_params'] = s_params
results['mft_params'] = mft_params
results['backSweep_results'] = backSweep_results
results['forSweep_results'] = forSweep_results

save_name = ( '%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f_reducedMFT_sweep_%s_JplusEE%0.3f_%s.mat' % \
            ( simID, net_type, sweep_param_name, stim_shape, stim_rel_amp, sweep_param_name, JplusEE, fname_end ) )
        
 
savemat(save_path + save_name, results)


        
    

