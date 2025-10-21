
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
from scipy.io import savemat


sys.path.append('/home/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/scripts/functions/') 
from fcn_compute_firing_stats import Dict2Class
 
sys.path.append('/home/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/scripts/MFT/funcs_MFT/clusteredNets_fullTheory/') 
import master_MFT_spatialVariance_EIclusters

sys.path.append('/home/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/scripts/master_analysis/master_MFT/reduced_2cluster_nets/')
from simParams_2cluster_sweep_sd_nu_ext_e_smallBackground import sim_params


#%% INFO FOR LOADING IN SIMULATION PARAMETERS


# name of parameter swept over
sweep_param_name = "sd_nu_ext_e_pert"

# swept parameter values
param_vals = np.arange(0.01, 0.5, 0.01)

# end of filename
end_fname = 'smallBackground'


#%% SET PATHS


# path for saving data
save_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/MFT_2Ecluster/MFT_sweep_sd_nu_ext_e_pert_2Ecluster/')



#%% MFT PARAMETERS

mft_params_dict = {}

mft_params_dict['solve_reduced'] = False
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
mft_params_dict['nu_high_E'] = 60
mft_params_dict['nu_high_I'] = 14
mft_params_dict['nu_low_E'] = 1
mft_params_dict['nu_low_I'] = 14
mft_params_dict['popVar_nu_high_E'] = 10
mft_params_dict['popVar_nu_high_I'] = 10
mft_params_dict['popVar_nu_low_E'] = 80
mft_params_dict['popVar_nu_low_I'] = 10
mft_params_dict['n_active_clusters_sweep'] = np.array([1])



#%% GET SIMULATION AND MFT PARAMETERS IN CLASS FORMAT
   
# sim_params         
s_params =  sim_params()
s_params.update_JplusAB()
s_params.set_dependent_vars()

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

save_name = ( 'MFT_sweep_%s_JplusEE%0.3f_2Ecluster_%s.mat' % ( sweep_param_name, s_params.JplusEE, end_fname ) )
        
 
savemat(save_path + save_name, results)


        
    

