
"""
RUN MFT FOR FIXED IN-DEGREE NETWORKS
POTENTIALLY CLUSTERED ARCHITECTURE
ARBITRARY NUMBER OF CLUSTERS
USES PARAMETERS FROM A SET OF OF SIMULATIONS
"""


#%% BASIC IMPORTS

import sys
import numpy as np
from scipy.io import loadmat
from scipy.io import savemat

sys.path.append('../../../')
import global_settings


sys.path.append(global_settings.path_to_src_code)    
sys.path.append(global_settings.path_to_src_code + 'functions/')  
sys.path.append(global_settings.path_to_src_code + '/MFT/funcs_MFT/clusteredNets_fullTheory/') 
 

from fcn_compute_firing_stats import Dict2Class
import master_MFT_fixedInDeg_EIclusters


#%% INFO FOR LOADING IN SIMULATION PARAMETERS

# simulation ID
simID = 'test0000'

# network name
net_type = 'baseEIclu'

# stim shape
stim_shape = 'diff2exp'

# relative stimulation amplitude
stim_rel_amp = 0.05

# set JeePlus values
JplusEE = 16.725

# sweep param name
sweep_param_name = 'pert_mean_nu_ext_ee'

# number of swept parameters
n_sweepParams = 3




#%% SET PATHS

# path for loading data
load_path = global_settings.path_to_sim_output + 'simParams_mft/'

# path for saving data
save_path = global_settings.path_to_sim_output + 'MFT_arousalSweep/'


#%% MFT PARAMETERS

mft_params_dict = {}


mft_params_dict['nu_high_E'] = 80
mft_params_dict['nu_high_I'] = 60
mft_params_dict['nu_low_E'] = 1
mft_params_dict['nu_low_I'] = 1
mft_params_dict['n_active_clusters_sweep'] = np.arange(1,12,1)
mft_params_dict['solve_reduced'] = True
mft_params_dict['nSteps_MFT_DynEqs'] = 2000
mft_params_dict['dt_MFT_DynEqs'] = 1e-4
mft_params_dict['nu_vec'] = np.nan
mft_params_dict['tau_e_MFT_DynEqs'] = 1e-3
mft_params_dict['tau_i_MFT_DynEqs'] = 1e-3
mft_params_dict['stopThresh_MFT_DynEqs'] = 1e-6
mft_params_dict['plot_MFT_DynEqs'] = False
mft_params_dict['stability_tau_e'] = 20e-3
mft_params_dict['stability_tau_i'] = 20e-3 

    
#%% LOAD IN MFT SIMULATION PARAMETERS

filename = ('%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f_simParams_mft_noDisorder.mat' % \
            ( simID, net_type, sweep_param_name, stim_shape, stim_rel_amp) )

simParams_mft = loadmat(load_path + filename, simplify_cells=True)   

#%% GET SIMULATION AND MFT PARAMETERS IN CLASS FORMAT
   
# sim_params         
s_params = Dict2Class(simParams_mft)

# update with new value of JplusEE
s_params.JplusEE = JplusEE

# mft_params
mft_params = Dict2Class(mft_params_dict)


#%% FORWARDS AND BACKWARDS SWEEPS OVER AROUSAL PARAMETERS FOR DIFFERENT NUMBERS OF ACTIVE CLUSTERS

# backwards sweep
print('backwards')
backSweep_results = master_MFT_fixedInDeg_EIclusters.fcn_paramSweep_high_to_low_rate(s_params, mft_params)
     
#%%
# forwards sweep
print('forwards')    
forSweep_results = master_MFT_fixedInDeg_EIclusters.fcn_paramSweep_low_to_high_rate(s_params, mft_params)
        
    
#%% SAVE THE DATA


results = {}
results['sim_params'] = s_params
results['mft_params'] = mft_params
results['backSweep_results'] = backSweep_results
results['forSweep_results'] = forSweep_results

save_name = ( '%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f_JplusEE%0.3f_MFT_noDisorder.mat' % \
            ( simID, net_type, sweep_param_name, stim_shape, stim_rel_amp, JplusEE ) )
        
 
savemat(save_path + save_name, results)


        
    

