

#%% basic imports
import sys
import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
import importlib
from fcn_update_params_forMFT import fcn_update_params_forMFT

#%% import mft parameters
import params_JeePlus_sweep_MFT_baseline_pEIclusters as params
sim_params_path = params.sim_params_path
func_path0 = params.func_path0
func_path1 = params.func_path1
func_path2 = params.func_path2
simParams_fname = params.simParams_fname
net_type = params.net_type
sweep_param_name = params.sweep_param_name
indParam = params.indParam
load_path = params.load_path
save_path = params.save_path
mft_params_dict = params.mft_params_dict


#%% load custom functions
sys.path.append(func_path0)    
sys.path.append(func_path1)  
sys.path.append(func_path2) 
 
from fcn_compute_firing_stats import Dict2Class
import master_MFT_fixedInDeg_EIclusters    
from fcn_simulation_loading import fcn_set_sweepParam_string
from fcn_simulation_setup import fcn_define_arousalSweep


#%% load simulation parameters
sys.path.append(sim_params_path)
params = importlib.import_module(simParams_fname) 
default_params = params.sim_params
del params

#%% unpack parmeters
simID = default_params['simID']
stim_shape = default_params['stim_shape']
stim_rel_amp = default_params['stim_rel_amp']
n_sweepParams = default_params['nParams_sweep']

#%% define parameter sweep
default_params = fcn_define_arousalSweep(default_params)
swept_params_dict = default_params['swept_params_dict']

#%% remove default sim parameters
del default_params

#%% get filename for simulations given arousal level at which we want to run mft

sweep_param_str_val = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, indParam)

fname =  ( ('%s_%s_sweep_%s_network%d_IC%d_stim%d_stimType_%s_stim_rel_amp%0.3f') % \
           ( simID, net_type, sweep_param_str_val, 0, 0, 0, stim_shape, stim_rel_amp) )


#%% load one simulation to get simulation parameters
data = loadmat(load_path + fname + '_simulationData.mat', simplify_cells=True)   


#%% get simulation and mft parameters in class format
   
# sim_params         
s_params = Dict2Class(data['sim_params'])

# update parameters for use in MFT functions
s_params = fcn_update_params_forMFT(s_params)
print(s_params.nu_ext_ee)

# mft_params
mft_params = Dict2Class(mft_params_dict)


#%% run the mft

# perform the backwards sweep over JeePlus
JeePlus_backSweep_results = master_MFT_fixedInDeg_EIclusters.fcn_JeePlus_sweep_backwards(s_params, mft_params)
        
# perform the forwards sweep over JeePlus
JeePlus_forSweep_results = master_MFT_fixedInDeg_EIclusters.fcn_JeePlus_sweep_forwards(s_params, mft_params)
    

#%% determine the critical value of Jee+ (backward solution = forward solution)

JeePlus_back = JeePlus_backSweep_results['JplusEE_back'].copy()
JeePlus_for = JeePlus_forSweep_results['JplusEE_for'].copy()

len_nActive = np.size(mft_params_dict['n_active_clusters_sweep'])
JeePlus_critical = np.ones(len_nActive)*np.nan

for ind_nActive in range(0, len_nActive):
    
    nu_e_backSweep = JeePlus_backSweep_results['nu_e_backSweep'][0,:,ind_nActive].copy()
    nu_e_forSweep = JeePlus_forSweep_results['nu_e_forSweep'][0,:,ind_nActive].copy()

    for ind_JeePlus_back in range(0, len(JeePlus_back)):
        
        indFor_solutionsMatch = np.nonzero( np.abs(nu_e_backSweep[ind_JeePlus_back] - nu_e_forSweep ) < 1e-6)
                                                                                                    
        if np.size(indFor_solutionsMatch) > 0:
            
            JeePlus_critical[ind_nActive] = JeePlus_back[ind_JeePlus_back]
            break
        
        
#%% SAVE THE RESULTS


results = {}
results['sim_params'] = s_params
results['mft_params'] = mft_params
results['JeePlus_backSweep_results'] = JeePlus_backSweep_results
results['JeePlus_forSweep_results'] = JeePlus_forSweep_results
results['JeePlus_critical'] = JeePlus_critical


fname =  ( ('%s_%s') % ( simID, net_type) )

if mft_params_dict['solve_reduced'] == True:
    save_name = save_path + fname + '_reducedMFT_sweepJeePlus_baseline.mat'

        
else:
    save_name = save_path + fname + '_MFT_sweepJeePlus_baseline.mat'


savemat(save_name, results)


        
    

