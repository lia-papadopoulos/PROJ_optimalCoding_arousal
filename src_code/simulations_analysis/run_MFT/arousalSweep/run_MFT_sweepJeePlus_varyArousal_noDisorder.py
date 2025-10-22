

#%% BASIC IMPORTS

import sys
import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
import argparse

sys.path.append('../../../')
import global_settings

sys.path.append(global_settings.path_to_src_code)    
sys.path.append(global_settings.path_to_src_code + 'functions/')  
sys.path.append(global_settings.path_to_src_code + 'MFT/funcs_MFT/clusteredNets_fullTheory/') 
 
from fcn_compute_firing_stats import Dict2Class
import master_MFT_fixedInDeg_EIclusters    

#%% SET PATHS

# path for saving data
save_path = global_settings.path_to_sim_output + 'MFT_sweep_JeePlus_arousalSweep/'

#%% ARGPARSER    

parser = argparse.ArgumentParser() 


# swept parameter name
parser.add_argument('-simID', '--simID', type=str)
parser.add_argument('-net_type', '--net_type', type=str)
parser.add_argument('-stim_shape', '--stim_shape', type=str)
parser.add_argument('-stim_rel_amp', '--stim_rel_amp', type=float)
parser.add_argument('-simParams_path', '--simParams_path', type=str)
parser.add_argument('-sweep_param_name', '--sweep_param_name', type=str)
parser.add_argument('-indParamSweep', '--indParamSweep', type = int)
    
# arguments of parser
args = parser.parse_args()


#-------------------- argparser values for later use -------------------------#
    

# name of swept parameter
simID = args.simID
net_type = args.net_type
stim_shape = args.stim_shape
stim_rel_amp = args.stim_rel_amp
simParams_path = args.simParams_path
sweep_param_name = args.sweep_param_name
indParamSweep = args.indParamSweep


#%% MFT PARAMETERS

mft_params_dict = {}

mft_params_dict['solve_reduced'] = True
mft_params_dict['nSteps_MFT_DynEqs'] = 30000
mft_params_dict['dt_MFT_DynEqs'] = 1e-4
mft_params_dict['nu_vec'] = np.nan
mft_params_dict['tau_e_MFT_DynEqs'] = 1e-3
mft_params_dict['tau_i_MFT_DynEqs'] = 1e-3
mft_params_dict['stopThresh_MFT_DynEqs'] = 1e-6
mft_params_dict['plot_MFT_DynEqs'] = False

mft_params_dict['min_JplusEE'] = 12
mft_params_dict['max_JplusEE'] = 25
mft_params_dict['delta_JplusEE'] = 0.025
mft_params_dict['nu_high_E'] = 80
mft_params_dict['nu_high_I'] = 60
mft_params_dict['nu_low_E'] = 1
mft_params_dict['nu_low_I'] = 1

mft_params_dict['stability_tau_e'] = 20e-3
mft_params_dict['stability_tau_i'] = 20e-3 

mft_params_dict['n_active_clusters_sweep'] = np.arange(1,13,1)
    

#%% LOAD IN MFT SIMULATION PARAMETERS

filename = ('%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f_simParams_mft_noDisorder.mat' % \
            ( simID, net_type, sweep_param_name, stim_shape, stim_rel_amp) )

simParams_mft = loadmat(simParams_path + filename, simplify_cells=True)   


#%% SET PARAMETERS FOR GIVEN VALUE OF SWEPT PARAMETERS INDEX

if ( sweep_param_name == 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread' or sweep_param_name == 'Jee_reduction_nu_ext_e_uniform_spread_nu_ext_i_uniform_spread'):
    
    simParams_mft['Jee'] = simParams_mft['Jee_sweep_vals'][indParamSweep]
    simParams_mft['nu_ext_e'][:] = simParams_mft['nu_ext_e_sweep_vals'][indParamSweep]
    simParams_mft['nu_ext_i'][:] = simParams_mft['nu_ext_i_sweep_vals'][indParamSweep]
    
    print('running')
        
else:
    print('failed')
    print(sweep_param_name)
    sys.exit()


#%% GET SIMULATION AND MFT PARAMETERS IN CLASS FORMAT
   
# sim_params         
s_params = Dict2Class(simParams_mft)

# mft_params
mft_params = Dict2Class(mft_params_dict)


#%% FORWARDS AND BACKWARDS SWEEPS OVER JEEPLUS FOR DIFFERENT NUMBERS OF ACTIVE CLUSTERS


# perform the backwards sweep over JeePlus
JeePlus_backSweep_results_full = master_MFT_fixedInDeg_EIclusters.fcn_JeePlus_sweep_backwards(s_params, mft_params)

# only save active, inactive, background rates    
JeePlus_backSweep_results = {}
JeePlus_backSweep_results['JplusEE_back'] = JeePlus_backSweep_results_full['JplusEE_back']
JeePlus_backSweep_results['nu_e_backSweep'] = JeePlus_backSweep_results_full['nu_e_backSweep'][np.array([0,-2,-1])]
JeePlus_backSweep_results['nu_i_backSweep'] = JeePlus_backSweep_results_full['nu_i_backSweep'][np.array([0,-2,-1])]


# perform the forwards sweep over JeePlus
JeePlus_forSweep_results_full = master_MFT_fixedInDeg_EIclusters.fcn_JeePlus_sweep_forwards(s_params, mft_params)

# only save active, inactive, background rates    
JeePlus_forSweep_results = {}
JeePlus_forSweep_results['JplusEE_for'] = JeePlus_forSweep_results_full['JplusEE_for']    
JeePlus_forSweep_results['nu_e_forSweep'] = JeePlus_forSweep_results_full['nu_e_forSweep'][np.array([0,-2,-1])]
JeePlus_forSweep_results['nu_i_forSweep'] = JeePlus_forSweep_results_full['nu_i_forSweep'][np.array([0,-2,-1])]
    

#%% AROUSAL LEVEL AS PERCENTAGE

arousal_level = s_params.arousal_levels[indParamSweep]*100

print(arousal_level)

    
#%% SAVE THE DATA


results = {}
results['sim_params'] = s_params
results['mft_params'] = mft_params
results['JeePlus_backSweep_results'] = JeePlus_backSweep_results
results['JeePlus_forSweep_results'] = JeePlus_forSweep_results

save_name = ( '%s_%s_%s_%0.3fpct_stimType_%s_stim_rel_amp%0.3f_reducedMFT_noDisorder_sweepJeePlus.mat' % \
            ( simID, net_type, sweep_param_name, arousal_level, stim_shape, stim_rel_amp ) )
        
 
savemat(save_path + save_name, results)


        
    

