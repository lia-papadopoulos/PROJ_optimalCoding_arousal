
"""
generate and save the simulation parameters required to run mft for a particular set of simulations

in this case, the background inputs to e and i cells were drawn from a beta
distribution in the simulations; for the mft, use parameters corresponding to
the mean of the input distribution
"""

#%% basic imports
import sys
import numpy as np
from scipy.io import savemat
import importlib
import copy

sys.path.append('../../../')
import global_settings
    
# paths for loading and saving data
func_path1 = global_settings.path_to_src_code + 'run_simulations/'
sim_params_path = global_settings.path_to_src_code + 'run_simulations/'
load_path = global_settings.path_to_sim_output + ''
save_path = global_settings.path_to_sim_output + 'simParams_mft/'
    
# simulation params
simParams_fname = 'simParams_051325_clu'
net_type = 'baseEIclu'
sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'


# number of values for mft sweep
n_paramVals_mft = 50
mft_sweepParam1 = 'Jee_reduction'
mft_sweepParam2 = 'pert_mean_nu_ext_ee'
mft_sweepParam3 = 'pert_mean_nu_ext_ie'

paramInds_toHalve = np.array([2,3])

#%% custom functions

sys.path.append(func_path1)    
from fcn_simulation_setup import fcn_define_arousalSweep
from fcn_simulation_setup import fcn_updateParams_givenArousal
from fcn_simulation_setup import fcn_basic_setup

#%% load parameters
sys.path.append(sim_params_path)
params = importlib.import_module(simParams_fname) 
default_params = params.sim_params
del params

#%% unpack parmeters
simID = default_params['simID']
stim_shape = default_params['stim_shape']
stim_rel_amp = default_params['stim_rel_amp']

#%% make copy of default parameters for setting up mft
s_params_mft = copy.deepcopy(default_params)


#%% define arousal sweep for sims

default_params = fcn_define_arousalSweep(default_params)
arousal_sweep_dict_sims = default_params['swept_params_dict']

#%% define arousal sweep for mft

# update number of arousal samples
minArousal = s_params_mft['arousal_levels'][0]
maxArousal = s_params_mft['arousal_levels'][-1]
s_params_mft['arousal_levels'] = np.linspace(minArousal, maxArousal, n_paramVals_mft)
# basic setup
s_params_mft = fcn_basic_setup(s_params_mft)
# arousal sweep
s_params_mft = fcn_define_arousalSweep(s_params_mft)


#%% get dictionary of swept parameters for arousal model
arousal_sweep_dict_mft = s_params_mft['swept_params_dict']

# update arousal sweep values based on fact that we want homogeneous external input modulation
for iParam in paramInds_toHalve:
    arousal_sweep_dict_mft['param_vals%d' % iParam] = arousal_sweep_dict_mft['param_vals%d' % iParam]/2
    
    
#%% set parameters for mft sweep

Jee_sweep_vals = np.ones(n_paramVals_mft)*np.nan
nu_ext_e_sweep_vals = np.ones(n_paramVals_mft)*np.nan
nu_ext_i_sweep_vals = np.ones(n_paramVals_mft)*np.nan

for indMft in range(0, n_paramVals_mft):
    
    # set arousal parameters in s_params_mft
    s_params_mft[mft_sweepParam1] = arousal_sweep_dict_mft['param_vals1'][indMft]
    s_params_mft[mft_sweepParam2] = arousal_sweep_dict_mft['param_vals2'][indMft]
    s_params_mft[mft_sweepParam3] = arousal_sweep_dict_mft['param_vals3'][indMft]

    # update parameters given arousal
    s_params_mft = fcn_updateParams_givenArousal(s_params_mft, 0)

    # save
    Jee_sweep_vals[indMft] = s_params_mft['Jee']
    nu_ext_e_sweep_vals[indMft] = s_params_mft['mean_nu_ext_ee'] + s_params_mft['pert_mean_nu_ext_ee']
    nu_ext_i_sweep_vals[indMft] = s_params_mft['mean_nu_ext_ie'] + s_params_mft['pert_mean_nu_ext_ie']
    
    
#%% make new simulation parameter dictionary for mft


# all values needed for mft
s_params_mft['extCurrent_poisson'] = s_params_mft['base_extCurrent_poisson']
s_params_mft['Cext'] = s_params_mft['pext_ee']*s_params_mft['N_e']
s_params_mft['nu_ext_e'] = s_params_mft['nu_ext_ee']       
s_params_mft['nu_ext_i'] = s_params_mft['nu_ext_ie'] 
s_params_mft['n_paramVals_mft'] = n_paramVals_mft
s_params_mft['swept_params_dict_sims'] = arousal_sweep_dict_sims
s_params_mft['swept_params_dict_mft'] = arousal_sweep_dict_mft
s_params_mft['sweep_param_name'] = sweep_param_name


#%% set arousal sweep for mft

s_params_mft['Jee_sweep_vals'] = Jee_sweep_vals
s_params_mft['nu_ext_e_sweep_vals'] = nu_ext_e_sweep_vals
s_params_mft['nu_ext_i_sweep_vals'] = nu_ext_i_sweep_vals

    
#%% save the data

savename =  ('%s%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f_simParams_mft_noDisorder.mat' % \
            ( save_path, simID, net_type, sweep_param_name, stim_shape, stim_rel_amp) )
    
savemat(savename, s_params_mft)
    