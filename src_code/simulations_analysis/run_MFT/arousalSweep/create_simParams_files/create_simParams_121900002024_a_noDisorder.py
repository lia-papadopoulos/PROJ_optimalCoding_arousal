#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
get simulation parameters necessary for mft from a set of simulations
in this case, the background inputs to e and i cells were drawn from a beta
distribution in the simulations; for the mft, use parameters corresponding to
the mean of the input distribution
"""

import sys
import numpy as np
from scipy.io import loadmat
from scipy.io import savemat

sys.path.insert(0, '/home/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/scripts/functions/')    
    
from fcn_simulation_loading import fcn_set_sweepParam_string

# paths for loading and saving data
load_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/'\
             'PROJ_VariabilityGainMod/data_files/test_stim_expSyn/')

save_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/'\
             'PROJ_VariabilityGainMod/data_files/test_stim_expSyn/simParams_mft/')
    
     
# simulation ID
simID = '121900002024_aClu'  

# netowrk name
net_type = 'baseEIclu'

# stim shape
stim_shape = 'diff2exp'

# relative stimulation amplitude
stim_rel_amp = 0.05

# sweep param name
sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'

# number of swept parameters
n_sweepParams = 3

# swept parameter values
swept_params_dict = {}
swept_params_dict['param_vals1'] = np.linspace(0, 0.75, 9)
swept_params_dict['param_vals2']= np.linspace(0, 13, 9)
swept_params_dict['param_vals3']= np.linspace(0, 13, 9)

# number of values for mft sweep
n_paramVals_mft = 50

# number of arousal levels
n_paramVals_sim = np.size(swept_params_dict['param_vals1'])

#%% load simulations for lowest arousal

indParam = 0
    
sweep_param_str_val = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, indParam)

fname =  ( '%s%s_%s_sweep_%s_network%d_IC%d_stim%d_stimType_%s_stim_rel_amp%0.3f' % \
         ( load_path, simID, net_type, sweep_param_str_val, 0, 0, 0, stim_shape, stim_rel_amp) )
    
# load data
data = loadmat((('%s_simulationData.mat') % fname), simplify_cells=True)                

# get simulation parameters
s_params = data['sim_params']

# get values of swept parameters
Jee = s_params['Jee']
nu_ext_ee_beta_spread = s_params['nu_ext_ee_beta_spread']
nu_ext_ie_beta_spread = s_params['nu_ext_ie_beta_spread']
mean_nu_ext_e = s_params['mean_nu_ext_ee']
mean_nu_ext_i = s_params['mean_nu_ext_ie']

# set parameters for mft sweep
Jee_mft_low = Jee
nu_ext_e_mft_low = mean_nu_ext_e + nu_ext_ee_beta_spread/2
nu_ext_i_mft_low = mean_nu_ext_i + nu_ext_ie_beta_spread/2


#%% make new simulation parameter dictionary for mft

s_params_mft = dict()

# all values needed for mft
s_params_mft['extCurrent_poisson'] = s_params['base_extCurrent_poisson']
s_params_mft['tau_r'] = s_params['tau_r']             
s_params_mft['tau_m_e'] = s_params['tau_m_e']          
s_params_mft['tau_m_i'] = s_params['tau_m_i']          
s_params_mft['tau_s_e'] = s_params['tau_s_e']         
s_params_mft['tau_s_i'] = s_params['tau_s_i']          
s_params_mft['Vr_e'] = s_params['Vr_e']               
s_params_mft['Vr_i'] = s_params['Vr_i']                
s_params_mft['Vth_e'] = s_params['Vth_e']              
s_params_mft['Vth_i'] = s_params['Vth_i']              
s_params_mft['Cee'] = s_params['Cee'] 
s_params_mft['Cei'] = s_params['Cei'] 
s_params_mft['Cii'] = s_params['Cii'] 
s_params_mft['Cie'] = s_params['Cie'] 
s_params_mft['Cext'] = s_params['pext_ee']*s_params['N_e']
s_params_mft['nu_ext_e'] = s_params['nu_ext_ee']       
s_params_mft['nu_ext_i'] = s_params['nu_ext_ie'] 
s_params_mft['Jee'] = s_params['Jee']
s_params_mft['Jei'] = s_params['Jei'] 
s_params_mft['Jii'] = s_params['Jii'] 
s_params_mft['Jie'] = s_params['Jie'] 
s_params_mft['Jee_ext'] = s_params['Jee_ext']        
s_params_mft['Jie_ext'] = s_params['Jie_ext'] 
s_params_mft['p'] = s_params['p']
s_params_mft['bgrE'] = s_params['bgrE']
s_params_mft['bgrI'] = s_params['bgrI']
s_params_mft['JplusEE'] = s_params['JplusEE']
s_params_mft['JplusEI'] = s_params['JplusEI']    
s_params_mft['JplusIE'] = s_params['JplusIE']
s_params_mft['JplusII'] = s_params['JplusII']     
s_params_mft['clusters'] = s_params['clusters']
s_params_mft['clusterWeights'] = s_params['clusterWeights']
s_params_mft['depress_interCluster'] = s_params['depress_interCluster']
s_params_mft['n_paramVals_mft'] = n_paramVals_mft
s_params_mft['sweep_param_name'] = sweep_param_name
s_params_mft['swept_params_dict_sims'] = swept_params_dict


#%% load simulations for highest arousal

indParam = n_paramVals_sim-1
    
sweep_param_str_val = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, indParam)

fname =  ( '%s%s_%s_sweep_%s_network%d_IC%d_stim%d_stimType_%s_stim_rel_amp%0.3f' % \
         ( load_path, simID, net_type, sweep_param_str_val, 0, 0, 0, stim_shape, stim_rel_amp) )
    
# load data
data = loadmat((('%s_simulationData.mat') % fname), simplify_cells=True)                

# get simulation parameters
s_params = data['sim_params']

# get values of swept parameters
Jee = s_params['Jee']
nu_ext_ee_beta_spread = s_params['nu_ext_ee_beta_spread']
nu_ext_ie_beta_spread = s_params['nu_ext_ie_beta_spread']
mean_nu_ext_e = s_params['mean_nu_ext_ee']
mean_nu_ext_i = s_params['mean_nu_ext_ie']


# set parameters for mft sweep
Jee_mft_high = Jee
nu_ext_e_mft_high = mean_nu_ext_e + nu_ext_ee_beta_spread/2
nu_ext_i_mft_high = mean_nu_ext_i + nu_ext_ie_beta_spread/2


#%% set values of swept parameters in mft params

s_params_mft['Jee_sweep_vals'] = np.linspace(Jee_mft_low, Jee_mft_high, n_paramVals_mft)
s_params_mft['nu_ext_e_sweep_vals'] = np.linspace(nu_ext_e_mft_low, nu_ext_e_mft_high, n_paramVals_mft)
s_params_mft['nu_ext_i_sweep_vals'] = np.linspace(nu_ext_i_mft_low, nu_ext_i_mft_high, n_paramVals_mft)

    
#%% save the data

savename =  ('%s%s_%s_sweep_%s_stimType_%s_stim_rel_amp%0.3f_simParams_mft_noDisorder.mat' % \
            ( save_path, simID, net_type, sweep_param_name, stim_shape, stim_rel_amp) )
    
savemat(savename, s_params_mft)
    