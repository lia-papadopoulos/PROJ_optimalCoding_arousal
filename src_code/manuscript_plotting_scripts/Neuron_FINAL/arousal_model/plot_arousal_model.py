
'''
This script generates
    Fig3C
'''

#%% standard imports
import sys
import numpy as np
from scipy.io import loadmat
import importlib
import os

#%% import global settings file
sys.path.append('../../../')
import global_settings

#%% plotting
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_path = global_settings.path_to_plotting_font  # Your font path goes here
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()
plt.rcParams["mathtext.default"]="regular"
plt.rcParams['axes.linewidth'] = 0.5

#%% load my functions

func_path = global_settings.path_to_src_code + 'functions/'
func_path0 = global_settings.path_to_src_code + 'run_simulations/'
                 
sys.path.append(func_path) 
from fcn_simulation_loading import fcn_set_sweepParam_string

sys.path.append(func_path0)
from fcn_simulation_setup import fcn_define_arousalSweep

#%% settings           
sim_params_path = global_settings.path_to_src_code + 'run_simulations/'
load_path = global_settings.path_to_sim_output
fig_path = global_settings.path_to_manuscript_figs_final + 'arousal_model/'
simParams_fname = 'simParams_051325_clu'
sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'
net_type = 'baseEIclu'
netInd = 0
stimInd = 0
trialInd = 0

figID1 = 'Fig3Ci'
figID2 = 'Fig3Cii_L'
figID3 = 'Fig3Cii_R'
figID4 = 'Fig3Cii'

#%% load sim parameters
sys.path.append(sim_params_path)
params = importlib.import_module(simParams_fname) 
s_params = params.sim_params

#%% arousal sweep
s_params = fcn_define_arousalSweep(s_params)

#%% unpack sim params
simID = s_params['simID']
stim_shape = s_params['stim_shape']
stim_rel_amp = s_params['stim_rel_amp']
n_sweepParams = s_params['nParams_sweep']
swept_params_dict = s_params['swept_params_dict']

n_arousalLevels = np.size(swept_params_dict['param_vals1'])
all_arousal_levels = s_params['arousal_levels']*100

del s_params
del params

#%% make output directory
if os.path.isdir(fig_path) == False:
    os.makedirs(fig_path)
    

#%% START PLOTTING...



#%% plot Jee vs arousal strength

Jee_vs_arousal = np.zeros(n_arousalLevels)

for ind_sweep_param in range(0, n_arousalLevels):

    sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, ind_sweep_param)     
    fname = ( '%s%s_%s_sweep_%s_network%d_IC%d_stim%d_stimType_%s_stim_rel_amp%0.3f_simulationData.mat' % \
            ( load_path, simID, net_type, sweep_param_str, netInd, trialInd, stimInd, stim_shape, stim_rel_amp) )
    data = loadmat(fname, simplify_cells=True)                
    s_params = data['sim_params']

    Jee_vs_arousal[ind_sweep_param] = s_params['Jee']
    

plt.rcParams.update({'font.size': 8})
fig = plt.figure(figsize=(1.3, 1.3))  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 

ax.plot(all_arousal_levels, Jee_vs_arousal/Jee_vs_arousal[0], color='darkblue')
ax.set_xlabel('arousal level [%]')
ax.set_ylabel(r'$J_{\mathrm{EE}} / J_{\mathrm{EE}}(0)$')
plt.savefig( ('%s%s.pdf' % (fig_path, figID1)) , bbox_inches='tight', pad_inches=0, transparent=True)


#%% plot external input distributions for low arousal

# lowest arousal
ind_sweep_param = 0    

# beginning of filename
sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, ind_sweep_param)     
fname = ( '%s%s_%s_sweep_%s_network%d_IC%d_stim%d_stimType_%s_stim_rel_amp%0.3f_simulationData.mat' % \
        ( load_path, simID, net_type, sweep_param_str, netInd, trialInd, stimInd, stim_shape, stim_rel_amp) )

# load data
data = loadmat(fname, simplify_cells=True)                
s_params = data['sim_params']

# inputs
nu_ext_ee = s_params['pert_nu_ext_ee'] + s_params['nu_ext_ee'] 
nu_ext_ie = s_params['pert_nu_ext_ie'] + s_params['nu_ext_ie'] 

# all inputs
all_inputs_low_arousal = np.append(nu_ext_ee, nu_ext_ie)

# plot
plt.rcParams.update({'font.size': 6})
fig = plt.figure(figsize=(0.5,0.5))  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 
ax.hist(all_inputs_low_arousal, bins=np.arange(6.5,20.5,1), density=False, color='dimgrey')
ax.set_xlabel('ext. input')
ax.set_ylabel('cell count')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim([6.,20])
plt.savefig( ('%s%s.pdf' % (fig_path, figID2)) , bbox_inches='tight', pad_inches=0, transparent=True)


#%% plot external input distributions for highest arousal

# highest arousal
ind_sweep_param = n_arousalLevels - 1   

# beginning of filename
sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, ind_sweep_param)     
fname = ( '%s%s_%s_sweep_%s_network%d_IC%d_stim%d_stimType_%s_stim_rel_amp%0.3f_simulationData.mat' % \
        ( load_path, simID, net_type, sweep_param_str, netInd, trialInd, stimInd, stim_shape, stim_rel_amp) )

# load data
data = loadmat(fname, simplify_cells=True)                
s_params = data['sim_params']

# inputs
nu_ext_ee = s_params['pert_nu_ext_ee'] + s_params['nu_ext_ee'] 
nu_ext_ie = s_params['pert_nu_ext_ie'] + s_params['nu_ext_ie'] 

# all inputs
all_inputs_high_arousal = np.append(nu_ext_ee, nu_ext_ie)

# plot
plt.rcParams.update({'font.size': 6})
fig = plt.figure(figsize=(0.5,0.5))  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 
ax.hist(all_inputs_high_arousal, bins=np.arange(6.5,20.5,1), density=False, color='dimgrey')
ax.set_xlabel('ext. input')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim([6.,20])
plt.savefig( ('%s%s.pdf' % (fig_path, figID3)) , bbox_inches='tight', pad_inches=0, transparent=True)


#%% plot nu_ext vs arousal strength (E and I separately)

Ne = s_params['N_e']
Ni = s_params['N_i']
N = s_params['N']

nu_ext_ee_vs_arousal = np.zeros((Ne, n_arousalLevels))
nu_ext_ie_vs_arousal = np.zeros((Ni, n_arousalLevels))


for ind_sweep_param in range(0, n_arousalLevels):

    sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, ind_sweep_param)     
    fname = ( '%s%s_%s_sweep_%s_network%d_IC%d_stim%d_stimType_%s_stim_rel_amp%0.3f_simulationData.mat' % \
            ( load_path, simID, net_type, sweep_param_str, netInd, trialInd, stimInd, stim_shape, stim_rel_amp) )
    data = loadmat(fname, simplify_cells=True)                
    s_params = data['sim_params']

    # save JEE
    nu_ext_ee_vs_arousal[:,ind_sweep_param] = s_params['pert_nu_ext_ee'] + s_params['nu_ext_ee'] 
    nu_ext_ie_vs_arousal[:,ind_sweep_param] = s_params['pert_nu_ext_ie'] + s_params['nu_ext_ie']    

plt.rcParams.update({'font.size': 8})
fig = plt.figure(figsize=(1.3, 1.3))  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 

indCell_minE = np.argsort(nu_ext_ee_vs_arousal[:,-1])[25]
indCell_midE = np.argsort(nu_ext_ee_vs_arousal[:,-1])[int(Ne/3.75)]
indCell_maxE = np.argsort(nu_ext_ee_vs_arousal[:,-1])[-1]

indCell_minI = np.argsort(nu_ext_ie_vs_arousal[:,-1])[0]
indCell_midI = np.argsort(nu_ext_ie_vs_arousal[:,-1])[int(Ni/1.25)]
indCell_maxI = np.argsort(nu_ext_ie_vs_arousal[:,-1])[-15]


ax.plot(all_arousal_levels, nu_ext_ee_vs_arousal[indCell_minE,:]/nu_ext_ee_vs_arousal[indCell_minE,0], color='cornflowerblue')
ax.plot(all_arousal_levels, nu_ext_ee_vs_arousal[indCell_maxE,:]/nu_ext_ee_vs_arousal[indCell_maxE,0], color='darkblue')
ax.plot(all_arousal_levels, nu_ext_ee_vs_arousal[indCell_midE,:]/nu_ext_ee_vs_arousal[indCell_midE,0], color='blue', label='E cells')

ax.plot(all_arousal_levels, nu_ext_ie_vs_arousal[indCell_minI,:]/nu_ext_ie_vs_arousal[indCell_minI,0], color='indianred')
ax.plot(all_arousal_levels, nu_ext_ie_vs_arousal[indCell_maxI,:]/nu_ext_ie_vs_arousal[indCell_maxI,0], color='darkred')
ax.plot(all_arousal_levels, nu_ext_ie_vs_arousal[indCell_midI,:]/nu_ext_ie_vs_arousal[indCell_midI,0], color='red',label='I cells')

ax.set_xlabel('arousal level [%]')
ax.set_ylabel(r'$\nu_{\mathrm{ext}} / \nu_{\mathrm{ext}}(0)$')
ax.legend(fontsize=6,frameon=False)
plt.savefig( ('%s%s.pdf' % (fig_path, figID4)) , bbox_inches='tight', pad_inches=0, transparent=True)