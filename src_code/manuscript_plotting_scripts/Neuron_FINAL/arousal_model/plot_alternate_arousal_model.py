'''
This script generates
    FigS3A
'''

#%% basic imports
import sys
import numpy as np
from scipy.io import loadmat
import importlib

#%% import global settings file
sys.path.append('../../../')
import global_settings

#%% plotting
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_path = global_settings.path_to_plotting_font
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()
plt.rcParams["mathtext.default"]="regular"
plt.rcParams['axes.linewidth'] = 0.5


#%% load my functions
func_path0 = global_settings.path_to_src_code + 'run_simulations/'
func_path1 = global_settings.path_to_src_code + 'functions/'

sys.path.append(func_path0)
sys.path.append(func_path1)

from fcn_simulation_loading import fcn_set_sweepParam_string
from fcn_simulation_setup import fcn_define_arousalSweep

#%% settings  

# paths
sim_params_path = global_settings.path_to_src_code + 'run_simulations/'
load_path = global_settings.path_to_sim_output
fig_path = global_settings.path_to_manuscript_figs_final + 'alternate_arousal_model/'

# sim parameters         
simParams_fname = 'simParams_050925_clu'
sweep_param_name = 'zeroMean_sd_nu_ext_ee'
net_type = 'baseEIclu'
netInd = 0
stimInd = 0
trialInd = 0
figureSize = (1.35, 1.1)
figID = 'FigS3A'

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


#%% plot external input vs arousal

# lowest arousal
ind_sweep_param = 0    

# beginning of filename
sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, ind_sweep_param)     
fname = ( '%s%s_%s_sweep_%s_network%d_IC%d_stim%d_stimType_%s_stim_rel_amp%0.3f_simulationData.mat' % \
        ( load_path, simID, net_type, sweep_param_str, netInd, trialInd, stimInd, stim_shape, stim_rel_amp) )
    
# load data
data = loadmat(fname, simplify_cells=True)                
s_params = data['sim_params']
Ne = s_params['N_e']

# initialize
nu_ext_ee_vs_arousal = np.zeros((Ne, n_arousalLevels))

# loop over arousal
for ind_sweep_param in range(0, n_arousalLevels):

    sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, ind_sweep_param)     
    fname = ( '%s%s_%s_sweep_%s_network%d_IC%d_stim%d_stimType_%s_stim_rel_amp%0.3f_simulationData.mat' % \
            ( load_path, simID, net_type, sweep_param_str, netInd, trialInd, stimInd, stim_shape, stim_rel_amp) )
    data = loadmat(fname, simplify_cells=True)                
    s_params = data['sim_params']

    nu_ext_ee_vs_arousal[:, ind_sweep_param] = s_params['pert_nu_ext_ee'] + s_params['nu_ext_ee'] 

plt.rcParams.update({'font.size': 8})
fig = plt.figure(figsize=figureSize)  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 

indCell_min = np.argmin(nu_ext_ee_vs_arousal[:,-1])
indCell_mid = np.argsort(nu_ext_ee_vs_arousal[:,-1])[int(Ne/2)]
indCell_max = np.argmax(nu_ext_ee_vs_arousal[:,-1])

ax.plot(all_arousal_levels, nu_ext_ee_vs_arousal[indCell_min,:]/nu_ext_ee_vs_arousal[indCell_min,0], color='cornflowerblue')
ax.plot(all_arousal_levels, nu_ext_ee_vs_arousal[indCell_max,:]/nu_ext_ee_vs_arousal[indCell_max,0], color='darkblue')
ax.plot(all_arousal_levels, nu_ext_ee_vs_arousal[indCell_mid,:]/nu_ext_ee_vs_arousal[indCell_mid,0], color='blue', label='E cells')
ax.set_yticks([0,1,2,3])

ax.set_xlabel('arousal level [%]')
ax.set_ylabel(r'$\nu_{\mathrm{ext}}^{E} / \nu_{\mathrm{ext}}^{E}(0)$')
plt.savefig( ('%s%s.pdf' % (fig_path, figID)) , bbox_inches='tight', pad_inches=0, transparent=True)


plt.close('all')