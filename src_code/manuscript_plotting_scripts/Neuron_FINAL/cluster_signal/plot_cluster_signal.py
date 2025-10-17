
'''
This script generates
    Fig7B
'''


#%% basic imports
from scipy.io import loadmat
import importlib
import sys
import os

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

figureSize = (2.2, 1.9)
fontSize = 8

figID = 'Fig7B'

#%% settings
fig_path = global_settings.path_to_manuscript_figs_final + 'cluster_signal/'
load_path = global_settings.path_to_sim_output + 'deltaRate_selective_nonselective/'
sim_params_path =  global_settings.path_to_src_code + 'run_simulations/'
simParams_fname = 'simParams_051325_clu'
sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'
net_type = 'baseEIclu'

#%% load sim parameters
sys.path.append(sim_params_path)
params = importlib.import_module(simParams_fname) 
s_params = params.sim_params

#%% unpack sim params
simID = s_params['simID']
stim_shape = s_params['stim_shape']
arousal_level = s_params['arousal_levels']*100

del s_params
del params

#%% filenames

fname = ( ('%s_%s_sweep_%s_stimType%s_selective_nonselective_rates.mat') % (simID, net_type, sweep_param_name, stim_shape) )
fig_name = ( ('%s_%s_sweep_%s_') % (simID, net_type, sweep_param_name) )

# make output directory
if os.path.isdir(fig_path) == False:
    os.makedirs(fig_path)

#%% load the data
data = loadmat(load_path + fname, simplify_cells = True)

#%% unpack the data

rates_diff_trialAvg_stimAvg_netAvg = data['rates_diff_trialAvg_stimAvg_netAvg']
rates_diff_trialAvg_stimAvg_netStd = data['rates_diff_trialAvg_stimAvg_netSd']

#%% plot cluster signal

plt.rcParams.update({'font.size': fontSize})
fig = plt.figure(figsize=figureSize)  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 
color = [0.2, 0.2, 0.2]

x = arousal_level
y = rates_diff_trialAvg_stimAvg_netAvg
yerr = rates_diff_trialAvg_stimAvg_netStd
ax.errorbar(x, y, yerr=yerr, xerr=None, color=color, linewidth=1, fmt='none')
ax.plot(x, y, '-o',  color=color, linewidth=1, markersize=2)

ax.set_xlim([-2,102])
plt.xlabel('arousal level [%]')
plt.ylabel('cluster signal $C_{s}$ [sp/s]')
plt.savefig(fig_path + figID + '.pdf', bbox_inches='tight', pad_inches=0, transparent=True)

