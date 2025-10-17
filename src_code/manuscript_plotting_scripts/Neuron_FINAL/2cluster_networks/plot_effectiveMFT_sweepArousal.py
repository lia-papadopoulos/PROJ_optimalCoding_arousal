
"""
This script generates
    Fig 6C
    Fig 6D
    Fig GE
"""


#%% standard imports
import sys
import os
import glob
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d

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

#%% import functions
sys.path.append(global_settings.path_to_src_code + 'functions/')
from fcn_simulation_loading import fcn_set_sweepParam_string

#%% plotting parameters

# plotting
figureSize = (1.4, 1.4)
fontSize = 8
outpath = global_settings.path_to_manuscript_figs_final + '2cluster_networks/'

# arousals plot
paramVals_plot = np.array([10, 23, 40])
xticksPlot = np.array([],)
colors = [[0.2,0.2,0.2], [0.5,0.5,0.5],[0.8,0.8,0.8]]

# loading
sweep_param_name = 'Jee_reduction_nu_ext_ee_nu_ext_ie'
fName_begin = '051300002025_clu'
fName_middle = 'effectiveMFT_ALT3'
n_sweepParams = 3
load_path = ( (global_settings.path_to_2clusterMFT_output + '/effectiveMFT_sweep_%s_2Ecluster/') % (fName_begin) )

fig1ID = 'Fig6D'
fig2ID = 'Fig6C'
fig3ID = 'Fig6E'


#%% setup

files =  sorted(glob.glob(load_path + fName_begin + '_' + fName_middle + '*'))
data = loadmat(files[0], simplify_cells=True)
swept_params_dict = data['settings_dictionary']['swept_params_dict']

nParams = np.size(swept_params_dict['param_vals1'])
arousal_level = np.arange(0,nParams)/(nParams-1)*100
sigma_smoothing = np.ones(nParams)

### make output directory ###
if os.path.isdir(outpath) == False:
    os.makedirs(outpath)

#%% loop over parameter value and plot potential


for count, indParam in enumerate(paramVals_plot):
    
    sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, indParam) 

    fname = ( ('%s%s_%s_%s.mat') % (load_path, fName_begin, fName_middle, sweep_param_str))
    data = loadmat(fname, simplify_cells=True)

    x = data['path_position']
    y = data['U_of_r']
    ysmooth = gaussian_filter1d(y, sigma_smoothing[indParam])
    
    plt.rcParams.update({'font.size': fontSize})
    fig = plt.figure(figsize=(1.2,1.2))  
    ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 
    ax.plot(x, ysmooth, '-', linewidth=2, color=colors[count])
    ax.set_xlabel('path position r\n[sp/s]')
    
    ax.set_ylabel('potential U(r) [sp/s]$^2$')
    ax.set_xticks([])
    ax.set_ylim([0,26])
    ax.set_yticks([0, 12, 24])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    figName = (outpath + '%s_%s.pdf' % (fig1ID, count))
    plt.savefig(figName, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close('all')    
    
    
#%% plot potential for schematic

indParam = paramVals_plot[1] 
sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, indParam) 

fname = ( ('%s%s_%s_%s.mat') % (load_path, fName_begin, fName_middle, sweep_param_str))
data = loadmat(fname, simplify_cells=True)

x = data['path_position']
y = data['U_of_r']
ysmooth = gaussian_filter1d(y, sigma_smoothing[indParam])

plt.rcParams.update({'font.size': fontSize})
fig = plt.figure(figsize=(1.0,1.0))  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 
ax.plot(x, ysmooth, '-', linewidth=2, color=[0.5,0.5,0.5])
ax.set_xlabel('path position r')

ax.set_ylabel('potential U(r)')
ax.set_xticks([])
ax.set_ylim([0,10])
ax.set_yticks([])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
figName = (outpath + '%s.pdf' % (fig2ID))
plt.savefig(figName, bbox_inches='tight', pad_inches=0, transparent=True)
plt.close('all')   


#%% loop over parameter value and plot barrier height

barrier_height = np.zeros(nParams)

for indParam in range(0, nParams):
    
    sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, indParam) 

    fname = ( ('%s%s_%s_%s.mat') % (load_path, fName_begin, fName_middle, sweep_param_str))
    data = loadmat(fname, simplify_cells=True)

    r = data['path_position']
    potential = data['U_of_r']

    barrier_loc = int((np.size(r)-1)/2)
    
    barrier_height[indParam] = potential[barrier_loc]
    


### plot
plt.rcParams.update({'font.size': fontSize})
fig = plt.figure(figsize=figureSize)  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 
plt_color = 'steelblue'

x = arousal_level
y = barrier_height
ax.plot( x, y, '-o', markersize=2, linewidth=1, color=plt_color)
ax.set_yticks([0, 12, 24])
ax.set_xlabel('arousal level [%]')
ax.set_ylabel('barrier height h [sp/s]$^2$')
figName = (outpath + '%s.pdf' % (fig3ID))
plt.savefig(figName, bbox_inches='tight', pad_inches=0, transparent=True)
plt.close('all')
