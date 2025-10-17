
"""
This script generates
    Fig S5F
    Fig S5G
    Fig S5H
"""


#%% standard imports
import sys
import numpy as np
from scipy.io import loadmat
import glob
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

#%% import functions
sys.path.append(global_settings.path_to_src_code + 'functions/')
from fcn_simulation_loading import fcn_set_sweepParam_string

#%% plot settings
outpath = global_settings.path_to_manuscript_figs_final + '2cluster_networks/'
indParam_plot = 0
sweep_param_name = 'Jee_reduction_nu_ext_ee_nu_ext_ie'
fName_begin = '051300002025_clu'
fName_middle = 'effectiveMFT_ALT3'
n_sweepParams = 3
load_path = ( (global_settings.path_to_2clusterMFT_output + '/effectiveMFT_sweep_%s_2Ecluster/') % (fName_begin) )

fig1ID = 'FigS5F'
fig2ID = 'FigS5GR'
fig3ID = 'FigS5GL'
fig4ID = 'FigS5GM'
fig5ID = 'FigS5H'


#%% setup

files =  sorted(glob.glob(load_path + fName_begin + '_' + fName_middle + '*'))
data = loadmat(files[0], simplify_cells=True)
swept_params_dict = data['settings_dictionary']['swept_params_dict']


### make output directory ###
if os.path.isdir(outpath) == False:
    os.makedirs(outpath)

              
#%% load data

sweep_param_str = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, indParam_plot) 
fname = ( ('%s%s_%s_%s.mat') % (load_path, fName_begin, fName_middle, sweep_param_str))
data = loadmat(fname, simplify_cells=True)


# unpack
path_distance_force = data['path_position']
path_distance_potential = data['path_position']
potential = data['U_of_r']
stable_fixedPoint = data['selective_fixedPoint']
unstable_fixedPoint = data['saddlePoint']
nu1_gridVals = data['nu1_grid']
nu2_gridVals = data['nu2_grid']
force1_gridVals = data['F_pop1']
force2_gridVals = data['F_pop2']
minEnergy_pathCoords_Stable1_to_Saddle = data['minEnergy_pathCoords_SaddletoA']
minEnergy_pathCoords_Stable2_to_Saddle = data['minEnergy_pathCoords_SaddletoB']
F_dot_dr = data['F_dot_dr']


#%% find index along path corresponding to stable and unstable fixed points


#%% Force field with minimum energy path

plt.rcParams.update({'font.size': 8})
fig = plt.figure(figsize=(1.8,1.8))  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 

# force field
ax.quiver(nu1_gridVals[::3,::3], nu2_gridVals[::3,::3], force1_gridVals[::3,::3], force2_gridVals[::3,::3])

# integration path
for i in range(0,np.shape(minEnergy_pathCoords_Stable1_to_Saddle)[0]-1):
    
    ax.plot([minEnergy_pathCoords_Stable1_to_Saddle[i,0],minEnergy_pathCoords_Stable1_to_Saddle[i+1,0]],\
             [minEnergy_pathCoords_Stable1_to_Saddle[i,1],minEnergy_pathCoords_Stable1_to_Saddle[i+1,1]],\
             '-o',color=np.array([120, 144, 156])/256,markersize=0.5, linewidth=0.75)
        
    ax.plot([minEnergy_pathCoords_Stable2_to_Saddle[i,0],minEnergy_pathCoords_Stable2_to_Saddle[i+1,0]],\
             [minEnergy_pathCoords_Stable2_to_Saddle[i,1],minEnergy_pathCoords_Stable2_to_Saddle[i+1,1]],\
             '-o',color=np.array([120, 144, 156])/256,markersize=0.5, linewidth=0.75)
    
# fixed points
ax.plot(stable_fixedPoint[0], stable_fixedPoint[1], 'o', markersize=3, color='r')
ax.plot(stable_fixedPoint[1], stable_fixedPoint[0], 'o', markersize=3, color='r')
ax.plot(unstable_fixedPoint[0], unstable_fixedPoint[1], 'o', markersize=4, markerfacecolor='none', markeredgecolor='r')

ax.arrow(minEnergy_pathCoords_Stable2_to_Saddle[10,0] ,minEnergy_pathCoords_Stable2_to_Saddle[10,1], -2, 2, color=np.array([120, 144, 156])/256, \
         width=0.15, head_width = 1.75, head_length = 1.5)


ax.set_xlabel(r'$\nu^{\mathrm{E,1}}_{\mathrm{in}}$ [spk/s]')
ax.set_ylabel(r'$\nu^{\mathrm{E,2}}_{\mathrm{in}}$ [spk/s]')

ax.set_yticks([1, 29, 57])
ax.set_xticks([1, 29, 57])

figName = (outpath + '%s.pdf' % (fig1ID))
plt.savefig(figName, bbox_inches='tight', pad_inches=0, transparent=True)


#%% zoom in on stable fixed point # 1


plt.rcParams.update({'font.size': 7})
fig = plt.figure(figsize=(1.1,1.1))  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 


upperX = stable_fixedPoint[1] + 5
lowerX = stable_fixedPoint[1] - 4

upperY = stable_fixedPoint[0] + 5
lowerY = stable_fixedPoint[0] - 4

indUpperX = np.argmin(np.abs(nu1_gridVals[0,:] - upperX))
indLowerX = np.argmin(np.abs(nu1_gridVals[0,:] - lowerX))

indUpperY = np.argmin(np.abs(nu2_gridVals[:,0] - upperY))
indLowerY = np.argmin(np.abs(nu2_gridVals[:,0] - lowerY))

nu1_gridVals_zoom = nu1_gridVals[:, indLowerX:indUpperX].copy()
nu1_gridVals_zoom = nu1_gridVals_zoom[indLowerY:indUpperY, :]
force1_gridVals_zoom = force1_gridVals[:, indLowerX:indUpperX].copy()
force1_gridVals_zoom = force1_gridVals_zoom[indLowerY:indUpperY, :].copy()

nu2_gridVals_zoom = nu2_gridVals[:, indLowerX:indUpperX].copy()
nu2_gridVals_zoom = nu2_gridVals_zoom[indLowerY:indUpperY, :]
force2_gridVals_zoom = force2_gridVals[:, indLowerX:indUpperX].copy()
force2_gridVals_zoom = force2_gridVals_zoom[indLowerY:indUpperY, :].copy()


# force field
ax.quiver(nu1_gridVals_zoom, nu2_gridVals_zoom, force1_gridVals_zoom, force2_gridVals_zoom)
    
# fixed points
ax.plot(stable_fixedPoint[1], stable_fixedPoint[0], 'o', markersize=4, color='r')

ax.set_xlabel(r'$\nu^{\mathrm{E,1}}$ [sp/s]')
ax.set_ylabel(r'$\nu^{\mathrm{E,2}}$ [sp/s]')
ax.set_xticks([1, 8])
ax.set_yticks([49, 56])


figName = (outpath + '%s.pdf' % (fig2ID))
plt.savefig(figName, bbox_inches='tight', pad_inches=0, transparent=True)


#%% zoom in on stable fixed point # 2


plt.rcParams.update({'font.size': 7})
fig = plt.figure(figsize=(1.1,1.1))  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 


upperX = stable_fixedPoint[0] + 5
lowerX = stable_fixedPoint[0] - 4

upperY = stable_fixedPoint[1] + 5
lowerY = stable_fixedPoint[1] - 4

indUpperX = np.argmin(np.abs(nu1_gridVals[0,:] - upperX))
indLowerX = np.argmin(np.abs(nu1_gridVals[0,:] - lowerX))

indUpperY = np.argmin(np.abs(nu2_gridVals[:,0] - upperY))
indLowerY = np.argmin(np.abs(nu2_gridVals[:,0] - lowerY))

nu1_gridVals_zoom = nu1_gridVals[:, indLowerX:indUpperX].copy()
nu1_gridVals_zoom = nu1_gridVals_zoom[indLowerY:indUpperY, :]
force1_gridVals_zoom = force1_gridVals[:, indLowerX:indUpperX].copy()
force1_gridVals_zoom = force1_gridVals_zoom[indLowerY:indUpperY, :].copy()

nu2_gridVals_zoom = nu2_gridVals[:, indLowerX:indUpperX].copy()
nu2_gridVals_zoom = nu2_gridVals_zoom[indLowerY:indUpperY, :]
force2_gridVals_zoom = force2_gridVals[:, indLowerX:indUpperX].copy()
force2_gridVals_zoom = force2_gridVals_zoom[indLowerY:indUpperY, :].copy()


# force field
ax.quiver(nu1_gridVals_zoom, nu2_gridVals_zoom, force1_gridVals_zoom, force2_gridVals_zoom)
    
# fixed points
ax.plot(stable_fixedPoint[0], stable_fixedPoint[1], 'o', markersize=4, color='r')

ax.set_xlabel(r'$\nu^{\mathrm{E,1}}$ [sp/s]')
ax.set_ylabel(r'$\nu^{\mathrm{E,2}}$ [sp/s]')
ax.set_yticks([1, 8])
ax.set_xticks([49, 56])


figName = (outpath + '%s.pdf' % (fig3ID))
plt.savefig(figName, bbox_inches='tight', pad_inches=0, transparent=True)


#%% zoom in on unstable fixed point

plt.rcParams.update({'font.size': 7})
fig = plt.figure(figsize=(1.1,1.1))  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 


upperX = unstable_fixedPoint[0] + 6
lowerX = unstable_fixedPoint[0] - 4

upperY = unstable_fixedPoint[1] + 6
lowerY = unstable_fixedPoint[1] - 4

indUpperX = np.argmin(np.abs(nu1_gridVals[0,:] - upperX))
indLowerX = np.argmin(np.abs(nu1_gridVals[0,:] - lowerX))

indUpperY = np.argmin(np.abs(nu2_gridVals[:,0] - upperY))
indLowerY = np.argmin(np.abs(nu2_gridVals[:,0] - lowerY))

nu1_gridVals_zoom = nu1_gridVals[:, indLowerX:indUpperX].copy()
nu1_gridVals_zoom = nu1_gridVals_zoom[indLowerY:indUpperY, :]
force1_gridVals_zoom = force1_gridVals[:, indLowerX:indUpperX].copy()
force1_gridVals_zoom = force1_gridVals_zoom[indLowerY:indUpperY, :].copy()

nu2_gridVals_zoom = nu2_gridVals[:, indLowerX:indUpperX].copy()
nu2_gridVals_zoom = nu2_gridVals_zoom[indLowerY:indUpperY, :]
force2_gridVals_zoom = force2_gridVals[:, indLowerX:indUpperX].copy()
force2_gridVals_zoom = force2_gridVals_zoom[indLowerY:indUpperY, :].copy()


# force field
ax.quiver(nu1_gridVals_zoom, nu2_gridVals_zoom, force1_gridVals_zoom, force2_gridVals_zoom)
    
# fixed points
ax.plot(unstable_fixedPoint[1], unstable_fixedPoint[0], 'o', markersize=5, markerfacecolor='none', markeredgecolor='r')

ax.set_xlabel(r'$\nu^{\mathrm{E,1}}$ [sp/s]')
ax.set_ylabel(r'$\nu^{\mathrm{E,2}}$ [sp/s]')
ax.set_yticks([25, 33])
ax.set_xticks([25, 33])

figName = (outpath + '%s.pdf' % (fig4ID))
plt.savefig(figName, bbox_inches='tight', pad_inches=0, transparent=True)


#%% plot potential energy

plt.rcParams.update({'font.size': 8})
fig = plt.figure(figsize=(1.8,1.8))  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 

ax.plot(path_distance_potential, potential, '-o', color=[0.5, 0.5, 0.5], markersize=2)
ax.plot([0, 0], [0, np.max(potential)], color='k')
ax.text(4, 10, 'h',fontsize=8)

ax.set_xlabel('path position r [sp/s]')
ax.set_ylabel(r'effective potential U [sp/s]$^{2}$')
ax.set_xticks([])

figName = (outpath + '%s.pdf' % (fig5ID))
plt.savefig(figName, bbox_inches='tight', pad_inches=0, transparent=True)

