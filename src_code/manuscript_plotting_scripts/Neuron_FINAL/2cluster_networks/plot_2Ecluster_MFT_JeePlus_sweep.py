
"""
This script generates
    Fig S5E
"""


#%% standard imports
from scipy.io import loadmat
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

#%% plotting parameters
load_path = global_settings.path_to_2clusterMFT_output + 'MFT_sweep_JeePlus_2Ecluster/'
outpath = global_settings.path_to_manuscript_figs_final + '2cluster_networks/'
fName_begin = '051300002025_clu'
special_JeePlus = 20.

figureID = 'FigS5E'

#%% setup

### make output directory ###
if os.path.isdir(outpath) == False:
    os.makedirs(outpath)

#%% load data
fname = ( ('%s%s_JeePlus_sweep_2cluster_fullMFT.mat') % (load_path, fName_begin))
data = loadmat(fname, simplify_cells=True)           

#%% plot Jplus EE sweep


plt.rcParams.update({'font.size': 8})
fig = plt.figure(figsize=(1.8,1.8))  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 


x = data['JplusEE_backwards']
yactive = data['nu_e_backwards'][0,:]
yinactive = data['nu_e_backwards'][1,:]

xfor = data['JplusEE_forwards']
yuniform = data['nu_e_forwards'][0,:]


ax.plot(x, yactive, '-', linewidth=1, color='lightseagreen', label='active')
ax.plot(x, yinactive, '-', linewidth=1, color='darkviolet', label='inactive')
ax.plot(xfor, yuniform, '-', linewidth=1, color='gray', label='uniform')

ax.plot([special_JeePlus, special_JeePlus], [0, 70], color='black')

ax.set_xlabel('E-to-E intracluster\nweight factor ($J_{EE}^{+}$)')
ax.set_ylabel('cluster rate [sp/s]')   
plt.legend(fontsize=6, loc='upper left')

figName = ( ((outpath + '%s.pdf') % (figureID)) )

plt.savefig(figName, bbox_inches='tight', pad_inches=0, transparent=True)

    

