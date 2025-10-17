
'''
This script generates
    Fig2D
'''

#%% basic imports
import numpy as np
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
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
font_path = '/home/liap/fonts/Arial.ttf'  # Your font path goes here
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()
plt.rcParams["mathtext.default"]="regular"
plt.rcParams['axes.linewidth'] = 0.5

#%% settings

outpath = global_settings.path_to_manuscript_figs_final + 'decoding_schematics/'
figID = 'Fig2D'

cmap = cm.get_cmap('plasma', 5)
cmap = cmap(range(5))
cmap_im = ListedColormap(cmap)


#%% make figure directory
    
if os.path.isdir(outpath) == False:
    os.makedirs(outpath)


#%% plot 2d schematic

# stimulus 1
x1=np.random.normal(1,0.175,100)
y1=np.random.normal(1,0.175,100)

# stimulus 2
x2=np.random.normal(0.2,0.175,100)
y2=np.random.normal(0.7,0.175,100)

# stimulus 3
x3=np.random.normal(0.7,0.175,100)
y3=np.random.normal(0.2,0.175,100)


plt.rcParams.update({'font.size': 7})
fig = plt.figure(figsize=(0.8,1.0))  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 

ax.plot(x1,y1,'.', color='navy', label='frequency 1', markersize=1.5)
ax.plot(x2,y2,'.', color='lightseagreen', label='frequency 3', markersize=1.5)
ax.plot(x3,y3,'.', color='darkviolet', label='frequency 5', markersize=1.5)

ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xlabel('cell 1 spike count')
ax.set_ylabel('cell 2 spike count')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig(outpath + figID + '.pdf', bbox_inches='tight', pad_inches=0, transparent=True)
