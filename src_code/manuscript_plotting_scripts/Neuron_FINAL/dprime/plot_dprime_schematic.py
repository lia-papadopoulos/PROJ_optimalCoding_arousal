'''
This script generates
    Fig2B
'''

#%% basic imports
import sys
import os
import numpy as np

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

outpath = global_settings.path_to_manuscript_figs_final + 'dprime_schematic/'
figID = 'Fig2B'

cmap = cm.get_cmap('plasma', 5)
cmap = cmap(range(5))
cmap_im = ListedColormap(cmap)

#%% make figure directory
    
if os.path.isdir(outpath) == False:
    os.makedirs(outpath)

#%% define two gaussians

respDist_a = np.random.normal(-2,1.5,100000)
respDist_b = np.random.normal(2,1.5,100000)

plt.rcParams.update({'font.size': 7})
fig = plt.figure(figsize=(0.8,1.))  
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8]) 

hist_data_a = respDist_a
hist_data_b = respDist_b

bin_width = 0.2

data_extreme = np.nanmax(np.abs(np.append(hist_data_a, hist_data_b)))
bins = np.arange( np.round(-data_extreme - bin_width, 1), np.round(data_extreme + 2*bin_width, 1), bin_width )

counts, bin_edges = np.histogram(hist_data_a, bins)
bin_centers = bin_edges[:-1] + bin_width/2
ax.bar(bin_centers, counts, width=np.diff(bin_edges), color='navy', alpha=0.5, label='tone A')

counts, bin_edges = np.histogram(hist_data_b, bins)
bin_centers = bin_edges[:-1] + bin_width/2
ax.bar(bin_centers, counts, width=np.diff(bin_edges), color='lightseagreen', alpha=0.5, label='tone B')
ax.set_xlabel('cell spike count')
ax.set_ylabel('frequency')
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig(outpath + figID + '.pdf', bbox_inches='tight', pad_inches=0, transparent=True)

