
"""
examine correlations between firing rate and pupil size
"""

#%%
import sys        
import numpy as np
import matplotlib
matplotlib.use('agg')
import scipy.stats
from scipy.io import savemat
import argparse

import pupil_rate_correlation_settings as settings

# add paths to functions
sys.path.append(settings.func_path1)       
sys.path.append(settings.func_path2)

from fcn_processedh5data_to_dict import fcn_processedh5data_to_dict
from fcn_SuData import fcn_spikeTimes_trials_cells_spont
from fcn_SuData import fcn_makeTrials_spont
from fcn_SuData import fcn_compute_spikeCnts_inTrials
from fcn_SuData import fcn_compute_avgRunSpeed_inTrials
from fcn_SuData import fcn_compute_avgPupilSize_inTrials


#%% settings

data_path = settings.data_path
outpath = settings.analyzed_data_path
fig_outpath = settings.fig_outpath
window_length = settings.window_length
inter_window_interval = settings.inter_window_interval
stim_duration = settings.stim_duration
percentileBin_size = settings.percentileBin_size
rateDrift_cellSelection = settings.rateDrift_cellSelection


#%% argparse inputs

# argparser
parser = argparse.ArgumentParser() 

# session name
parser.add_argument('-session_name', '--session_name', type=str, default = '')
    
# arguments of parser
args = parser.parse_args()

# argparse inputs
session_name = args.session_name


#%% GET DATA

data_name = '' + '_rateDrift_cellSelection'*rateDrift_cellSelection

session_info = fcn_processedh5data_to_dict(session_name, data_path, fname_end = data_name)


#%% EXTRACT RELEVANT DATA

spont_blocks = session_info['spont_blocks']
nCells = session_info['nCells']

session_info['spontBlock_start'] = spont_blocks[0,:].copy()
session_info['spontBlock_end'] = spont_blocks[1,:].copy()
session_info['n_spontBlocks'] = np.size(session_info['spontBlock_start'])
                                        
                                        
#%% SPLIT SPONTANEOUS BLOCKS INTO ANALYSIS WINDOWS

session_info = fcn_makeTrials_spont(session_info, window_length, inter_window_interval)

trial_start = session_info['trial_start'].copy()
trial_end = session_info['trial_end'].copy()

print('session trials computed')


# compute spike times of each cell in every trial
session_info = fcn_spikeTimes_trials_cells_spont(session_info)

# compute spike counts of all cells in each trial
session_info = fcn_compute_spikeCnts_inTrials(session_info)

# compute rates of all cells in each trial
trial_rates = session_info['spkCounts_trials_cells']/window_length


#%% AVERAGE RUNNING SPEED IN EACH WINDOW

session_info = fcn_compute_avgRunSpeed_inTrials(session_info, trial_start, trial_end)
avg_runSpeed = session_info['avg_runSpeed'].copy()

#%% AVERAGE PUPIL SIZE IN EACH WINDOW

avg_pupilSize = fcn_compute_avgPupilSize_inTrials(session_info, trial_start, trial_end)
session_info['avg_pupilSize'] = avg_pupilSize


#%% CORRELATION BETWEEN RUNNING OR AND PUPIL SIZE AND CELL RATE

corr_pupil_rates = np.ones((nCells, 2))*np.nan
corr_run_rates = np.ones((nCells, 2))*np.nan

for cellInd in range(0, nCells):
    
    nonNan = ~np.isnan(avg_pupilSize)
    
    rspearman, p = scipy.stats.spearmanr(trial_rates[nonNan, cellInd], avg_pupilSize[nonNan])
    corr_pupil_rates[cellInd, 0] = rspearman
    corr_pupil_rates[cellInd, 1] = p

    rspearman, p = scipy.stats.spearmanr(trial_rates[nonNan, cellInd], avg_runSpeed[nonNan])
    corr_run_rates[cellInd,0] = rspearman
    corr_run_rates[cellInd,1] = p
    
    
#%% BIN DATA BASED ON SPEED AND PUPIL PERCENTILE

nBins = int(1/percentileBin_size)

avg_pupilSize_inPercentiles = np.zeros(nBins)
pupil_binEdges = np.zeros(nBins+1)
pupil_binCenters = np.zeros(nBins)
run_binEdges = np.zeros(nBins+1)
run_binCenters = np.zeros(nBins)

# get perentile bins
for indBin in range(0, nBins):
    
    pct_low = int(indBin*percentileBin_size*100)
    pct_high = pct_low + percentileBin_size*100
        
    pupil_binEdges[indBin] = np.nanpercentile(avg_pupilSize, pct_low)
    pupil_binEdges[indBin+1] = np.nanpercentile(avg_pupilSize, pct_high)
    pupil_binCenters[indBin] = np.mean([pupil_binEdges[indBin], pupil_binEdges[indBin+1]])
    
    run_binEdges[indBin] = np.nanpercentile(avg_runSpeed, pct_low)
    run_binEdges[indBin+1] = np.nanpercentile(avg_runSpeed, pct_high)  
    run_binCenters[indBin] = np.mean([run_binEdges[indBin], run_binEdges[indBin+1]])

    
# bin data    
avgPupil_binID = np.digitize(avg_pupilSize, pupil_binEdges)
avgSpeed_binID = np.digitize(avg_runSpeed, run_binEdges)

# average rate in each pupil & running bin
avgRate_pupilBins = np.zeros((nCells, nBins))
avgRate_runBins = np.zeros((nCells, nBins))


for indBin in range(0, nBins):
    
    data_inBin = np.nonzero(avgPupil_binID == indBin+1)[0]
    avgRate_pupilBins[:, indBin] = np.mean(trial_rates[data_inBin, :], axis=0)
    avg_pupilSize_inPercentiles[indBin] = np.mean(avg_pupilSize[data_inBin])
    
    data_inBin = np.nonzero(avgSpeed_binID == indBin+1)[0]
    avgRate_runBins[:, indBin] = np.mean(trial_rates[data_inBin, :], axis=0)    
    


#%% COMPUTE CORRELATION

corr_pupil_rates_pctileBins = np.zeros((nCells, 2))
corr_run_rates_pctileBins = np.zeros((nCells, 2))

for indCell in range(0, nCells):
    
    rspearman, p = scipy.stats.spearmanr(avgRate_pupilBins[indCell, :], avg_pupilSize_inPercentiles, nan_policy='omit')
    corr_pupil_rates_pctileBins[indCell, 0] = rspearman
    corr_pupil_rates_pctileBins[indCell, 1] = p

    rspearman, p = scipy.stats.spearmanr(avgRate_runBins[indCell, :], run_binCenters, nan_policy='omit')
    corr_run_rates_pctileBins[indCell, 0] = rspearman
    corr_run_rates_pctileBins[indCell, 1] = p


#%% SAVE THE DATA

params = {}
results = {}

params['session_name'] = session_name
params['window_length'] = window_length
params['inter_window_interval'] = inter_window_interval
params['stim_duration'] = stim_duration
params['percentileBin_size'] = percentileBin_size
params['rateDrift_cellSelection'] = rateDrift_cellSelection

results['params'] = params
results['avg_pupilSize_inPercentiles'] = avg_pupilSize_inPercentiles
results['pupil_binEdges'] = pupil_binEdges
results['pupil_binCenters'] = pupil_binCenters
results['avgRate_pupilBins'] = avgRate_pupilBins
results['corr_run_rates'] = corr_run_rates
results['corr_pupil_rates'] = corr_pupil_rates
results['corr_run_rates_pctileBins'] = corr_run_rates_pctileBins
results['corr_pupil_rates_pctileBins'] = corr_pupil_rates_pctileBins


savemat('%s%s_pupil_run_rate_correlation_spont_windLength%0.3fs%s.mat' % (outpath, session_name, window_length, data_name), results)


