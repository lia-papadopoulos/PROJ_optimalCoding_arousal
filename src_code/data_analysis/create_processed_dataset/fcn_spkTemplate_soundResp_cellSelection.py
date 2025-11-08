
'''
function to select good units based on behavior of spike template amplitude from kilosort
'''

#%%

import numpy as np
import pandas as pd
import scipy.stats as spstats
import scipy.signal as spsig
import sys
import h5py
from scipy.io import loadmat


def fcn_spkTemplate_soundResp_cellSelection(session_name, params, raw_data_path, processed_data_path):

    # unpack parameters
    windowSize = params['windowSize']
    fracData_affected = params['fracData_affected']
    ratioCut = params['ratioCut']
    distanceCut = params['distanceCut']
    noiseFloor_cut = params['noiseFloor_cut']  
    peakLocation_cut = params['peakLocation_cut']
    rateThresh = params['rateThresh']
    cut_type = params['cut_type']
    
    
    # load in the raw data
    spike_clusters = np.load(raw_data_path + 'spike_clusters.npy')
    spike_times = np.load(raw_data_path + 'spike_times.npy')
    timestamps = np.load(raw_data_path + 'timestamps.npy')
    amplitudes = np.load(raw_data_path + 'amplitudes.npy')
    cluster_info = pd.read_csv(raw_data_path + 'cluster_info.tsv', sep = '\t')
    
    
    # load in the preprocessed data
    f = h5py.File(processed_data_path + session_name + '.mat','r')
    
    # extract data
    time_stamp = f['time_stamp'][0]
    nCells = f['spk_Good_Aud'].shape[1]
    good_cells = f['clstGAudID'][0]
        
    # cell spike times
    cell_spk_times = np.zeros(nCells,dtype='object')
    for i in range(0,nCells):
        spkTimes_ref = f['spk_Good_Aud'][0,i]
        cell_spk_times[i] = f[spkTimes_ref][0]
    
    # close file
    f.close()
    
    # responsive cells
    
    # unpack parameters
    pval_soundResp = params['pval_soundResp']
    psth_path = params['psth_path']
    nSig_tones_thresh = params['nSig_tones_thresh']

    
    # psth tone responsiveness
    psth_data = loadmat(( ('%spsth_allTrials_%s_windLength0.100s_no_cellSelection.mat') % (psth_path, session_name) ), simplify_cells=True)
    pval = psth_data['psth_pval_corrected']
    min_pval = np.min(pval, axis=1)
    min_pval_thresh = min_pval < pval_soundResp
    nSig_tones = np.sum(min_pval_thresh, 1)
    
    responsiveCells = np.nonzero(nSig_tones >= nSig_tones_thresh)[0]
        

    # get good clusters
    
    cluster_ids = cluster_info['cluster_id'].to_numpy()
    cluster_label = cluster_info['group'].to_numpy()
    good_clusters = cluster_ids[cluster_label == 'good']
        
    # a couple manual hacks for some of the sessions in order to match good clusters from raw data to good cells in data from su
    if session_name == 'LA11_session1':
        
        good_clusters = good_cells.copy()
        
    if session_name == 'LA9_session1':
        
        cluster_depth = cluster_info['depth'].to_numpy()
        
        depth_range = np.array([0, 1320])
        good_depth = cluster_ids[np.nonzero( (cluster_depth >= depth_range[0]) & (cluster_depth <= depth_range[1]) )]
        good_clusters = np.intersect1d(good_clusters, good_depth)

        
    #%% identify cells that have extremely low firing rate
    
    # compute average firing rate of each cell across experiment
    avgRate = np.zeros(nCells)
    total_length = np.max(time_stamp) - np.min(time_stamp)
    
    for i in range(0, nCells):
    
        avgRate[i] = np.size(cell_spk_times[i])/total_length
    
    lowRate_cells = np.nonzero(avgRate <= rateThresh)[0]
    
    
    #%% identify cells to remove from analysis
    
    # sampling rate
    sampling_rate = 1/np.rint(1/np.mean(np.diff(time_stamp)))
            
    # number of samples in each window
    windowSize_samples = windowSize/sampling_rate
        
    # the number of bins that would correspond to fracData_affected
    nBins_fracData_affected = np.ones( len(good_clusters) )*np.nan
    
    # for determining clusters that have two peaks in the same time window
    peakRatio = np.ones( len(good_clusters), dtype='object' )*np.nan
    min_peakRatio = np.ones( len(good_clusters) )*np.nan
    pctChange_peakRatio_location = np.ones( len(good_clusters), dtype='object' )*np.nan
    max_pctChange_peakRatio_location = np.ones( len(good_clusters) )*np.nan
    removeClusters_ratioCut = np.array([])
    removeClusters_twoPeaks = np.array([])
    
    # for determining clusters that drift around noise floor
    max_percentDiff_peak_location = np.ones( len(good_clusters) )*np.nan
    peak_location = np.ones([len(good_clusters)], dtype='object')*np.nan
    mode_noiseFloor_percentDiff_cluster_all = np.array([])
    mode_noiseFloor_percentDifference = np.ones( len(good_clusters), dtype='object' )*np.nan
    removeClusters_noiseFloor_drift = np.array([])
    
    # for determining cells that have significant data around the noise floor, even if relatively constant amplitude
    noiseFloor_clusters = np.array([])
    
    
    # loop over all good clusters
    for indCluster in range(0, len(good_clusters)):
        
        print(indCluster)
    
        # get cluster information
        cluster = good_clusters[indCluster]
        ind_cluster_spikes = np.nonzero(spike_clusters == cluster)[0]
        amp = amplitudes[ind_cluster_spikes]
        ind_tStamp = spike_times[ind_cluster_spikes]
        tStamp = timestamps[ind_tStamp]
        bins = np.arange(np.min(tStamp), np.max(tStamp), windowSize_samples)
        bin_inds = np.digitize(tStamp, bins)
    
        # estimate noise floor as minimum value of amplitude
        noiseFloor = np.min(amp)
        
        # number of bins corresponding to fracData_affected
        nBins_fracData_affected[indCluster] = np.floor(fracData_affected*len(bins)).astype(int)
    
        # initialize arrays to hold info about ratio of peaks, distance between peaks
        peakRatio_cluster = np.ones(np.size(bins))*np.nan
        pctChange_peakRatio_location_cluster = np.ones(np.size(bins))*np.nan
    
        # intialize array to hold percent difference between noise floor and distribution mode
        mode_noiseFloor_percentDifference_cluster = np.ones(np.size(bins))*np.inf
    
        # initialize array to hold location of highest peak in each bin
        peak_location_cluster = np.zeros(np.size(bins))*np.nan
    
        
        # determine clusters that have substantial data around the noise floor, even if constant amplitude
    
        # pdf of all data
        kernel=spstats.gaussian_kde(amp[:,0])
        amp_vals_sample = np.arange(0, np.max(amp)+10, 0.1)
        pdf = kernel(amp_vals_sample)
    
        # indices of distribution peaks
        indPeaks, _ = spsig.find_peaks(pdf) 
    
        # peak location and peak height
        peakVals = pdf[indPeaks]
        peakLocs = amp_vals_sample[indPeaks]
    
        # highest peak
        highestPeak = np.nanmax(peakVals)
    
        # loop over peaks and compute ratio against max peak and distance from noise floor
        noiseFloor_peak = np.zeros(len(indPeaks))
    
        for indPeak in range(0, len(indPeaks)):
    
            maxPeak_ratio = highestPeak/peakVals[indPeak]
            noiseFloor_pctDiff = 100*np.abs( peakLocs[indPeak] - noiseFloor )/( (peakLocs[indPeak] + noiseFloor)/2 )
    
            if ( (maxPeak_ratio <= ratioCut) and (noiseFloor_pctDiff <= noiseFloor_cut)):
    
                noiseFloor_peak[indPeak] = 1
    
        # does at least one peak exist near noise floor?
        if np.any(noiseFloor_peak == 1):
    
            noiseFloor_clusters = np.append(noiseFloor_clusters, indCluster)
    
    
        # sliding window analysis
        for indBin in range(0, np.size(bins)):
    
            indData_in_bin = np.nonzero(bin_inds == indBin+1)[0]
    
            if np.size(indData_in_bin) > 1:
    
                # pdf of data in this bin
                amp_in_bin = amp[indData_in_bin,0]
                kernel=spstats.gaussian_kde(amp_in_bin)
                amp_vals_sample = np.arange(0, np.max(amp_in_bin)+10, 0.1)
                pdf = kernel(amp_vals_sample)
                indPeaks, _ = spsig.find_peaks(pdf)
    
                # values of peaks
                peakVals = pdf[indPeaks]
                peakLocs = amp_vals_sample[indPeaks]
    
                # highest mode of distribution
                ind_highestMode = np.argmax(peakVals)
                loc_highestMode = amp_vals_sample[indPeaks[ind_highestMode]]
                pdf_highestMode = peakVals[ind_highestMode]
    
    
                # make sure largest peak is non nan
                if np.isnan(pdf_highestMode):
                    sys.exit('nan value error')
    
    
                # save location of highest peak
                peak_location_cluster[indBin] = loc_highestMode
    
    
                # percent difference between highest mode location and noise floor location
                mode_noiseFloor_percentDifference_cluster[indBin] = 100*np.abs(noiseFloor - loc_highestMode) / ( (noiseFloor + loc_highestMode)/2 )
    
    
                # check for mulitple peaks in the same bin
                if len(indPeaks) >= 2:
    
                    sorted_peaks = np.flip(np.sort(peakVals))
                    sorted_peaks = sorted_peaks[~np.isnan(sorted_peaks)]
    
                    sorted_indPeaks = np.array([])
                    for i in range(0, len(indPeaks)):
    
                        sorted_indPeaks = np.append( sorted_indPeaks, np.nonzero(pdf == sorted_peaks[i])[0] )
    
                    sorted_indPeaks = sorted_indPeaks.astype(int)
                    sorted_peak_locs = amp_vals_sample[sorted_indPeaks]
                    peakRatio_cluster[indBin] = sorted_peaks[0]/sorted_peaks[1]
                    pctChange_peakRatio_location_cluster[indBin] = 100*np.abs(sorted_peak_locs[0]-sorted_peak_locs[1])/( (sorted_peak_locs[0] +  sorted_peak_locs[1])/2 )
    
    
        # save info for this session
        peak_location[indCluster] = peak_location_cluster
        mode_noiseFloor_percentDifference[indCluster] = mode_noiseFloor_percentDifference_cluster
        mode_noiseFloor_percentDiff_cluster_all = np.append(mode_noiseFloor_percentDiff_cluster_all, mode_noiseFloor_percentDifference_cluster)
        peakRatio[indCluster] = peakRatio_cluster
        min_peakRatio[indCluster] = np.nanmin(peakRatio_cluster)
        pctChange_peakRatio_location[indCluster] = pctChange_peakRatio_location_cluster
    
    
        # apply cuts
        
        ############ does this cluster have two or more peaks in the same window ? ###########
    
        # does this cluster pass the peak ratio cut?
        if min_peakRatio[indCluster] <= ratioCut:
    
            removeClusters_ratioCut = np.append(removeClusters_ratioCut, indCluster)
            max_pctChange_peakRatio_location[indCluster] = np.nanmax(  pctChange_peakRatio_location_cluster[peakRatio_cluster <= ratioCut]   )
    
        # bins that fail both cuts
        bins_failCuts = np.nonzero( (peakRatio_cluster <= ratioCut )  & (pctChange_peakRatio_location_cluster >= distanceCut) )[0]
    
        # if number of failed bins is large enough fraction of the data, then we want to remove this cluster
        if np.size(bins_failCuts) >=  nBins_fracData_affected[indCluster]:
    
            removeClusters_twoPeaks = np.append(removeClusters_twoPeaks, indCluster)
        
        
        ############ does this cluster drift towards/away from noise floor? ###########
    
        # bins that are near the noise floor
        cut_mode_noiseFloor_percentDifference_cluster = mode_noiseFloor_percentDifference[indCluster] <= noiseFloor_cut
    
        # noise floor bins
        noiseFloor_bins = np.nonzero(cut_mode_noiseFloor_percentDifference_cluster == 1)[0]
        bulk_bins = np.nonzero(cut_mode_noiseFloor_percentDifference_cluster == 0)[0]
    
        # percent difference between largest and smallest peak location
        max_percentDiff_peak_location[indCluster] = 100*(np.nanmax(peak_location_cluster) - np.nanmin(peak_location_cluster))/( (np.nanmax(peak_location_cluster) + np.nanmin(peak_location_cluster))/2 )
    
        # if a substantial fraction (but not all) of the data is at the noise floor, then we want to remove this cell
        if ( ( (np.size(noiseFloor_bins) >= nBins_fracData_affected[indCluster]) and (np.size(bulk_bins) >=1) and (max_percentDiff_peak_location[indCluster] >= peakLocation_cut) )  or \
             ( (np.size(noiseFloor_bins) >= 1) and (np.size(bulk_bins) >=nBins_fracData_affected[indCluster]) and (max_percentDiff_peak_location[indCluster] >= peakLocation_cut) ) ):
    
            removeClusters_noiseFloor_drift = np.append(removeClusters_noiseFloor_drift, indCluster)
            
        
        ############ does this cluster have any data at all around noise floor? ###########
        if (np.size(noiseFloor_bins) >= nBins_fracData_affected[indCluster]) :
            
            noiseFloor_clusters = np.append(noiseFloor_clusters, indCluster)
            
            
    # only keep unique noise floor clusters
    noiseFloor_clusters = np.unique(noiseFloor_clusters)
    
    
    #%% combine results from all cuts to get cells that should be removed and cells that we should keep
    
    if cut_type == 'version1':
    
        removeClusters_all = np.concatenate((lowRate_cells, removeClusters_twoPeaks, removeClusters_noiseFloor_drift))
        removeClusters_all = np.unique(removeClusters_all)
        goodClusters_all = np.rint(np.setdiff1d( np.arange(0,len(good_clusters)), removeClusters_all)).astype(int)
        
        print(lowRate_cells)
        print(removeClusters_twoPeaks)
        print(removeClusters_noiseFloor_drift)
    
    elif cut_type == 'version2':
        
        removeClusters_all = np.concatenate((lowRate_cells, removeClusters_twoPeaks, removeClusters_noiseFloor_drift, noiseFloor_clusters))
        removeClusters_all = np.unique(removeClusters_all)
        goodClusters_all = np.rint(np.setdiff1d( np.arange(0,len(good_clusters)), removeClusters_all)).astype(int)
    
    else:
        
        sys.exit('issue with cut type')
    
    
    all_good_units = np.intersect1d(goodClusters_all, responsiveCells)
        
    #%% save only spike times of good units
    
    f = h5py.File(processed_data_path + session_name + '.mat','r')
    
    # extract data
    nCells = f['spk_Good_Aud'].shape[1]
        
    # cell spike times
    cell_spk_times_good = np.zeros(nCells,dtype='object')
    for i in range(0,nCells):
        spkTimes_ref = f['spk_Good_Aud'][0,i]
        cell_spk_times_good[i] = f[spkTimes_ref][0]
    
    # close file
    f.close()
    
    # spike times of significant cells only
    cell_spk_times_good = cell_spk_times_good[all_good_units]
    
            
    #%% output results as dictionary
    
    output_params = {}
    results = {}
    
    output_params['session_name'] = session_name
    output_params['raw_data_path'] = raw_data_path
    output_params['preprocessed_data_path_Su'] = processed_data_path
    output_params['windowSize'] = windowSize
    output_params['fracData_affected'] = fracData_affected
    output_params['ratioCut'] = ratioCut
    output_params['distanceCut'] = distanceCut
    output_params['noiseFloor_cut'] = noiseFloor_cut    
    output_params['peakLocation_cut'] = peakLocation_cut
    output_params['rateThresh'] = rateThresh
    output_params['cut_type'] = cut_type

    results['good_units'] = all_good_units
    results['cell_spk_times'] = cell_spk_times_good
    
    return output_params, results
