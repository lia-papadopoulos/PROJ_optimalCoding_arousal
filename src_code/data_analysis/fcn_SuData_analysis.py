
import numpy as np
import scipy.stats


#%% find significantly responding cells based on comparing prestim and post stim window

def fcn_significant_preStim_vs_postStim(psth_data, sig_level):
    
    pval = psth_data['pval_preStim_vs_postStim']
    trialAvg_gain_alt = psth_data['trialAvg_gain_alt']
    nFreq = psth_data['nFreq']
    tInd_postStim = psth_data['postStim_tInd']
    
    nCells = np.size(pval, 0)
    
    sigCells = np.ones((nFreq), dtype='object')*np.nan
    sigCells_posResponse = np.ones((nFreq), dtype='object')*np.nan
    sigCells_negResponse = np.ones((nFreq), dtype='object')*np.nan
    
    allSigCells = np.array([])
    allSigCells_posResponse = np.array([])
    allSigCells_negResponse = np.array([])
    
    binarizedResp_vector = np.zeros((nCells, nFreq))
    resp_vector = np.zeros((nCells, nFreq))

    for indFreq in range(0, nFreq):
        
        sigCells[indFreq] = np.array([])
        sigCells_posResponse[indFreq] = np.array([])
        sigCells_negResponse[indFreq] = np.array([])
        
        for indCell in range(0, nCells):
            
            respAmp = trialAvg_gain_alt[indCell, tInd_postStim, indFreq].copy()
            sigResp = pval[indCell, indFreq] < sig_level
            
            resp_vector[indCell, indFreq] = respAmp
            
            if sigResp:
                
                sigCells[indFreq] = np.append(sigCells[indFreq], indCell)
                allSigCells = np.append(allSigCells, indCell)
                
                if respAmp > 0:
                    
                    sigCells_posResponse[indFreq] = np.append(sigCells_posResponse[indFreq], indCell)
                    allSigCells_posResponse = np.append(allSigCells_posResponse, indCell)
                    
                    binarizedResp_vector[indCell, indFreq] = 1

                    
                elif respAmp < 0:
                    
                    sigCells_negResponse[indFreq] = np.append(sigCells_negResponse[indFreq], indCell)
                    allSigCells_negResponse = np.append(allSigCells_negResponse, indCell)
                    
                    binarizedResp_vector[indCell, indFreq] = -1
                    
            else:
                
                binarizedResp_vector[indCell, indFreq] = 0
                
        sigCells[indFreq] = sigCells[indFreq].astype(int)
                

                    
    allSigCells = np.unique(allSigCells)
    allSigCells_posResponse = np.unique(allSigCells_posResponse)
    allSigCells_negResponse = np.unique(allSigCells_negResponse)
                    
    output_dict = {}
    output_dict['nCells'] = nCells
    output_dict['binarizedResp_vector'] = binarizedResp_vector
    output_dict['resp_vector'] = resp_vector
    output_dict['sigCells'] = sigCells
    output_dict['sigCells_posResponse'] = sigCells_posResponse
    output_dict['sigCells_negResponse'] = sigCells_negResponse
    output_dict['allSigCells'] = allSigCells
    output_dict['allSigCells_posResponse'] = allSigCells_posResponse
    output_dict['allSigCells_negResponse'] = allSigCells_negResponse

                    
    return output_dict


#%% alternate way of getting significant responses

def fcn_significant_preStim_vs_postStim_alt(psth_data, sig_level):
    
    pval = psth_data['psth_pval_corrected']
    timeInd_peakResponse_in_stimWindow = psth_data['timeInd_peakResponse_in_stimWindow']
    nCells = np.size(pval, 0)
    nFreq = psth_data['nFreq']
    time = psth_data['t_window']
    
    nCells = np.size(pval, 0)
    
    sigCells_eachFreq = np.ones((nFreq), dtype='object')*np.nan
    sigTime_eachFreq = np.ones((nCells, nFreq))*np.nan
    allSigCells = np.array([])

    for indFreq in range(0, nFreq):
        
        sigCells_eachFreq[indFreq] = np.array([])
        
        for indCell in range(0, nCells):
            
            sigResp = np.min(pval[indCell, :, indFreq]) < sig_level
            ind_sigTime = timeInd_peakResponse_in_stimWindow[indCell, indFreq].astype(int)
            sigTime_eachFreq[indCell, indFreq] = time[ind_sigTime]
            
            if sigResp:
                
                sigCells_eachFreq[indFreq] = np.append(sigCells_eachFreq[indFreq], indCell)
                allSigCells = np.append(allSigCells, indCell)
                                
        sigCells_eachFreq[indFreq] = sigCells_eachFreq[indFreq].astype(int)
                

    allSigCells = np.unique(allSigCells)
                    
    output_dict = {}
    output_dict['nCells'] = nCells
    output_dict['sigCells'] = sigCells_eachFreq
    output_dict['sigTime'] = sigTime_eachFreq
    output_dict['allSigCells'] = allSigCells
    
    return output_dict


#%% compute average stimulus response to each tone

def fcn_avg_resp_eachTone(t_gain, trialAvg_gain, stim_window):
    
    
    stimInds = np.nonzero( (t_gain <= stim_window[1]) & (t_gain >= stim_window[0]) )[0]
    
    avg_resp_eachTone = np.mean(trialAvg_gain[:, stimInds, :], axis=1) # cells x freq
    
    return avg_resp_eachTone


#%% compute signal correlation between each pair of cells based on average stimulus response

def fcn_signalCorr_avgResp(avg_resp_eachTone, corr_type = 'spearman'):

    nCells = np.size(avg_resp_eachTone, 0)
    
    sig_corr = np.zeros((nCells, nCells))
    
    for indCell_i in range(0, nCells):
        for indCell_j in range(0, nCells):
            
            if corr_type == 'spearman':
                sig_corr[indCell_i, indCell_j], _ = scipy.stats.spearmanr(avg_resp_eachTone[indCell_i,:], avg_resp_eachTone[indCell_j,:])
            else:
                sig_corr[indCell_i, indCell_j], _ = scipy.stats.pearsonr(avg_resp_eachTone[indCell_i,:], avg_resp_eachTone[indCell_j,:])
                
    return sig_corr
        

#%% similarity between two response vectors

# responseVectors:  (nCells, nStim)

def fcn_cosineSim_respVectors(responseVectors):
    
    nCells = np.size(responseVectors,0)
    
    cosineSim = np.zeros((nCells,nCells))
    
    for i in range(0,nCells):
        
        for j in range(0,nCells):
            
            cosineSim[i,j] = np.dot(responseVectors[i,:],responseVectors[j,:])/(np.linalg.norm(responseVectors[i,:])*np.linalg.norm(responseVectors[j,:]))


    return cosineSim


#%% compute session average of a normalized quantity vs pupil bin

'''
pupilBin_centers: (nSessions, nPupils) of pupilBin centers for each session
minPupil : scalar indicating minimum possible pupil value
maxPupil: scalar indicating maximum possible pupil value
binSize: size of pupil bins for combining data across sessions
quantity_toBin: (nSessions, nPupils) array of the quantity we want to bin vs pupil size across sessions
'''
    

def fcn_sessionAvg_quantity_vs_pupilSize_bins_alt(pupilBin_centers, minPupil, maxPupil, binSize, quantity_toBin, return_pupilSize_binnedData = False):
    
    

    pupilSize_bins = np.arange(minPupil, maxPupil + binSize, binSize)
    pupilSize_binCenters = np.arange(binSize/2, maxPupil, binSize)
    
    nSessions = np.shape(pupilBin_centers)[0]
    
    data_in_pupilBins_allSessions = np.ones((len(pupilSize_bins)-1), dtype='object')*np.nan
    pupilSize_data_in_pupilBins_allSessions = np.ones((len(pupilSize_bins)-1), dtype='object')*np.nan
    
    
    for binInd in range(0, len(pupilSize_bins)-1):
        data_in_pupilBins_allSessions[binInd] = np.array([])
        pupilSize_data_in_pupilBins_allSessions[binInd] = np.array([])
        
    for indSession in range(0, nSessions):
    
        # bin indices corresponding to each pupil size of this session
        pupilSize_binInds = np.digitize(pupilBin_centers[indSession, :], pupilSize_bins) - 1

    
        # loop over bins
        for binInd in range(0, len(pupilSize_bins)-1):
            
            
            # indices of all data in this bin
            data_in_bin = np.nonzero(pupilSize_binInds == binInd)[0]
    
            # modulation indices vs pupil
            avg_inBin = np.mean(quantity_toBin[indSession, data_in_bin])
            
            # pupil size of data in this bin
            avg_pupil_inBin = np.mean(pupilBin_centers[indSession, data_in_bin])

            # save data in this bin
            data_in_pupilBins_allSessions[binInd] = np.append(data_in_pupilBins_allSessions[binInd], avg_inBin)
            pupilSize_data_in_pupilBins_allSessions[binInd] = np.append(pupilSize_data_in_pupilBins_allSessions[binInd], avg_pupil_inBin)
            
            
    # average data in bin across sessions
    avg_quantity_toBin = np.zeros(len(pupilSize_bins)-1)
    std_quantity_toBin = np.zeros(len(pupilSize_bins)-1)
    sem_quantity_toBin = np.zeros(len(pupilSize_bins)-1)
    
    for binInd in range(0, len(pupilSize_bins)-1):

        avg_quantity_toBin[binInd] = np.nanmean(data_in_pupilBins_allSessions[binInd]) 
        std_quantity_toBin[binInd] = np.nanstd(data_in_pupilBins_allSessions[binInd]) 
        sem_quantity_toBin[binInd] = scipy.stats.sem(data_in_pupilBins_allSessions[binInd], nan_policy='omit') 

    if return_pupilSize_binnedData:

        return pupilSize_binCenters, avg_quantity_toBin, std_quantity_toBin, sem_quantity_toBin, data_in_pupilBins_allSessions, pupilSize_data_in_pupilBins_allSessions    
    
    else:
        
        return pupilSize_binCenters, avg_quantity_toBin, std_quantity_toBin, sem_quantity_toBin, data_in_pupilBins_allSessions    
        
    
    
#%%

'''
quantity_toBin:     object array w/ dimensions (nSessions, nPupils)
                    quantity_toBin[i,j] has size (nCells)
'''

def fcn_cellAvg_sessionAvg_quantity_vs_pupilSize_bins_alt(pupilBin_centers, minPupil, maxPupil, binSize, quantity_toBin, return_pupilSize_binnedData = False):
    
    

    pupilSize_bins = np.arange(minPupil, maxPupil + binSize, binSize)
    pupilSize_binCenters = np.arange(binSize/2, maxPupil, binSize)
    
    nSessions = np.shape(pupilBin_centers)[0]
    
    data_in_pupilBins_allSessions = np.ones((len(pupilSize_bins)-1), dtype='object')*np.nan
    pupilSize_data_in_pupilBins_allSessions = np.ones((len(pupilSize_bins)-1), dtype='object')*np.nan
    
    # loop over pupil size bins and initialize arrays
    for binInd in range(0, len(pupilSize_bins)-1):
        data_in_pupilBins_allSessions[binInd] = np.array([])
        pupilSize_data_in_pupilBins_allSessions[binInd] = np.array([])
        
    # loop over sessions
    for indSession in range(0, nSessions):
    
        # pupil size bin indices corresponding to each pupil data point in this session
        pupilSize_binInds = np.digitize(pupilBin_centers[indSession, :], pupilSize_bins) - 1

    
        # loop over pupil size bins
        for binInd in range(0, len(pupilSize_bins)-1):
            
            
            # indices of all pupil size percentile bins in this pupil size bin
            pupilPercentiles_inBin = np.nonzero(pupilSize_binInds == binInd)[0]
    
            # average of all data in bin
            
            # number of data points (pupil size percentile bins) in this pupil bin
            n_pupilPercentiles_inBin = np.size(pupilPercentiles_inBin)
            n_Cells = np.size(quantity_toBin[indSession, 0])
            
            data_inBin = np.ones((n_pupilPercentiles_inBin, n_Cells))*np.nan
            for pupilPercentile_count, pupilPercentile_indx in enumerate(pupilPercentiles_inBin):
                data_inBin[pupilPercentile_count, :] = quantity_toBin[indSession, pupilPercentile_indx][:].copy()
                
            # for each cell, average quantity across all pupil percentiles of this session that fall in this pupil size bin
            # size of this = nCells
            avg_inBin = np.nanmean(data_inBin,0)
            
            # average of pupil decile centers in this pupil size bin
            avg_pupilSize_data_inBin = np.mean(pupilBin_centers[indSession, pupilPercentiles_inBin])

            # save data in this bin
            data_in_pupilBins_allSessions[binInd] = np.append(data_in_pupilBins_allSessions[binInd], avg_inBin)
            pupilSize_data_in_pupilBins_allSessions[binInd] = np.append(pupilSize_data_in_pupilBins_allSessions[binInd], avg_pupilSize_data_inBin)
            
            
    # average data in bin across all cells and sessions
    avg_data_eachBin = np.zeros(len(pupilSize_bins)-1)
    std_data_eachBin = np.zeros(len(pupilSize_bins)-1)
    sem_data_eachBin = np.zeros(len(pupilSize_bins)-1)
    
    for binInd in range(0, len(pupilSize_bins)-1):

        avg_data_eachBin[binInd] = np.nanmean(data_in_pupilBins_allSessions[binInd]) 
        std_data_eachBin[binInd] = np.nanstd(data_in_pupilBins_allSessions[binInd]) 
        sem_data_eachBin[binInd] = scipy.stats.sem(data_in_pupilBins_allSessions[binInd], nan_policy='omit') 

    if return_pupilSize_binnedData:

        return pupilSize_binCenters, avg_data_eachBin, std_data_eachBin, sem_data_eachBin, data_in_pupilBins_allSessions, pupilSize_data_in_pupilBins_allSessions    
    
    else:
        
        return pupilSize_binCenters, avg_data_eachBin, std_data_eachBin, sem_data_eachBin, data_in_pupilBins_allSessions    
        
    

#%% cell and session-averaged vector quantity in different pupil bins

'''
singleCell_spectra:        (nSessions, n_pupil_percentiles) object array
singleCell_spectra[i,j]:   (nCells, nFreq) array
'''

def cellAvg_sessionAvg_spectra_vs_pupilDiamBins(singleCell_spectra, pupilBin_centers, bin_lower, bin_upper):

    # number of sessions
    nSessions = np.size(singleCell_spectra, 0)

    # number of pupil bins
    n_bins = len(bin_lower)
    
    # should be the same for all sessions and pupil diameters
    nFreq = np.size(singleCell_spectra[0,0], 1)
    
    grandAvg_binned_spec = np.ones((n_bins,nFreq))*np.nan
    grandStd_binned_spec = np.ones((n_bins,nFreq))*np.nan
    grandSem_binned_spec = np.ones((n_bins,nFreq))*np.nan
    
    
    for iBin in range(0, n_bins):
    
        lowLim = bin_lower[iBin]
        highLim = bin_upper[iBin]
        
        spec_eachBin = np.empty((0, nFreq))
    
        for indSession in range(0, nSessions):
    
            nCells = np.size(singleCell_spectra[indSession,0],0)
            
            # all percentile bins in this pupil bin
            data_in_bin = np.nonzero( (pupilBin_centers[indSession, :] >= lowLim) & (pupilBin_centers[indSession, :] < highLim) )[0]
            if np.size(data_in_bin) == 0:
                continue
    
            # combine across all percentile bins in this pupil bin
            spec_eachSession_combined = np.ones((nCells, len(data_in_bin), nFreq))*np.nan
    
            for indData, valData in enumerate(data_in_bin):
    
                spec_eachSession_combined[:, indData, :] = singleCell_spectra[indSession, valData].copy()
    
            # average across all percentile bins in this pupil bin
            avg_powSpec_session = np.mean(spec_eachSession_combined, 1) # cells, frequency
            
            # combine cells across all sessions
            spec_eachBin = np.vstack((spec_eachBin, avg_powSpec_session))
    
        # average over cells from all sessions
        grandAvg_binned_spec[iBin,:] = np.nanmean(spec_eachBin, 0)
        grandStd_binned_spec[iBin,:] = np.nanstd(spec_eachBin, 0)
        grandSem_binned_spec[iBin,:] = scipy.stats.sem(spec_eachBin, 0, nan_policy='omit')


    return grandAvg_binned_spec, grandStd_binned_spec, grandSem_binned_spec


    

