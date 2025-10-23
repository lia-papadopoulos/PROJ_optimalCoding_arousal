
"""
helper functions for plotting the fano factor
"""

#%% basic imports
import numpy as np
    
#%% compute cells who pass baseline rate threshold

# cells that pass rate cuts when only looking at spontaneous data
def fcn_units_pass_rateCut_allPupils_spont(fano_dict, rate_thresh):
    
    # time window
    window_length = fano_dict['params']['window_length']
    
    # good units
    good_units = fano_dict['params']['good_units'].copy()
    
    # trial average rate
    trialAvg_rate = fano_dict['fanofactor_den'].copy()/window_length
    trialAvg_rate = trialAvg_rate[good_units, :].copy()
            
    # cells that pass rate cut
    units_pass_rateCut = np.nonzero(np.all( trialAvg_rate >= rate_thresh, 1 ))[0]
    
    return units_pass_rateCut


# cells that pass rate cut for each frequency (evoked data)
def fcn_units_pass_rateCut_eachFreq(rates, rate_thresh):
    
    nFreq = np.size(rates, 1)
    
    units_pass_rateCut_eachFreq = np.nan * np.ones((nFreq), dtype='object')
    
    for iFreq in range(0, nFreq):
        
        units_pass_rateCut_eachFreq[iFreq] = np.nonzero(rates[:, iFreq] >= rate_thresh)[0]
    
    
    return units_pass_rateCut_eachFreq


# assumes psth_data only includes good units
def fcn_units_pass_rateCut(psth_data, rate_thresh):
    
    trialAvg_psth = psth_data['trialAvg_psth'].copy() # cells, time, freq
    baseline_window = psth_data['params']['baseline_window']
    t_windows = psth_data['t_window']
    base_tInds = np.nonzero( (t_windows <= baseline_window[1]) & (t_windows > baseline_window[0]) )[0]

    
    freqAvg_trialAvg_psth = np.mean(trialAvg_psth, 2)
    baseAvg_freqAvg_trialAvg_psth = np.mean(freqAvg_trialAvg_psth[:, base_tInds], 1)
    
    units_pass_rateCut = np.nonzero(baseAvg_freqAvg_trialAvg_psth > rate_thresh)[0]
    
    return units_pass_rateCut


def fcn_units_pass_rateCut_allPupils(psthPupil_data, rate_thresh):
    
    
    # unnpack
    trialAvg_psth = psthPupil_data['trialAvg_psth'].copy() # cells, time, freq, pupil
    base_tInds = psthPupil_data['base_tInds'].copy()
    good_units = psthPupil_data['params']['good_units'].copy()
    
    freqAvg_trialAvg_psth = np.mean(trialAvg_psth, 2)
    baseAvg_freqAvg_trialAvg_psth = np.mean(freqAvg_trialAvg_psth[:, base_tInds, :], 1)
    baseAvg_freqAvg_trialAvg_psth = baseAvg_freqAvg_trialAvg_psth[good_units, :].copy()
    
    units_pass_rateCut = np.nonzero(np.all(baseAvg_freqAvg_trialAvg_psth > rate_thresh, 1))[0]

    return units_pass_rateCut


#%% compute significant cells who pass baseline rate threshold

def fcn_sigUnits_pass_rateCut(units_pass_rate_cut, all_sig_units, sig_units_eachFreq, nFreq):

    # significant cells that pass rate cut
    all_sig_units_pass_rateCut = np.intersect1d(all_sig_units, units_pass_rate_cut)
    
    sig_units_eachFreq_pass_rateCut = np.zeros((nFreq), dtype='object')
    
    # significant cells at each frequency
    for indFreq in range(0, nFreq):
        
        sig_units_eachFreq_pass_rateCut[indFreq] = np.intersect1d(sig_units_eachFreq[indFreq], units_pass_rate_cut)
        
    
    return all_sig_units_pass_rateCut, sig_units_eachFreq_pass_rateCut


#%% compute significant cells at each frequency who also pass rate cut at that frequency

def fcn_units_pass_rateCut_sigCut_eachFreq(units_pass_rateCut_eachFreq, sigUnits_eachFreq):
    
    nFreq = np.size(units_pass_rateCut_eachFreq)
    
    sig_units_eachFreq_passCuts = np.nan * np.ones((nFreq), dtype='object')
    all_sig_units_passCuts = np.array([])
    
    for iFreq in range(0, nFreq):
        
        sig_units_eachFreq_passCuts[iFreq] = np.intersect1d(units_pass_rateCut_eachFreq[iFreq], sigUnits_eachFreq[iFreq])
        
        all_sig_units_passCuts = np.append(all_sig_units_passCuts, sig_units_eachFreq_passCuts[iFreq])
        
    all_sig_units_passCuts = np.unique(all_sig_units_passCuts.astype(int)).astype(int)
    
    
    return all_sig_units_passCuts, sig_units_eachFreq_passCuts



#%% units that pass fano factor cut

# fano_data:        (nGoodUnits,)


def fcn_units_pass_FFcut(fano_data, fanoCut):
    
    units_pass_FFcut = np.nonzero(fano_data > fanoCut)[0]
    
    return units_pass_FFcut
    


#%% units that pass significance at each frequency, rate cut, fano factor cut

def fcn_units_pass_sigFreq_rate_FF_cuts(sig_units_eachFreq, units_pass_rate_cut, units_pass_FFcut):
    
    nFreq = np.size(sig_units_eachFreq)
    
    units_pass_cuts = np.zeros(nFreq, dtype='object')
    
    for iFreq in range(0, nFreq):
        
        units_pass_cuts[iFreq] = np.intersect1d( np.intersect1d(sig_units_eachFreq[iFreq], units_pass_rate_cut),  units_pass_FFcut)


    return units_pass_cuts

#%% average quantity over signficant frequencies 

# quantity:             (nUnits, nFreq)
# sig_units_eachFreq:   (nFreq,)

def fcn_avg_quantity_overFreq(quantity, units_eachFreq):
    
    nFreq = np.size(units_eachFreq)
    nUnits = np.size(quantity,0)
    
    freqAvg_quantity = np.ones(nUnits,)*np.nan
    
    for iUnit in range(0, nUnits):
        
        quantity_i = np.array([])
        
        for iFreq in range(0, nFreq):
            
            if iUnit in units_eachFreq[iFreq]:
                
                quantity_i = np.append(quantity_i, quantity[iUnit, iFreq])
                
        freqAvg_quantity[iUnit] = np.nanmean(quantity_i)
        
    return freqAvg_quantity


#%% average quantity over units and frequencies

# quantity:             (nUnits, nFreq)
# sig_units_eachFreq:   (nFreq,)

def fcn_avg_quantity_freq_units(quantity, units_eachFreq):
    
    nFreq = np.size(units_eachFreq)
    nUnits = np.size(quantity,0)
    
    quantity_flat = np.array([])
    
    for iUnit in range(0, nUnits):
                
        for iFreq in range(0, nFreq):
            
            if iUnit in units_eachFreq[iFreq]:
                
                quantity_flat = np.append(quantity_flat, quantity[iUnit, iFreq])
                

    avg_quantity = np.nanmean(quantity_flat)
        
    return avg_quantity



def fcn_avg_FF_timecourse_cellSubset_multipleStim(FF_timecourse, avg_units_eachFreq):
    
    nStim = np.size(FF_timecourse, 1)
    n_tPts = np.size(FF_timecourse, 2)
    
    avg_FF_timecourse = np.nan*np.ones((nStim, n_tPts))
    
    for indStim in range(0, nStim):
        
        avg_FF_timecourse[indStim, :] = np.nanmean(FF_timecourse[avg_units_eachFreq[indStim], indStim, :], axis=0)
        
    stimAvg_FF_timecourse = np.nanmean(avg_FF_timecourse, 0)
        
    return stimAvg_FF_timecourse




def fcn_compute_fano_cellSubset_multipleStim(FFevoked, FFdiff, cellSubset, tavg = 'min_allStim'):
    

    nCells = np.size(FFevoked, 0)
    nStim = np.size(FFevoked, 1)
    nTpts = np.size(FFevoked, 2)
    
    # initialize
    cellAvg_FFevoked = np.ones((nStim, nTpts))*np.nan
    evoked_fano_atMin = np.zeros((nCells, nStim))
    diff_fano_atMin = np.zeros((nCells, nStim))
    
    # loop over stimuli and compute cell averaged FF
    for indStim in range(0, nStim):
    
        # cell averaged evoked fano timecourse
        cellAvg_FFevoked[indStim, :] = np.nanmean(FFevoked[cellSubset[indStim], indStim, :], 0)
        
        
    # compute min of evoked and diff FF
    if ( (isinstance(tavg, str)) and (tavg == 'min_allStim') ):
        
        # time corresponding to minimum of cell and stim averaged fano factor
        tInd_min_evoked_fano = np.argmin(np.nanmean(cellAvg_FFevoked, 0))

        for indStim in range(0, nStim):
            
            # evoked fano at minimum time point for each cell
            evoked_fano_atMin[:, indStim] = FFevoked[:, indStim, tInd_min_evoked_fano].copy()
        
            # diff fano at minimum time point for each cell
            diff_fano_atMin[:, indStim] = FFdiff[:, indStim, tInd_min_evoked_fano].copy()     
            
    else: 
        
        
        for indCell in range(0, nCells):
        
            for indStim in range(0, nStim):
                
                tInd = tavg[indCell, indStim]
                
                # evoked fano at minimum time point for each cell
                evoked_fano_atMin[indCell, indStim] = FFevoked[indCell, indStim, tInd].copy()
            
                # diff fano at minimum time point for each cell
                diff_fano_atMin[indCell, indStim] = FFdiff[indCell, indStim, tInd].copy()             
    


    #------ save results for all cells
    evokedFF_cellSubset = fcn_avg_quantity_overFreq(evoked_fano_atMin, cellSubset)
    diffFF_cellSubset = fcn_avg_quantity_overFreq(diff_fano_atMin, cellSubset)
    
    return evokedFF_cellSubset, diffFF_cellSubset


#%% combine vector quantity across sessions

# quantity:     object array of size (nSessions, )
# quantity[i]:  array of size (n_i, )

def fcn_combine_vecQuantity_overSessions(quantity):
    
    nSessions = np.size(quantity)
    
    quantity_allSessions = np.array([])
    
    for iSession in range(0, nSessions):
        
        quantity_allSessions = np.append(quantity_allSessions, quantity[iSession])
        
    return quantity_allSessions
        
        
#%% combine quantity at low and high pupils across sessions

# quantity          object array of size (nSessions, nPupil)
# quantity[i,j]     array of size (n_i, )


def fcn_combine_vecQuantity_low_high_pupil_overSessions(quantity):    
    
    
    quantity_allSessions_lowPupil = fcn_combine_vecQuantity_overSessions(quantity[:, 0])
    quantity_allSessions_highPupil = fcn_combine_vecQuantity_overSessions(quantity[:, -1])
    
    return quantity_allSessions_lowPupil, quantity_allSessions_highPupil
    
    
