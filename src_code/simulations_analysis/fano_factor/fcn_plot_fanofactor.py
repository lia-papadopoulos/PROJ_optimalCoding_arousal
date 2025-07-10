
"""
helper functions for plotting the fano factor
"""

#%% basic imports

import sys
import numpy as np
from scipy.io import loadmat

sys.path.append('../../')
import global_settings

sys.path.append(global_settings.path_to_src_code + 'data_analysis/')     
from fcn_SuData_analysis import fcn_significant_cells
from fcn_SuData_analysis import fcn_significant_cells_responseSign


#%%

def fcn_sigUnits_pass_rateCut(units_pass_rate_cut, all_sig_units, sig_units_eachFreq, nStim):

    # significant cells that pass rate cut
    all_sig_units_pass_rateCut = np.intersect1d(all_sig_units, units_pass_rate_cut)
    
    sig_units_eachFreq_pass_rateCut = np.zeros((nStim), dtype='object')
    
    # significant cells at each frequency
    for indFreq in range(0, nStim):
        
        sig_units_eachFreq_pass_rateCut[indFreq] = np.intersect1d(sig_units_eachFreq[indFreq], units_pass_rate_cut)
        
    
    return all_sig_units_pass_rateCut, sig_units_eachFreq_pass_rateCut


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


#%% combine vector quantity across sessions

# quantity:     object array of size (nSessions, )
# quantity[i]:  array of size (n_i, )

def fcn_combine_vecQuantity_overNetworks(quantity):
    
    nSessions = np.size(quantity)
    
    quantity_allSessions = np.array([])
    
    for iSession in range(0, nSessions):
        
        quantity_allSessions = np.append(quantity_allSessions, quantity[iSession])
        
    return quantity_allSessions
        
        
#%% combine quantity at low and high pupils across sessions

# quantity          object array of size (nSessions, nPupil)
# quantity[i,j]     array of size (n_i, )


def fcn_combine_vecQuantity_low_high_pupil_overNetworks(quantity):    
    
    
    quantity_allSessions_lowPupil = fcn_combine_vecQuantity_overNetworks(quantity[:, 0])
    quantity_allSessions_highPupil = fcn_combine_vecQuantity_overNetworks(quantity[:, -1])
    
    return quantity_allSessions_lowPupil, quantity_allSessions_highPupil
    

#%% compute significantly responding cells


def fcn_sig_cells_allStim(nCells, psth_path, simID, net_type, sweep_param_name, sweep_param_vals, \
                          indNetwork, nStim, stim_shape, stim_rel_amp, windL, tSig, sig_level, sig_type):

    fname_end = ( '_stimType_%s_stim_rel_amp%0.3f_' % (stim_shape, stim_rel_amp) )
    

    base_rate = np.zeros((nCells, nStim, len(sweep_param_vals)))
    all_sig_cells = np.array([])
    significant_cells_eachStim = np.zeros((nStim), dtype='object')
    
    for indStim in range(0, nStim):
    
        
        # psth data
        data = loadmat((psth_path +  '%s_%s_sweep_%s_network%d_stim%d' + fname_end + '_psth_windSize%0.3fs.mat') % \
                       (simID, net_type, sweep_param_name, indNetwork, indStim, windL), simplify_cells=True)
            

        # UNPACK PSTH         
        base_window = data['parameters']['base_window']
        stim_window = data['parameters']['stim_window']  
        t_window = data['bin_times']  
        trialAvg_psth = data['trialAvg_psth_allBaseMod']
        trialAvg_gain = data['trialAvg_gain_allBaseMod']
        trialAvg_psth_eachBaseMod = data['trialAvg_psth_eachBaseMod']
        psth_pval_corrected = data['psth_pval_corrected']

        
        # baseline time windows
        base_bins = np.nonzero( (t_window <= base_window[1]) & (t_window >= base_window[0]) )[0]
        
        for indCell in range(0, nCells):
            
            for indBaseMod in range(0, len(sweep_param_vals)):
        
                # time average baseline firing rate
                base_rate[indCell, indStim, indBaseMod] = np.mean(trialAvg_psth_eachBaseMod[indCell, base_bins, indBaseMod]) # N x nBaseMod
        

        # significance of stimulus response
        _, sig_cells = \
            fcn_significant_cells(psth_pval_corrected, \
                                  tSig, t_window, \
                                      trialAvg_psth, \
                                          base_window, stim_window, sig_level)
                
        sig_cells_pos, sig_cells_neg = fcn_significant_cells_responseSign(sig_cells, t_window, trialAvg_gain, base_window, stim_window)
                
        if sig_type == 'all':
            significant_cells_eachStim[indStim] = sig_cells.copy()
            all_sig_cells = np.append(all_sig_cells, sig_cells)            
        elif sig_type == 'pos':
            significant_cells_eachStim[indStim] = sig_cells_pos.copy()
            all_sig_cells = np.append(all_sig_cells, sig_cells_pos)
        elif sig_type == 'neg':
            significant_cells_eachStim[indStim] = sig_cells_neg.copy()
            all_sig_cells = np.append(all_sig_cells, sig_cells_neg)
        else:
            sys.exit('unknown value for sig_type')

    # save all signficant cells
    significant_cells = np.unique(all_sig_cells).astype(int)

    return significant_cells, significant_cells_eachStim, base_rate



#%% average fano factor timecourse over specific set of units for each stimulus

'''
FF_timecourse: (N, nStim, nTpts)
avg_units: (nStim,) object array
'''

def fcn_avg_FF_timecourse_cellSubset_multipleStim(FF_timecourse, avg_units):
    
    nStim = np.size(FF_timecourse, 1)
    n_tPts = np.size(FF_timecourse, 2)
    
    avg_FF_timecourse = np.nan*np.ones((nStim, n_tPts))
    
    for indStim in range(0, nStim):
        
        avg_FF_timecourse[indStim, :] = np.nanmean(FF_timecourse[avg_units[indStim], indStim, :], axis=0)
        
    return avg_FF_timecourse

    
#%% compute fano factor quantities averaged over certain set of units

def fcn_compute_fano_cellSubset_singleStim(FFspont, FFevoked, FFdiff, cellSubset, tavg = 'min_allStim'):
    
    
    # cell averaged evoked fano timecourse
    cellAvg_evoked_fano = np.nanmean(FFevoked[cellSubset, :], 0)
    
    # time corresponding to minimum evoked fano
    if ( (tavg == 'min_allStim') or (tavg == 'min_eachStim') ):
        tInd_min_evoked_fano = np.argmin(cellAvg_evoked_fano)
    else:
        tInd_min_evoked_fano = tavg
    
    # evoked fano at minimum time point for each cell
    evoked_fano_atMin = FFevoked[cellSubset, tInd_min_evoked_fano].copy()
    
    # diff fano at minimum time point for each cell
    diff_fano_atMin = FFdiff[cellSubset, tInd_min_evoked_fano].copy()
    
    
    #------ save results for all cells
    spontFF_cellSubset = FFspont[cellSubset].copy()
    evokedFF_cellSubset = evoked_fano_atMin.copy()
    diffFF_cellSubset = diff_fano_atMin.copy()
    
    return spontFF_cellSubset, evokedFF_cellSubset, diffFF_cellSubset


def fcn_compute_fano_cellSubset_multipleStim(FFspont, FFevoked, FFdiff, cellSubset, tavg = 'min_allStim'):
    

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
    if tavg == 'min_eachStim':
        
        for indStim in range(0, nStim):
            
            # time corresponding to minimum evoked fano
            tInd_min_evoked_fano = np.argmin(cellAvg_FFevoked[indStim, :])
            
            # evoked fano at minimum time point for each cell
            evoked_fano_atMin[:, indStim] = FFevoked[:, indStim, tInd_min_evoked_fano].copy()
        
            # diff fano at minimum time point for each cell
            diff_fano_atMin[:, indStim] = FFdiff[:, indStim, tInd_min_evoked_fano].copy()
    
    
    elif tavg == 'min_allStim':
        
        # time corresponding to minimum of cell and stim averaged fano factor
        tInd_min_evoked_fano = np.argmin(np.nanmean(cellAvg_FFevoked, 0))

        for indStim in range(0, nStim):
            
            # evoked fano at minimum time point for each cell
            evoked_fano_atMin[:, indStim] = FFevoked[:, indStim, tInd_min_evoked_fano].copy()
        
            # diff fano at minimum time point for each cell
            diff_fano_atMin[:, indStim] = FFdiff[:, indStim, tInd_min_evoked_fano].copy()     
            
    else: 
        
        # use given time point
        tInd_min_evoked_fano = tavg
        
        for indStim in range(0, nStim):
            
            # evoked fano at minimum time point for each cell
            evoked_fano_atMin[:, indStim] = FFevoked[:, indStim, tInd_min_evoked_fano].copy()
        
            # diff fano at minimum time point for each cell
            diff_fano_atMin[:, indStim] = FFdiff[:, indStim, tInd_min_evoked_fano].copy()             
    


    #------ save results for all cells
    spontFF_cellSubset = fcn_avg_quantity_overFreq(FFspont, cellSubset)
    evokedFF_cellSubset = fcn_avg_quantity_overFreq(evoked_fano_atMin, cellSubset)
    diffFF_cellSubset = fcn_avg_quantity_overFreq(diff_fano_atMin, cellSubset)
    
    return spontFF_cellSubset, evokedFF_cellSubset, diffFF_cellSubset
