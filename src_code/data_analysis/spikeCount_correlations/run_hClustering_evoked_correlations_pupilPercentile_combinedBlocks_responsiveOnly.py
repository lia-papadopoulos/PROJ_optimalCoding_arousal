
#%% imports

# basic imports
import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
import sys
import numpy.matlib

# settings
import evoked_corr_settings as settings

# paths to functions
sys.path.append(settings.func_path1)
sys.path.append(settings.func_path2)
import fcn_hierarchical_clustering
from fcn_SuData_analysis import fcn_significant_preStim_vs_postStim

# import settings
linkage_method = settings.linkage_method
fcluster_criterion = settings.fcluster_criterion
cells_toKeep = settings.cells_toKeep
rate_thresh = settings.rate_thresh
wind_length = settings.wind_length_clustering
sig_level = settings.sig_level
global_pupilNorm = settings.global_pupilNorm
highDownsample = settings.highDownsample
runShuffle_corr = settings.runShuffle_corr
cellSelection = settings.cellSelection
load_path = settings.load_path_clustering
psth_path = settings.psth_path
outpath = settings.outpath_clustering
all_sessions_to_run = settings.sessions_to_run
base_subtract = settings.base_subtract

#%% loop over sessions

data_name = '' + cellSelection + '_globalPupilNorm'*global_pupilNorm + '_downSampled'*highDownsample



for session in all_sessions_to_run:


    base_subtract_str = '_baselineSubtract'*base_subtract

    fname = ('evoked_correlations_pupilPercentile_combinedBlocks_%s_windLength%0.3fs_%s%s.mat' % (session, wind_length, base_subtract_str, data_name))
        
        
    ### psth data
    psth_data = loadmat(('%spsth_allTrials_%s_windLength0.100s%s.mat' % (psth_path, session, data_name)), simplify_cells=True)

    ### correlations
    data = loadmat(load_path + fname, simplify_cells=True)
    nSubsamples = data['params']['nSubsamples']
    rates_allPupil = data['trialAvg_evoked_spkCount']/wind_length
    nTrials_sample = data['nTrials_subsample']
    corr_allPupil_raw = data['corr_evoked_pupilAvg_freqAvg'].copy()

    if runShuffle_corr:
        corr_allPupil_raw_shuffle = data['corr_evoked_pupilAvg_freqAvg_shuffle'].copy()
        print('make sure that you are running the shuffle you want.')


    nCells = np.size(corr_allPupil_raw, 0)
    
    ### significant cells
    sig_cells_dict = fcn_significant_preStim_vs_postStim(psth_data, sig_level)
    
    if cells_toKeep == 'allSigCells':
        allSigCells = sig_cells_dict['allSigCells'].astype(int)
    elif cells_toKeep == 'allSigCells_posResponse':
        allSigCells = sig_cells_dict['allSigCells_posResponse'].astype(int)
    else:
        allSigCells = np.arange(0,nCells)
        

    nSigCells = np.size(allSigCells)
    
    ### remove non significant cells
    remove_cells = np.setdiff1d(np.arange(0,nCells), allSigCells)
    
    # rates during each pupil bin
    rates_allPupil = rates_allPupil[allSigCells].copy()
    
    # correlation for all pupil bins
    corr_allPupil_sigCells = corr_allPupil_raw[:,allSigCells][allSigCells,:].copy()

    # find low rate and nan cells
    bad_sigCells = fcn_hierarchical_clustering.fcn_find_badCells(corr_allPupil_sigCells, rates_allPupil, rate_thresh)
    remove_cells = np.sort(np.append(remove_cells, allSigCells[bad_sigCells]))


    ### cleaned correlation matrices
    corr_allPupil_cleaned = fcn_hierarchical_clustering.fcn_remove_badCells(corr_allPupil_raw, remove_cells)
            
        
    # if running clustering on shuffled correlation matrix
    if runShuffle_corr:
        
        nCells_cleaned = np.size(corr_allPupil_cleaned, 1)
        corr_allPupil_shuffle_cleaned = np.zeros((nCells_cleaned, nCells_cleaned, nSubsamples))
        
        for indSample in range(0, nSubsamples):
            
            corr_allPupil_shuffle_cleaned[:,:,indSample] = \
                fcn_hierarchical_clustering.fcn_remove_badCells(corr_allPupil_raw_shuffle[:, :, indSample], remove_cells)
        
        # set any nan values to zero
        corr_allPupil_shuffle_cleaned[np.isnan(corr_allPupil_shuffle_cleaned)] = 0.



    ### clustering
    
    # all pupil
    dissimilarity_allPupil = fcn_hierarchical_clustering.fcn_compute_dissimilarity(corr_allPupil_cleaned)
    linkageMatrix_allPupil = fcn_hierarchical_clustering.fcn_run_hierarchical_clustering(corr_allPupil_cleaned, linkage_method)
    
    # all pupil shuffle
    if runShuffle_corr:
        
        dissimilarity_allPupil_shuffle = np.ones((nSubsamples), dtype='object')
        linkageMatrix_allPupil_shuffle = np.ones((nSubsamples), dtype='object')
        
        for indSample in range(0, nSubsamples):
            
            dissimilarity_allPupil_shuffle[indSample] = fcn_hierarchical_clustering.fcn_compute_dissimilarity(corr_allPupil_shuffle_cleaned[:,:,indSample])
            linkageMatrix_allPupil_shuffle[indSample] =                       fcn_hierarchical_clustering.fcn_run_hierarchical_clustering(corr_allPupil_shuffle_cleaned[:,:,indSample], linkage_method)

 
    ### save the data
    data_save = dict()
        
    data_save['params'] = dict()
    data_save['params']['rate_thresh'] = rate_thresh
    data_save['params']['sig_level'] = sig_level
    data_save['params']['wind_length'] = wind_length
    
    data_save['dissimilarity_allPupil'] = dissimilarity_allPupil
    data_save['linkageMatrix_allPupil'] = linkageMatrix_allPupil
    data_save['corr_allPupil'] = corr_allPupil_cleaned        

    
    if runShuffle_corr:
        data_save['dissimilarity_allPupil_shuffle'] = dissimilarity_allPupil_shuffle
        data_save['linkageMatrix_allPupil_shuffle'] = linkageMatrix_allPupil_shuffle   
        data_save['corr_allPupil_shuffle'] = corr_allPupil_shuffle_cleaned
        

    data_save['linkage_method'] = linkage_method
    data_save['fcluster_criterion'] = fcluster_criterion
    data_save['cells_toKeep'] = cells_toKeep
    data_save['remove_cells'] = remove_cells
        
    savemat(('%s%s_responsiveOnly_windLength%0.3fs_rateThresh%0.3fHz_hClustering_%s%s%s.mat' % (outpath, session, wind_length, rate_thresh, \
                                                                                                cells_toKeep, base_subtract_str, data_name)), data_save)
        
    print(session)
    
