
"""
run hierarchical clustering
evoked correlations
"""


#%%

import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
import sys
import numpy.matlib
import importlib
import os

# loading parameters
simParams_fname = 'simParams_051325_clu'
#simParams_fname = 'simParams_051325_hom'
sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'
net_type = 'baseEIclu'
#net_type = 'baseHOM'
nNetworks = 10

# parameters
psth_windSize = 100e-3
corr_windSize = 100e-3
Ecells_only = True
sigLevel = 0.05
run_parCorr = False
rate_thresh = -np.inf
run_configCorr = False
run_shuffleCorr = True
nNulls = 100
n_neuronDraws = 10
linkage_method = 'average'

# data paths
func_path0 = '/home/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/scripts/master_sims/'
func_path1 = '/home/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/scripts/functions/'
sim_params_path = '/home/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/scripts/master_sims/'
psth_path = '/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/test_stim_expSyn/psth/'
corr_path = '/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/test_stim_expSyn/evoked_corr/'
sim_path = '/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/test_stim_expSyn/'
outpath = '/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/test_stim_expSyn/evoked_corr/hClustering/'


sys.path.append(func_path0)
sys.path.append(func_path1)
import fcn_hierarchical_clustering
import fcn_analyze_corr
import fcn_compute_firing_stats
from fcn_simulation_setup import fcn_define_arousalSweep


#%% load sim parameters
sys.path.append(sim_params_path)
params = importlib.import_module(simParams_fname) 
s_params = params.sim_params

#%% arousal sweep
s_params = fcn_define_arousalSweep(s_params)

#%% unpack simulation parameters
simID = s_params['simID']
n_sweepParams = s_params['nParams_sweep']
swept_params_dict = s_params['swept_params_dict']
simID = s_params['simID']
nStim = s_params['nStim']
stim_shape = s_params['stim_shape']
stim_type = s_params['stim_type']
stim_rel_amp = s_params['stim_rel_amp']

del params
del s_params


#%% filenames
fname_psth = ( '%s%s_%s_sweep_%s_network%d_stim%d_stimType_%s_stim_rel_amp%0.3f_psth_windSize%0.3fs.mat')
fname_corr = ( '%s%s_%s_sweep_%s_network%d_stimType_%s_stim_rel_amp%0.3f_evoked_corr_windL%0.3f.mat' )

if os.path.exists(outpath) == False:
    os.makedirs(outpath) 

#%% number of arousal values
nParam_vals = np.size(swept_params_dict['param_vals1'])

#%% loop over networks

for indNet in range(0, nNetworks): 


    # load psth data to get significant cells for each stim
    sigCells_eachStim = np.zeros((nStim), dtype='object')
    resp_eachStim = np.zeros((nStim), dtype='object')
    
    # each stimulus
    for indStim in range(0, nStim):
        
        # filename
        params_tuple = (psth_path, simID, net_type, sweep_param_name, indNet, indStim, stim_shape, stim_rel_amp, psth_windSize)        
        fname_psth_full = ( (fname_psth) % ( params_tuple ))
        
        # load psth data
        psth_data = loadmat(fname_psth_full, simplify_cells=True)
        
        # significant cells and responses
        sigCells_eachStim[indStim], _ = fcn_compute_firing_stats.fcn_compute_sigCells_respAmp(psth_data, sigLevel)
        

    # cells that respond significantly to any stimulus
    allSigCells = fcn_compute_firing_stats.fcn_sigCells_anyStimulus(sigCells_eachStim)
    allSigCells = allSigCells.astype(int)
    
    
    # initialize
    remove_cells_subsample = np.ones((n_neuronDraws), dtype='object')*np.nan
    keep_cells_subsample = np.ones((n_neuronDraws), dtype='object')*np.nan
    
    corr_save = np.ones((n_neuronDraws), dtype='object')*np.nan
    dissimilarity = np.ones((n_neuronDraws), dtype='object')*np.nan
    linkageMatrix = np.ones((n_neuronDraws), dtype='object')*np.nan
    
    corr_shuffle_save = np.ones((n_neuronDraws), dtype='object')*np.nan
    dissimilarity_shuffle = np.ones((n_neuronDraws, nNulls), dtype='object')*np.nan
    linkageMatrix_shuffle = np.ones((n_neuronDraws, nNulls), dtype='object')*np.nan

    corr_config_save = np.ones((n_neuronDraws), dtype='object')*np.nan    
    dissimilarity_config = np.ones((n_neuronDraws, nNulls), dtype='object')*np.nan
    linkageMatrix_config = np.ones((n_neuronDraws, nNulls), dtype='object')*np.nan  
    
    
    # load correlation data
    params_tuple = (corr_path, simID, net_type, sweep_param_name, indNet, stim_shape, stim_rel_amp, corr_windSize)
    fname_corr_full = ( (fname_corr) % (params_tuple) )
    corr_data = loadmat(fname_corr_full, simplify_cells = True)
            
    
    # unpack corr data
    nTrials = corr_data['parameters']['nTrials']
    nSamples = nTrials * nStim * nParam_vals
    nCells_all = corr_data['parameters']['nCells_all']
    
    
    # remove I cells
    if Ecells_only == True:
        Icells = np.nonzero(allSigCells >= nCells_all)[0]
        allSigCells = np.delete(allSigCells, Icells)

    
    # loop over neuron draws
    for indDraw in range(0, n_neuronDraws):
    
        avg_spikeCount_allTrials = corr_data['avg_spkCount_allTrials_subsample'][:, indDraw].copy()
        corr = corr_data['corr_stimAvg_arousalAvg'][:,:,indDraw].copy()
        corr_shuffle = corr_data['corr_stimAvg_arousalAvg_shuffle'][:,:,indDraw,:].copy()
        indCells_sample = corr_data['parameters']['indCells_sample'][:, indDraw].copy()

    
        # only keep subsample cells in sig cells
        allSigCells_subsample = np.array([])
        remove_cells_subsample[indDraw] = np.array([])
        for indCell, cell in enumerate(indCells_sample):
            if cell not in allSigCells:
                remove_cells_subsample[indDraw] = np.append(remove_cells_subsample[indDraw], indCell)
            else:
                allSigCells_subsample = np.append(allSigCells_subsample, indCell)
    
        allSigCells_subsample = allSigCells_subsample.astype(int)
    
        nSigCells_subsample = np.size(allSigCells_subsample)
        
    
        # rates during each pupil bin
        avg_spikeCount_allTrials = avg_spikeCount_allTrials[allSigCells_subsample].copy()
        
        # correlation for all pupil bins
        corr_sigCells = corr.copy()
        corr_sigCells = corr_sigCells[:,allSigCells_subsample][allSigCells_subsample,:].copy()
        bad_sigCells = fcn_hierarchical_clustering.fcn_find_badCells(corr_sigCells, avg_spikeCount_allTrials, rate_thresh)
        
        remove_cells_subsample[indDraw] = np.append(remove_cells_subsample[indDraw], allSigCells_subsample[bad_sigCells]).astype(int)    
    
        ### cleaned correlation matrices
        corr_cleaned = fcn_hierarchical_clustering.fcn_remove_badCells(corr, remove_cells_subsample[indDraw])            
        corr_save[indDraw] = corr_cleaned.copy()
        
        corr_shuffle_cleaned = np.zeros((np.size(corr_cleaned,0),np.size(corr_cleaned,0),nNulls))    
                
        for indShuffle in range(0, nNulls):
            corr_shuffle_instance = corr_shuffle[:,:,indShuffle].copy()
            corr_shuffle_cleaned[:,:,indShuffle] = fcn_hierarchical_clustering.fcn_remove_badCells(corr_shuffle_instance, remove_cells_subsample[indDraw])    
        
        corr_shuffle_cleaned[np.isnan(corr_shuffle_cleaned)] = 0.
        corr_shuffle_save[indDraw] = corr_shuffle_cleaned.copy()
        
    
        # if running clustering on configuration null model         
        if run_configCorr:
            
            # correlation matrix
            C = corr_cleaned.copy()
            
            # number of cells
            N = C.shape[0] 
    
            # initialize configuration null model
            C_config = np.zeros((N, N, nNulls))
            
            # compute null model
            C_con = fcn_analyze_corr.fcn_compute_configCorr_nullModel(C)
                
            # loop over number of repetitions
            for indNull in range(0, nNulls):
                    
                # correlation matrix from configuration null model
                C_sam = fcn_analyze_corr.fcn_sample_configCorr_nullModel(C_con, nSamples)
            
                # store
                C_config[:,:,indNull] = C_sam.copy()
    
    
            corr_config_save[indDraw] = C_config.copy()
        
            
        ### update removed cells
        keep_cells_subsample[indDraw] = np.delete(indCells_sample, remove_cells_subsample[indDraw].astype(int))
    
        ### clustering
        
        # all pupil
        dissimilarity[indDraw] = fcn_hierarchical_clustering.fcn_compute_dissimilarity(corr_cleaned)
        linkageMatrix[indDraw] = fcn_hierarchical_clustering.fcn_run_hierarchical_clustering(corr_cleaned, linkage_method)
        
        # shuffle corr
        if run_shuffleCorr:
            
            for indNull in range(0, nNulls):
                
                dissimilarity_shuffle[indDraw, indNull] = fcn_hierarchical_clustering.fcn_compute_dissimilarity(corr_shuffle_cleaned[:,:,indNull])
                linkageMatrix_shuffle[indDraw, indNull] = fcn_hierarchical_clustering.fcn_run_hierarchical_clustering(corr_shuffle_cleaned[:,:,indNull], linkage_method)        
    
        # configuration corr
        if run_configCorr:
            
            for indNull in range(0, nNulls):
                
                dissimilarity_config[indDraw, indNull] = fcn_hierarchical_clustering.fcn_compute_dissimilarity(C_config[:,:,indNull])
                linkageMatrix_config[indDraw, indNull] = fcn_hierarchical_clustering.fcn_run_hierarchical_clustering(C_config[:,:,indNull], linkage_method)
                        
            
   
    ### save the data
    data_save = dict()
        
    data_save['params'] = dict()
    data_save['params']['rate_thresh'] = rate_thresh
    data_save['params']['sig_level'] = sigLevel
    data_save['params']['run_parCorr'] = run_parCorr
    data_save['params']['psth_windSize'] = psth_windSize
    data_save['params']['corr_windSize'] = corr_windSize
    
    data_save['dissimilarity'] = dissimilarity
    data_save['linkageMatrix'] = linkageMatrix
    data_save['corr'] = corr_save        
    

    if run_shuffleCorr:
        print('here')
        data_save['dissimilarity_shuffle'] = dissimilarity_shuffle
        data_save['linkageMatrix_shuffle'] = linkageMatrix_shuffle   
        data_save['corr_shuffle'] = corr_shuffle_save
        
    if run_configCorr:
        data_save['dissimilarity_config'] = dissimilarity_config
        data_save['linkageMatrix_config'] = linkageMatrix_config   
        data_save['corr_config'] = corr_config_save

        
    data_save['linkage_method'] = linkage_method
    data_save['remove_cells_subsample'] = remove_cells_subsample   
    data_save['keep_cells_subsample'] = keep_cells_subsample   

    savename = ( ('%s%s_%s_sweep_%s_network%d_stimType_%s_stim_rel_amp%0.3f.mat') % (outpath, simID, net_type, sweep_param_name, indNet, stim_shape, stim_rel_amp) )
    savemat(savename, data_save)        
        
    print(indNet)