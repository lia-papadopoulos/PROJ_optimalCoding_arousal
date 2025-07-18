
"""
analyze correlations
"""

import numpy as np
import sys
import matplotlib.pyplot as plt

#%%


def fcn_get_pupilBins(good_pupilBins, avg_pupilSize_trials):
    
    allBins = np.array([])
    
    for indPupil in range(0, len(good_pupilBins)):
        
        allBins = np.append(allBins, good_pupilBins[indPupil])
        
    allBins = np.unique(allBins)
    
    pupilBins = allBins.astype(int)
    pupilSizes = avg_pupilSize_trials[pupilBins].copy()
    
    return pupilBins, pupilSizes
    

#%% COMPUTE DIMENSIONALITY
#
#   INPUTS:
#       cov_sc --                   output of fcn_compute_spikeCount_cov
#                                   [pairwise spike counts covariance matrix]
#
#   OUPUTS:
#       dimensionality --           a single number corresponding to the
#                                   dimensionality of the neural data

def fcn_compute_dim(cov_sc):

    tr_cov = np.trace(cov_sc)
    tr_cov_sq = np.trace(np.matmul(cov_sc,cov_sc))
    dimensionality = (tr_cov**2)/tr_cov_sq 
    
    # return
    return dimensionality


#%%

def fcn_sorted_corrMatrix(corr_matrix, clusterID):
    
    
    sorted_inds = np.argsort(clusterID)   
    corr_mat_sort = np.zeros(np.shape(corr_matrix))
    corr_mat_sort[:,:] = corr_matrix[sorted_inds, :][:,sorted_inds]
    
    return corr_mat_sort

#%%

def fcn_plot_corrMatrix_withClusters(corr_matrix, clusterID, clustersHighlight, plotColor='turquoise', cbar_label = 'corr', cmap='seismic', cbar_lim = 0.5):
    
    
    sorted_inds = np.argsort(clusterID)
    
    corr_mat_sort = np.zeros(np.shape(corr_matrix))
    corr_mat_sort[:,:] = corr_matrix[sorted_inds, :][:,sorted_inds]
    
    min_indx = np.zeros(len(clustersHighlight))
    max_indx = np.zeros(len(clustersHighlight))
    
    for indClu, clu in enumerate(clustersHighlight):
        
        cells_inClu = np.nonzero(clusterID == clu)[0]
        
        sortedInd_inClu = np.zeros(len(cells_inClu))
        
        for indx, cell in enumerate(cells_inClu):
        
            sortedInd_inClu[indx] = np.nonzero(sorted_inds == cell)[0]
            
        min_indx[indClu] = np.min(sortedInd_inClu) - 0.5
        max_indx[indClu] = np.max(sortedInd_inClu) + 0.5
        
    
    plt.rcParams.update({'font.size': 10})
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_axes([0.3, 0.3, 0.6, 0.6]) 
    if cbar_lim == 'corr_based':
        f = ax.imshow(corr_mat_sort, vmin=-np.nanmax(np.abs(corr_mat_sort)), vmax=np.nanmax(np.abs(corr_mat_sort)), cmap=cmap, interpolation=None)
        cbar = fig.colorbar(f, ax=ax, shrink = 0.5, label=cbar_label)
    else:
        f = ax.imshow(corr_mat_sort, vmin=-cbar_lim, vmax=cbar_lim, cmap=cmap, interpolation=None)   
        cbar = fig.colorbar(f, ax=ax, shrink = 0.5, label=cbar_label, ticks = [-cbar_lim, 0, cbar_lim])
        cbar.ax.set_yticklabels([('< %0.1f' % -cbar_lim), '0', ('> %0.1f' % cbar_lim)]) 

    for indClu in range(0, len(clustersHighlight)):
        ax.plot([min_indx[indClu], min_indx[indClu]], [min_indx[indClu],max_indx[indClu]], color=plotColor, linewidth=2.)
        ax.plot([max_indx[indClu], max_indx[indClu]], [min_indx[indClu],max_indx[indClu]], color=plotColor, linewidth=2.)
        ax.plot([min_indx[indClu], max_indx[indClu]], [min_indx[indClu],min_indx[indClu]], color=plotColor, linewidth=2.)
        ax.plot([min_indx[indClu], max_indx[indClu]], [max_indx[indClu],max_indx[indClu]], color=plotColor, linewidth=2.)
                  
    
    return fig, ax


#%%


def fcn_withinCluster_corr(clusterID, corr_matrix, goodClusters, avg_type, include_selfCorr):
    
    sizeClusters = fcn_sizeClusters(clusterID)
    nClusters = np.size(sizeClusters)
    
    corr = corr_matrix.copy()
    
    if include_selfCorr == True:
        np.fill_diagonal(corr, 1)
    elif include_selfCorr == False:    
        np.fill_diagonal(corr, 0)
        
    n_goodClusters = np.size(goodClusters) 
    size_goodClusters = sizeClusters[goodClusters]
    
    avg_within_corr = np.ones((nClusters))*np.nan
    

    if n_goodClusters == 1:
        
        if avg_type == 'v1':
            
            total_within_corr = 0.
            
        elif avg_type == 'v2':
        
            total_within_corr = 0.
                    
        return total_within_corr
    
    
    for _, clu_i in enumerate(goodClusters):
        
        inCluster = np.nonzero(clusterID == clu_i)[0]
        
        withinCluster_corr = corr[inCluster,:][:,inCluster].copy()
        
        if avg_type == 'v1':
            
            avg_within_corr[clu_i] = np.sum( np.mean(withinCluster_corr, 1) )
        
        elif avg_type == 'v2':
        
            avg_within_corr[clu_i] = np.mean(withinCluster_corr)
            
        else:
            
            sys.exit()
            
    if avg_type == 'v1':
            
        total_within_corr = np.sum(avg_within_corr[goodClusters])/np.sum(size_goodClusters)
            
    else:
            
        total_within_corr = np.mean(avg_within_corr[goodClusters])
                    

    return total_within_corr


def fcn_compare_sigCorr_within_true_shuffled_clustering(clusterID, sig_corr, goodClusters, avg_type, include_selfCorr, nShuffles):
    

    within_shuff = np.ones((nShuffles))*np.nan
    

    for indShuf in range(0, nShuffles):
        
        rand_clusterID = np.random.permutation(clusterID)
        
        within_shuff[indShuf] = fcn_withinCluster_corr(rand_clusterID, sig_corr, goodClusters, avg_type, include_selfCorr)  
            
    
    within_true = fcn_withinCluster_corr(clusterID, sig_corr, goodClusters, avg_type, include_selfCorr)     
    

    return within_true, within_shuff



#%%

def fcn_within_minus_between_corr(clusterID, corr_matrix, goodClusters, avg_type, include_selfDist, betweenCorr):
    
    sizeClusters = fcn_sizeClusters(clusterID)
    nClusters = np.size(sizeClusters)
    
    if betweenCorr == 'all':
        betweenClusters = np.arange(0, nClusters)
    else:
        betweenClusters = goodClusters.copy()
    
    corr = corr_matrix.copy()
    
    if include_selfDist == True:
        np.fill_diagonal(corr, 1)
    elif include_selfDist == False:    
        np.fill_diagonal(corr, 0)
        
    n_goodClusters = np.size(goodClusters) 
    size_goodClusters = sizeClusters[goodClusters]
    
    avg_within_between_cluster_corr = np.ones((nClusters))*np.nan
    

    if n_goodClusters == 1:
        
        if avg_type == 'v1':
            
            total_within_minus_between_cluster_corr = 0.
            
        elif avg_type == 'v2':
        
            total_within_minus_between_cluster_corr = 0.
                    
        return total_within_minus_between_cluster_corr
    
    
    for _, clu_i in enumerate(goodClusters):
        
        inCluster = np.nonzero(clusterID == clu_i)[0]
        not_inCluster = np.array([])
        
        for _, clu_j in enumerate(betweenClusters):
            
            if clu_j == clu_i:
                not_inCluster = np.append(not_inCluster, np.array([]))
            else:
                not_inCluster = np.append(not_inCluster, np.nonzero(clusterID == clu_j)[0])
            
        not_inCluster = not_inCluster.astype(int)
        if (np.size(np.intersect1d(inCluster, not_inCluster)) != 0):
            sys.exit('in and not-in cluster ids overlap')
        
        withinCluster_corr = corr[inCluster,:][:,inCluster].copy()
        betweenCluster_corr = corr[inCluster,:][:,not_inCluster].copy()
        
        if avg_type == 'v1':
            
            avg_within_between_cluster_corr[clu_i] = np.sum(np.mean(withinCluster_corr, 1) - np.mean(betweenCluster_corr, 1))
        
        elif avg_type == 'v2':
        
            avg_within_between_cluster_corr[clu_i] = np.mean(withinCluster_corr) - np.mean(betweenCluster_corr) 
            
        else:
            
            sys.exit()
            
    if avg_type == 'v1':
            
        total_within_minus_between_cluster_corr = np.sum(avg_within_between_cluster_corr[goodClusters])/np.sum(size_goodClusters)
            
    else:
            
        total_within_minus_between_cluster_corr = np.mean(avg_within_between_cluster_corr[goodClusters])


    return total_within_minus_between_cluster_corr



def fcn_compare_sigCorr_within_vs_between_true_shuffled_clustering(clusterID, sig_corr, goodClusters, avg_type, include_selfCorr, betweenCorr, nShuffles):
    

    within_minus_between_shuf = np.ones((nShuffles))*np.nan
    

    for indShuf in range(0, nShuffles):
        
        rand_clusterID = np.random.permutation(clusterID)
        
        within_minus_between_shuf[indShuf] = fcn_within_minus_between_corr(rand_clusterID, sig_corr, goodClusters, avg_type, include_selfCorr, betweenCorr)  
            
    
    within_minus_between_true = fcn_within_minus_between_corr(clusterID, sig_corr, goodClusters, avg_type, include_selfCorr, betweenCorr)     
    

    return within_minus_between_true, within_minus_between_shuf




#%%

def fcn_within_minus_between_corr_alt(clusterID, corr_matrix, goodClusters, avg_type, include_selfDist):
    
    sizeClusters = fcn_sizeClusters(clusterID)
    nClusters = np.size(sizeClusters)
    
    corr = corr_matrix.copy()
    
    if include_selfDist == True:
        np.fill_diagonal(corr, 1)
    elif include_selfDist == False:    
        np.fill_diagonal(corr, 0)
        
    size_goodClusters = sizeClusters[goodClusters]
    
    avg_within_between_cluster_corr = np.ones((nClusters))*np.nan
    
    
    if np.size(goodClusters) == 1:
        goodClusters = np.array([goodClusters])
    
    for _, clu_i in enumerate(goodClusters):
        
        inCluster = np.nonzero(clusterID == clu_i)[0]
        not_inCluster = np.nonzero(clusterID != clu_i)[0]
        
        withinCluster_corr = corr[inCluster,:][:,inCluster].copy()
        betweenCluster_corr = corr[inCluster,:][:,not_inCluster].copy()
        
        if avg_type == 'v1':
            
            avg_within_between_cluster_corr[clu_i] = np.sum(np.mean(withinCluster_corr, 1) - np.mean(betweenCluster_corr, 1))
        
        elif avg_type == 'v2':
        
            avg_within_between_cluster_corr[clu_i] = np.mean(withinCluster_corr) - np.mean(betweenCluster_corr) 
            
        else:
            
            sys.exit()
            
    if avg_type == 'v1':
            
        total_within_minus_between_cluster_corr = np.sum(avg_within_between_cluster_corr[goodClusters])/np.sum(size_goodClusters)
            
    else:
            
        total_within_minus_between_cluster_corr = np.mean(avg_within_between_cluster_corr[goodClusters])


    return total_within_minus_between_cluster_corr



def fcn_compare_sigCorr_within_vs_between_true_shuffled_clustering_alt(clusterID, sig_corr, goodClusters, avg_type, include_selfCorr, nShuffles):
    

    within_minus_between_shuf = np.ones((nShuffles))*np.nan
    

    for indShuf in range(0, nShuffles):
        
        rand_clusterID = np.random.permutation(clusterID)
        
        within_minus_between_shuf[indShuf] = fcn_within_minus_between_corr_alt(rand_clusterID, sig_corr, goodClusters, avg_type, include_selfCorr)  
            
    
    within_minus_between_true = fcn_within_minus_between_corr_alt(clusterID, sig_corr, goodClusters, avg_type, include_selfCorr)     
    

    return within_minus_between_true, within_minus_between_shuf


#%% within cluster frequency difference

def fcn_cluster_BFdiff(indBF, clusterID, neg_BF_dist = 2):
    
        
    nClusters = np.size(np.unique(clusterID))

    sizeClusters = np.zeros((nClusters))

    for indClu in range(0,nClusters):

        sizeClusters[indClu] = np.size(np.nonzero(clusterID == indClu))
        
    
    bfDiff = np.zeros((nClusters))
    

    for indClu in range(0, nClusters):
        
        if sizeClusters[indClu] <= 1:
            
            bfDiff[indClu] = np.nan
            
            continue
        
        cells_inClu = np.nonzero(clusterID == indClu)[0].astype(int)
        
        bfDiff_cluster = np.array([])

        for ind_i, cell_i in enumerate(cells_inClu):
            
            for ind_j, cell_j in enumerate(cells_inClu):
                
                if ind_j > ind_i:
                    
                    indBF_i = indBF[cell_i]
                    indBF_j = indBF[cell_j]
                    
                    if ((indBF_i == -1) and (indBF_j != -1)):
                        
                        bfDiff_cluster = np.append(bfDiff_cluster, neg_BF_dist)
                        
                    elif ((indBF_i != -1) and (indBF_j == -1)):
                        
                        bfDiff_cluster = np.append(bfDiff_cluster, neg_BF_dist)                        
                        
                    else:
                    
                        bfDiff_cluster = np.append(bfDiff_cluster, np.abs(indBF_i-indBF_j))
                    
        bfDiff[indClu] = np.mean(bfDiff_cluster)
                    
                    
    return bfDiff


#%% cluster selectivity index

# close to nFreq: even distribution of frequencies
# close to 1: single frequency for all neurons

def fcn_cluster_selectivity(bf, clusterID, freqVals):
    
    nCells = np.size(bf)
    
    nFreqs = np.size(freqVals)
    
    nClusters = np.size(np.unique(clusterID))

    sizeClusters = np.zeros((nClusters))

    for indClu in range(0,nClusters):

        sizeClusters[indClu] = np.size(np.nonzero(clusterID == indClu))
        
        
    bf_fraction = np.zeros((nFreqs))
    
    for indFreq in range(0, nFreqs):
        
        bf_fraction[indFreq] = np.size(np.nonzero(bf == freqVals[indFreq]))/nCells
        
    
    cluSelective_norm = np.zeros((nClusters))
    cluSelective = np.zeros((nClusters))
    

    for indClu in range(0, nClusters):
        
        if sizeClusters[indClu] <= 1:
            
            cluSelective[indClu] = np.nan
            cluSelective_norm[indClu] = np.nan
            
            continue
        
        cells_inClu = np.nonzero(clusterID == indClu)[0].astype(int)
        bf_inClu = bf[cells_inClu].copy()

        n_cells_inCluster = np.size(cells_inClu)
        
        bf_fraction_true = np.zeros((nFreqs))
        bf_fraction_true_norm = np.zeros((nFreqs))
    
    
        for indFreq in range(0, nFreqs):
    
            bf_fraction_true_norm[indFreq] = np.size(np.nonzero(bf_inClu == freqVals[indFreq]))/n_cells_inCluster/bf_fraction[indFreq]
            bf_fraction_true[indFreq] = np.size(np.nonzero(bf_inClu == freqVals[indFreq]))/n_cells_inCluster
                        
        
        cluSelective[indClu] = (np.sum(bf_fraction_true)**2)/(np.sum(bf_fraction_true**2))
        cluSelective_norm[indClu] = (np.nansum(bf_fraction_true_norm)**2)/(np.nansum(bf_fraction_true_norm**2))
        
    

    return cluSelective, cluSelective_norm, sizeClusters


#%% size of clusters

def fcn_sizeClusters(clusterID):
    
    maxCluster = np.max(clusterID).astype(int)
    
    sizeClusters = np.zeros((maxCluster + 1))
    
    for indClu in range(0, maxCluster + 1):

        sizeClusters[indClu] = np.size(np.nonzero(clusterID == indClu))
    
    
    return sizeClusters



#%% compare correlation between cells of same and different best frequencies

def fcn_corr_sameBF_diffBF(bf, corr):
    
    
    corr_sameBF = np.array([])
    corr_diffBF = np.array([])
    
    for i in range(0, np.size(bf)):
        for j in range(0, np.size(bf)):
            
            if j > i:
                
                bf_i = bf[i]
                bf_j = bf[j]
                
                if bf_i == bf_j:
                    
                    corr_sameBF = np.append(corr_sameBF, corr[i,j])
                
                else:
                    
                    corr_diffBF = np.append(corr_diffBF, corr[i,j])

    return corr_sameBF, corr_diffBF




