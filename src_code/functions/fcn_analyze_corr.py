
"""
functions to analyze correlation matrices and clustering structure
"""

import numpy as np
import sys


#%% fcn_sizeClusters

def fcn_sizeClusters(clusterID):
    
    '''
    compute size of each cluster in the partition
    '''
    
    maxCluster = np.max(clusterID).astype(int)
    
    sizeClusters = np.zeros((maxCluster + 1))
    
    for indClu in range(0, maxCluster + 1):

        sizeClusters[indClu] = np.size(np.nonzero(clusterID == indClu))
    
    
    return sizeClusters

#%% fcn_sorted_corrMatrix

def fcn_sorted_corrMatrix(corr_matrix, clusterID):
    
    '''
    sort correlation matrix according to clusters
    '''
    
    sorted_inds = np.argsort(clusterID)   
    corr_mat_sort = np.zeros(np.shape(corr_matrix))
    corr_mat_sort[:,:] = corr_matrix[sorted_inds, :][:,sorted_inds]
    
    return corr_mat_sort


#%% fcn_within_minus_between_corr_alt

def fcn_within_minus_between_corr_alt(clusterID, corr_matrix, goodClusters, avg_type, include_selfDist):
    
    '''
    summary statistic quantifying the difference between within and between cluster correlations
    '''
    
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


#%% fcn_compare_sigCorr_within_vs_between_true_shuffled_clustering_alt

def fcn_compare_sigCorr_within_vs_between_true_shuffled_clustering_alt(clusterID, sig_corr, goodClusters, avg_type, include_selfCorr, nShuffles):
    
    '''
    compare summary statistic quantifying the difference between within and between cluster correlations
    between true and permuted data
    '''

    within_minus_between_shuf = np.ones((nShuffles))*np.nan
    

    for indShuf in range(0, nShuffles):
        
        rand_clusterID = np.random.permutation(clusterID)
        
        within_minus_between_shuf[indShuf] = fcn_within_minus_between_corr_alt(rand_clusterID, sig_corr, goodClusters, avg_type, include_selfCorr)  
            
    
    within_minus_between_true = fcn_within_minus_between_corr_alt(clusterID, sig_corr, goodClusters, avg_type, include_selfCorr)     
    

    return within_minus_between_true, within_minus_between_shuf




