
"""
set of functions used to run hierarchical clustering analyses
"""

from scipy.cluster.hierarchy import linkage, fcluster, ward, cut_tree
from scipy.spatial.distance import squareform
import numpy as np
import sys
  
#%% 

def fcn_find_badCells(corr, rates, rate_thresh):
    
    ### low rate cells
    lowrate = np.nonzero( rates < rate_thresh )[0]
    
    ### find nan cells
    nanInds = fcn_find_nan_v2(corr)

    ### update lowrate to include nan cells
    all_badCells = np.unique(np.append(lowrate,nanInds)).astype(int)    
    
    return all_badCells



def fcn_find_nan_v2(corr_mat):
    
    np.fill_diagonal(corr_mat,1)

    nCells = np.size(corr_mat,0)
    
    # find number of nan values in each row
    nNan_eachCell = np.zeros(nCells)
    for i in range(0, nCells):
        nNan_eachCell[i] = np.size(np.nonzero(np.isnan(corr_mat[i,:])))

    # iteratively remove cells with highest number of nans until none exit
    
    sortCells_mostNans = np.flip(np.argsort(nNan_eachCell)) # most to least
    nan_inds = np.array([])
    count = -1
    
    if np.any(nNan_eachCell > 0):
        
        stillNans = True
        
    else:
        
        stillNans = False
    
    
    while stillNans:
        
        count += 1
        
        corr_mat_copy = corr_mat.copy()
        
        nan_inds = np.append(nan_inds, sortCells_mostNans[count])
        keep_cells = np.setdiff1d(np.arange(0, nCells), nan_inds)
        
        corr_mat_copy = corr_mat_copy[:,keep_cells][keep_cells,:]
        
        nNans = np.size(np.nonzero(np.isnan(corr_mat_copy.flatten())))
                
        if nNans != 0:
            
            stillNans = True
            
        else:
            
            stillNans = False
            
    nan_inds = nan_inds.astype(int)

    return nan_inds        
        

#%%

def fcn_remove_badCells(corr, remove_cells):
    

    corr_mat_new = corr.copy()
    
    if np.size(remove_cells) > 0:
        
        corr_mat_new = np.delete(corr_mat_new, remove_cells, 0)
        corr_mat_new = np.delete(corr_mat_new, remove_cells, 1)
    
    np.fill_diagonal(corr_mat_new, 1)
    
    return corr_mat_new
    


#%% run hierarchical clustering


def fcn_run_hierarchical_clustering(corr_matrix, linkage_method = 'average'):
    
    corr_matrix = fcn_sym_corrMatrix(corr_matrix)
        
    dissimilarity = fcn_compute_dissimilarity(corr_matrix)
    
    if linkage_method == 'ward':
        Z = ward(squareform(dissimilarity))
    else:
        Z = linkage(squareform(dissimilarity), linkage_method)
        
    return Z


def fcn_compute_dissimilarity(corr_matrix):
        
    corr_matrix = fcn_sym_corrMatrix(corr_matrix)
    
    dissimilarity = 1-corr_matrix
        
    return dissimilarity


def fcn_sym_corrMatrix(corr_matrix):

    corr_matrix = (corr_matrix + np.transpose(corr_matrix))/2
    np.fill_diagonal(corr_matrix,1)
    corr_matrix[np.isnan(corr_matrix)] = 0
    
    return corr_matrix


'''
Z: linkage matrix
t:  if criterion = 'distance', t is the threshold for forming clusters
    if criterion = 'maxclust', t is the maximum number of clusters
fcluster_criterion: distance or maxclust
'''

def fcn_threshold_hierarchical_clustering(Z, t, fcluster_criterion='maxclust'):
    
    if fcluster_criterion not in ['distance', 'maxclust']:
        sys.exit('this function only supports distance or maxclust criteria')
    
    if fcluster_criterion == 'distance':
        labels = fcluster(Z, t, criterion=fcluster_criterion) - 1
    
    if fcluster_criterion == 'maxclust':
        labels = cut_tree(Z, t)[:,0]
    
    return labels


#%% size of clusters

def fcn_sizeClusters(clusterID):
    
    maxCluster = np.max(clusterID).astype(int)
    
    sizeClusters = np.zeros((maxCluster + 1))
    
    for indClu in range(0, maxCluster + 1):

        sizeClusters[indClu] = np.size(np.nonzero(clusterID == indClu))
    
    
    return sizeClusters


#%% contrast function vs threshold


def fcn_optimalPartition_contrast_shuffle(corr_shuffle, Z_shuffle, include_selfCorr):
    
    nShuffles = np.size(Z_shuffle)
    nCells = np.size(corr_shuffle,0)
    
    nClusters_shuffle = np.zeros((nCells, nShuffles))
    Q_shuffle = np.zeros((nCells, nShuffles))
    opt_nClu_shuffle = np.zeros(nShuffles)
    clusterID_hCluster_shuffle = np.zeros((nCells, nShuffles))
    
    for iShuffle in range(0,nShuffles):
        nClusters_shuffle[:,iShuffle], Q_shuffle[:,iShuffle] = fcn_contrast_vs_nClusters(corr_shuffle[:,:,iShuffle], Z_shuffle[iShuffle], include_selfCorr)
        opt_nClu_shuffle[iShuffle] = nClusters_shuffle[np.nanargmax(Q_shuffle[:,iShuffle]), iShuffle]
        clusterID_hCluster_shuffle[:,iShuffle] = fcn_threshold_hierarchical_clustering(Z_shuffle[iShuffle], opt_nClu_shuffle[iShuffle], 'maxclust')
        print(iShuffle)
        
    return nClusters_shuffle, Q_shuffle, clusterID_hCluster_shuffle



def fcn_contrast_vs_nClusters(corr_matrix, Z, include_selfDist):
    
    
    max_nClu = np.size(corr_matrix, 1) 

    contrast = np.ones((max_nClu))*np.nan
    nClusters = np.ones((max_nClu))*np.nan

    for nClu in range(0, max_nClu):
        
        labels = fcn_threshold_hierarchical_clustering(Z, nClu+1, fcluster_criterion = 'maxclust')
    
        nClusters[nClu] = np.size(np.unique(labels))
        
        contrast[nClu] = fcn_contrast(corr_matrix, labels, include_selfDist)
        
        
    
    return nClusters, contrast



def fcn_contrast(corr, labels, include_selfDist):
    
    corr_matrix = corr.copy()

    
    if include_selfDist == True:
        np.fill_diagonal(corr_matrix, 1)
    elif include_selfDist == False:    
        np.fill_diagonal(corr_matrix, 0)
        
    sizeClusters = fcn_sizeClusters(labels)
    nClusters = np.size(np.unique(labels))
    
    within_minus_between_corr = np.ones(nClusters)*np.nan
    
    if nClusters == 1:
        contrast = np.nanmean(np.nanmean(corr_matrix,1))
    
    else:
    
        for indClu in range(0, nClusters):
            
            inClu = np.nonzero(labels == indClu)[0]
            not_inClu = np.nonzero(labels != indClu)[0]
        
            withinClu_corr = corr_matrix[inClu,:][:,inClu].copy()
            betweenClu_corr = corr_matrix[inClu,:][:,not_inClu].copy()
         

            within_minus_between_corr[indClu] = np.nansum( np.nanmean(withinClu_corr, 1) ) - np.nansum( np.nanmean(betweenClu_corr, 1) )
        contrast = np.nansum(within_minus_between_corr)/np.nansum(sizeClusters) 
    
    return contrast


#%% compute statistical significance of extracted clusters

def fcn_compute_sigClusters_againstShuffle(true_corr, true_partition, null_corr, null_partition, quality_metric, include_selfDist, clusterSize_cutoff, sig_level=0.05):
        
    nShuffles = np.size(null_partition, 1)
    nClusters = np.size(np.unique(true_partition))

    nullQuality = np.array([])
    sigClusters = np.array([],dtype=np.int32)

    sizeClusters = fcn_sizeClusters(true_partition)
    goodSize_clusters = np.nonzero(sizeClusters >= clusterSize_cutoff)[0]
    sigLevel_corrected = sig_level/len(goodSize_clusters)
    
    
    if quality_metric == 'contrast':
        
        trueQuality = fcn_clusterContrast(true_corr, true_partition, include_selfDist)


        for indShuffle in range(0, nShuffles):
            
            sizeClusters_shuffle = fcn_sizeClusters(null_partition[:, indShuffle])
            goodSize_clusters_shuffle = np.nonzero(sizeClusters_shuffle >= clusterSize_cutoff)
            
            nullQuality_thisShuffle = fcn_clusterContrast(null_corr[:,:,indShuffle], null_partition[:, indShuffle], include_selfDist)
            nullQuality = np.append(nullQuality, nullQuality_thisShuffle[goodSize_clusters_shuffle])

        min_pval = 1/(len(nullQuality)+1)
        if sigLevel_corrected < min_pval:
            sys.exit('not enough shuffles to obtain requested significance')

        for indCluster in range(0, nClusters):
            if sizeClusters[indCluster] < clusterSize_cutoff:
                pvalue = np.inf
            else:
                pvalue = ( 1 + np.sum(nullQuality >= trueQuality[indCluster]) ) / (len(nullQuality) + 1)
            if pvalue < sigLevel_corrected:
                sigClusters = np.append(sigClusters, indCluster)
    else:

        sys.exit('only implements contrast for stat sig')
        
    sig_cutoff = np.percentile(nullQuality, 100-sigLevel_corrected*100)
    
    return sigClusters, trueQuality, nullQuality, sig_cutoff
               



def fcn_compute_sigClusters_againstPermutation(true_corr, true_partition, nPermutations, quality_metric, include_selfDist, clusterSize_cutoff, sig_level=0.05):
    
    
    nClusters = np.size(np.unique(true_partition))
    
    nullQuality = np.zeros((nClusters, nPermutations))
    sigClusters = np.array([],dtype=np.int32)
    
    sizeClusters = fcn_sizeClusters(true_partition)
    goodSize_clusters = np.nonzero(sizeClusters >= clusterSize_cutoff)[0]
    sigLevel_corrected = sig_level/len(goodSize_clusters)
    
    min_pval = (1/(1+nPermutations))
    if sigLevel_corrected < min_pval:
        sys.exit('not enough shuffles to obtain requested significance')
    
    if quality_metric == 'contrast':
        
        trueQuality = fcn_clusterContrast(true_corr, true_partition, include_selfDist)

        for indPerm in range(0, nPermutations):
            
            null_partition = np.random.permutation(true_partition)
            nullQuality[:, indPerm] = fcn_clusterContrast(true_corr, null_partition, include_selfDist)
            
        for indCluster in range(0, nClusters):
            if sizeClusters[indCluster] < clusterSize_cutoff:
                pvalue = np.inf
            else:
                pvalue = ( 1 + np.sum(nullQuality[indCluster, :] >= trueQuality[indCluster]) ) / (nPermutations + 1)
            if pvalue < sigLevel_corrected:
                sigClusters = np.append(sigClusters, indCluster)

    else:

        sys.exit('only implements contrast for stat sig')

    return sigClusters, trueQuality, nullQuality
    


def fcn_clusterContrast(corr, partition, include_selfDist):
    
    corr_matrix = corr.copy()

    nClusters = np.size(np.unique(partition))
    
    if include_selfDist == True:
        np.fill_diagonal(corr_matrix, 1)
    elif include_selfDist == False:    
        np.fill_diagonal(corr_matrix, 0)
    
    contrast = np.ones(nClusters)*np.nan
    
    
    if nClusters == 1:
        
        contrast[0] = np.mean(np.mean(corr_matrix,1))
    
    else:
        
        for indCluster in range(0, nClusters):
            
            inClu = np.nonzero(partition == indCluster)[0]
            not_inClu = np.nonzero(partition != indCluster)[0]
    
            
            withinCluster_corr = np.mean(corr_matrix[inClu,:][:,inClu], 1)
            betweenCluster_corr = np.mean(corr_matrix[inClu, :][:, not_inClu], 1)
            contrast[indCluster] = np.mean(withinCluster_corr - betweenCluster_corr)

    
    return contrast



#%% statistics

def fcn_oneSided_pval(true_value, null_value, greater_or_less):
    

    null_value = null_value[np.isnan(null_value)==False]
    n_nullValues = np.size(null_value)

    
    if np.isnan(true_value):
        pval = np.nan
    
     
    elif greater_or_less == 'greater':
        
        pval = (1 + np.size( np.nonzero(null_value >= true_value)[0]))/( 1 + n_nullValues) 
        
    elif greater_or_less == 'less':

        pval = (1 + np.size( np.nonzero(null_value <= true_value)[0]))/( 1 + n_nullValues) 
        
    else:
        
        sys.exit()
        
    return pval

