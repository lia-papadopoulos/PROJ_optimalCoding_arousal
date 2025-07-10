
"""
setup functions for clustering
"""

from scipy.cluster.hierarchy import linkage, fcluster, ward, dendrogram, cut_tree
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
import sys


def fcn_get_pupilBins(good_pupilBins, avg_pupilSize_trials):
    
    pupilBins = np.zeros(3)
    allBins = np.array([])
    
    for indPupil in range(0, len(good_pupilBins)):
        
        allBins = np.append(allBins, good_pupilBins[indPupil])
        
    allBins = np.unique(allBins)
    
    
    pupilBins[0] = np.min(allBins)
    pupilBins[1] = allBins[int(np.size(allBins)/2)]
    pupilBins[2] = np.max(allBins)
    pupilBins = pupilBins.astype(int)
    
    pupilSizes = avg_pupilSize_trials[pupilBins].copy()
    
    return pupilBins, pupilSizes
    


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

def fcn_find_badCells(corr, rates, rate_thresh):
    
    ### low rate cells
    lowrate = np.nonzero( rates < rate_thresh )[0]
    
    ### find nan cells
    nanInds = fcn_find_nan_v2(corr)

    ### update lowrate to include nan cells
    all_badCells = np.unique(np.append(lowrate,nanInds)).astype(int)    
    
    return all_badCells


def fcn_remove_badCells(corr, remove_cells):
    

    corr_mat_new = corr.copy()
    
    if np.size(remove_cells) > 0:
        
        corr_mat_new = np.delete(corr_mat_new, remove_cells, 0)
        corr_mat_new = np.delete(corr_mat_new, remove_cells, 1)
    
    np.fill_diagonal(corr_mat_new, 1)
    
    return corr_mat_new
    


def fcn_setup_corrMatrix_eachPupil(good_pupilBins, avg_pupilSize_trials, rates_eachPupil, rate_thresh, corr_eachPupil_raw, corr_allPupil_raw):
    
    # update pupil bins
    pupilBins, pupilSizes = fcn_get_pupilBins(good_pupilBins, avg_pupilSize_trials)
    
    # get relevant correlation matrices
    corr_eachPupil_raw = corr_eachPupil_raw[:,:,pupilBins]
    
    ### low rate cells
    lowrate = np.nonzero(np.any( (rates_eachPupil < rate_thresh), 1))[0]

    ### find nan
    nanInds_pupil = np.array([])

    for count, indPupil in enumerate(pupilBins):
    
        nanInds = fcn_find_nan_v2(corr_eachPupil_raw[:,:,count])
        nanInds_pupil = np.append(nanInds_pupil, nanInds)

    lowrate = np.unique(np.append(lowrate,nanInds_pupil)).astype(int)
    
    
    ### remove nan and low rate cells
    n_goodUnits = np.size(corr_eachPupil_raw,0) - np.size(lowrate)

    corr_eachPupil = np.ones((n_goodUnits, n_goodUnits, len(pupilBins)))*np.nan

    for count, indPupil in enumerate(pupilBins):

        corr_eachPupil[:,:,count] = fcn_remove_badCells(corr_eachPupil_raw[:,:,count], lowrate)


    corr_allPupil = fcn_remove_badCells(corr_allPupil_raw, lowrate)
    
    
    ### return
    return corr_eachPupil, corr_allPupil, pupilBins, pupilSizes, lowrate



#%% run hierarchical clustering

def fcn_sym_corrMatrix(corr_matrix):

    corr_matrix = (corr_matrix + np.transpose(corr_matrix))/2
    np.fill_diagonal(corr_matrix,1)
    corr_matrix[np.isnan(corr_matrix)] = 0
    
    return corr_matrix

def fcn_compute_dissimilarity(corr_matrix):
        
    corr_matrix = fcn_sym_corrMatrix(corr_matrix)
    
    dissimilarity = 1-corr_matrix
        
    return dissimilarity


def fcn_run_hierarchical_clustering(corr_matrix, linkage_method = 'average'):
    
    corr_matrix = fcn_sym_corrMatrix(corr_matrix)
        
    dissimilarity = fcn_compute_dissimilarity(corr_matrix)
    
    if linkage_method == 'ward':
        Z = ward(squareform(dissimilarity))
    else:
        Z = linkage(squareform(dissimilarity), linkage_method)
        
    return Z


#%%
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


#%%
    
def fcn_plot_dendrogram(Z):
    
    fig, axs = plt.subplots(1)
    dendrogram(Z)
    
    return fig, axs



#%% one standard error rule
### error of estimator is standard deviation across multiple estimates

'''
inputs
    true_value:         (n_controlParams,)
    null_vales:         (n_controlParams, nEstimates)
    min_or_max_statistic: 'min' or 'max', whether to maximize or minimize true - null value
    largest_or_smallest_param: 'largest' or 'smallest', whether to find largest or smallest parameter satisfying 1se rule
outputs
    ind_bestParam       index of best parameter according to 1se rule
'''

def fcn_oneSE_rule( true_value, null_values, min_or_max_statistic, largest_or_smallest_param ):
   
    n_controlParams = np.size(null_values, 0)
    nEstimates = np.size(null_values, 1)
    
    statistic = np.zeros((n_controlParams, nEstimates))

    for i in range(0, nEstimates):
        
        statistic[:, i] = true_value - null_values[:, i]
        
    mean_stat = np.nanmean(statistic, 1)
    std_stat = np.nanstd(statistic, 1)*np.sqrt(1+1/nEstimates)
    
    if min_or_max_statistic == 'max':
        indmax_stat = np.nanargmax(mean_stat)
        validInds = np.nonzero(mean_stat >= mean_stat[indmax_stat] - std_stat[indmax_stat])[0]
    elif min_or_max_statistic == 'min':
        indmin_stat = np.nanargmin(mean_stat)
        validInds = np.nonzero(mean_stat <= mean_stat[indmin_stat] + std_stat[indmin_stat])[0]
    else:
        sys.exit()
        
    if largest_or_smallest_param == 'largest':
        ind_bestParam = np.nanmax(validInds)
    elif largest_or_smallest_param == 'smallest':
        ind_bestParam = np.nanmin(validInds)
    else:
        sys.exit()
        
    return ind_bestParam



def fcn_gap_rule( true_value, null_values, min_or_max_statistic, largest_or_smallest_param ):
   
    n_controlParams = np.size(null_values, 0)
    nEstimates = np.size(null_values, 1)
    
    statistic = np.zeros((n_controlParams, nEstimates))

        
    
    if min_or_max_statistic == 'min':
        for i in range(0, nEstimates):
            statistic[:, i] = np.log(null_values[:, i]) - np.log(true_value)
        
    else:
        for i in range(0, nEstimates):
            statistic[:, i] = np.log(true_value) - np.log(null_values[:, i])

                
    mean_stat = np.mean(np.log(statistic), 1)
    std_stat = np.std(np.log(statistic), 1)*np.sqrt(1 + 1/nEstimates)


    validInds = np.nonzero(mean_stat[:-1] >= mean_stat[1:] - std_stat[1:])[0]
    
    print(validInds)

    if np.size(validInds) == 0:
        
        ind_bestParam = 0

    else:
        
        if largest_or_smallest_param == 'largest':
            ind_bestParam = np.nanmax(validInds)
        elif largest_or_smallest_param == 'smallest':
            ind_bestParam = np.nanmin(validInds)
        else:
            sys.exit()
        
    return ind_bestParam


def fcn_globalExtrema_rule(true_value, null_values, min_or_max_statistic):
    
    n_controlParams = np.size(null_values, 0)
    nEstimates = np.size(null_values, 1)
    
    statistic = np.zeros((n_controlParams, nEstimates))

    for i in range(0, nEstimates):
        
        statistic[:, i] = true_value - null_values[:, i]
        
    mean_stat = np.nanmean(statistic, 1)
    
    if min_or_max_statistic == 'max':
        ind_bestParam = np.nanargmax(mean_stat)
    elif min_or_max_statistic == 'min':
        ind_bestParam = np.nanargmin(mean_stat)
    else:
        sys.exit()
        
    return ind_bestParam



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



def fcn_twoSided_pval(true_value, null_value):
    
    n_nullValues = np.size(null_value)
    
    if np.isnan(true_value):
        pval = np.nan
    
    elif np.any(np.isnan(null_value)):
        pval = np.nan
        
    else:
    
        pval = (1 + np.size( np.nonzero(np.abs(null_value) >= np.abs(true_value))[0]))/( 1 + n_nullValues) 
        
    return pval


#%%

def fcn_clusterSizes(partition):
    
    nClusters = np.size(np.unique(partition))
    
    clusterSizes = np.zeros((nClusters))
    
    for iClu in range(0, nClusters):
        
        clusterSizes[iClu] = np.size(np.nonzero(partition == iClu)[0])
        
    return clusterSizes




#%% size of clusters

def fcn_sizeClusters(clusterID):
    
    maxCluster = np.max(clusterID).astype(int)
    
    sizeClusters = np.zeros((maxCluster + 1))
    
    for indClu in range(0, maxCluster + 1):

        sizeClusters[indClu] = np.size(np.nonzero(clusterID == indClu))
    
    
    return sizeClusters




#%% silhouette method



def fcn_silhouette(labels, distances):
    
    nCells = np.size(labels)
    nClusters = np.size(np.unique(labels))
    
    if ( (nClusters > nCells - 1) or (nClusters < 2) ):
        
        silhouette_stat = np.nan
    
    else:
    
        silhouette_stat = silhouette_score(distances, labels, metric='precomputed')
    
    return silhouette_stat



def fcn_silhouette_vs_nClusters(corr_matrix, Z):
    
    
    # distances
    distances = fcn_compute_dissimilarity(corr_matrix)
    
    # max number of clusters
    max_nClu = np.size(corr_matrix, 0)
    
    
    # initialize within cluster distances
    S = np.zeros((max_nClu))
    nClusters = np.zeros((max_nClu))
    
    # loop over number of clusters and compare partition
    for ind_maxClu, n_maxClu in enumerate(np.arange(1, max_nClu + 1)):
        
        # partition
        labels = fcn_threshold_hierarchical_clustering(Z, n_maxClu, fcluster_criterion='maxclust')         
        
        # silhouette     
        S[ind_maxClu] = fcn_silhouette(labels, distances)
  
      
        nClusters[ind_maxClu] = np.size(np.unique(labels))
        
        
    return nClusters, S
        


#%% contrast function vs threshold


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
            #within_minus_between_corr[indClu] = np.nanmean(withinClu_corr) - np.nanmean(betweenClu_corr)
        contrast = np.nansum(within_minus_between_corr)/np.nansum(sizeClusters) 
        #contrast = np.nanmean(within_minus_between_corr)
    
    return contrast
                
            

def fcn_contrast_vs_nClusters(corr_matrix, Z, include_selfDist):
    
    
    max_nClu = np.size(corr_matrix, 1) 

    contrast = np.ones((max_nClu))*np.nan
    nClusters = np.ones((max_nClu))*np.nan

    for nClu in range(0, max_nClu):
        
        labels = fcn_threshold_hierarchical_clustering(Z, nClu+1, fcluster_criterion = 'maxclust')
    
        nClusters[nClu] = np.size(np.unique(labels))
        
        contrast[nClu] = fcn_contrast(corr_matrix, labels, include_selfDist)
        
        
    
    return nClusters, contrast



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
    
            #withinCluster_corr = np.mean(corr_matrix[inClu,:][:,inClu])
            #betweenCluster_corr = np.mean(corr_matrix[inClu, :][:, not_inClu])
            #contrast[indCluster] = withinCluster_corr - betweenCluster_corr
            
            withinCluster_corr = np.mean(corr_matrix[inClu,:][:,inClu], 1)
            betweenCluster_corr = np.mean(corr_matrix[inClu, :][:, not_inClu], 1)
            contrast[indCluster] = np.mean(withinCluster_corr - betweenCluster_corr)

    
    return contrast


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
    
    

#%% within cluster distance v1

def fcn_withinCluster_distance_v1(labels, distances, include_selfDist):
    
    sizeClusters = fcn_sizeClusters(labels)

    
    nClusters = np.size(np.unique(labels))

    # within cluster distances
    norm_sum_withinCluster_dist = np.zeros((nClusters))
    
    for iClu in range(0, nClusters):
        
        inClu = np.nonzero(labels == iClu)[0]
        
        n_inClu = np.size(inClu)

        # within cluster distances
        withinDist = distances[inClu,:][:,inClu].copy()
        
        if include_selfDist == False:

            # size 1 clusters
            if n_inClu <= 1:
                
                norm_sum_withinCluster_dist[iClu] = 0.

                
            else:
                        
                # average within-cluster distance for each unit in cluster, then summed across units
                norm_sum_withinCluster_dist[iClu] = np.sum(withinDist)/(n_inClu-1)
            
        else:
            
            norm_sum_withinCluster_dist[iClu] = np.sum(withinDist)/n_inClu
            

        
    W = np.sum(norm_sum_withinCluster_dist)/np.sum(sizeClusters)
    
    
    return W


#%% within cluster distance v2

def fcn_withinCluster_distance_v2(labels, distances, include_selfDist):
    
    nClusters = np.size(np.unique(labels))
    
    avg_withinCluster_dist = np.zeros((nClusters))
    
    for iClu in range(0, nClusters):
        
        inClu = np.nonzero(labels == iClu)[0]
        
        n_inClu = np.size(inClu)

        # within cluster distances
        withinDist = distances[inClu,:][:,inClu].copy()
        
        if include_selfDist == False:
            
            if n_inClu <= 1:
                
                avg_withinCluster_dist[iClu] = 0.
            
            else:
        
                # don't include diagonal
                np.fill_diagonal(withinDist, np.nan)
                
                avg_withinCluster_dist[iClu] = np.nanmean(withinDist)

        
        else:
        
            avg_withinCluster_dist[iClu] = np.mean(withinDist)
    
    # avg across all clusters
    # if we sum instead of average, then a partition with more high quality 
    # clusters of size > 1 unit could do worse than a partition with fewer,
    # lower quality clusters (in situtation that we ignore clusters of size 1)
    
    W = np.mean(avg_withinCluster_dist)


    return W



#%% within minus between cluster distance 
 

def fcn_within_minus_between_cluster_distance(clusterID, distances, avg_type):
    
    sizeClusters = fcn_sizeClusters(clusterID)

    nClusters = np.size(np.unique(clusterID))
        
    avg_within_between_cluster_dist = np.ones((nClusters))*np.nan
    
    avg_within_cluster_dist = np.ones((nClusters))*np.nan
    avg_between_cluster_dist = np.ones((nClusters))*np.nan


    
    if nClusters == 1:
                        
        inCluster = np.nonzero(clusterID == 0)[0]
        withinCluster_dist = distances[inCluster,:][:,inCluster].copy()
        
        if avg_type == 'v1':
            
            total_within_minus_between_cluster_dist = 0.

        elif avg_type == 'v2':
                    
            total_within_minus_between_cluster_dist = 0.
                    
        return total_within_minus_between_cluster_dist
    
    
    for clu_i in range(0, nClusters):
        
        inCluster = np.nonzero(clusterID == clu_i)[0]
        not_inCluster = np.setdiff1d(np.arange(0,np.size(clusterID)), inCluster)
                    
        not_inCluster = not_inCluster.astype(int)
        
        
        withinCluster_dist = distances[inCluster,:][:,inCluster].copy()
        betweenCluster_dist = distances[inCluster,:][:,not_inCluster].copy()
                
        if avg_type == 'v1':
            
   
            avg_within_between_cluster_dist[clu_i] = np.sum( np.mean(withinCluster_dist, 1) - np.mean(betweenCluster_dist, 1) )
        
        elif avg_type == 'v2':
                    
            if np.size(inCluster) <= 1:
                
                avg_within_cluster_dist[clu_i] = 0.
                avg_between_cluster_dist[clu_i] = np.mean(betweenCluster_dist)
                #avg_within_cluster_dist[clu_i] = 0.
                #avg_between_cluster_dist[clu_i] = 0.             

            else:
                
                # don't include diagonal
                #np.fill_diagonal(withinCluster_dist, np.nan)
                #iu = np.triu_indices(withinCluster_dist)
                #avg_within_cluster_dist[clu_i] = np.nanmean(withinCluster_dist[iu])
                
                avg_within_cluster_dist[clu_i] = np.nanmean(withinCluster_dist)
                avg_between_cluster_dist[clu_i] = np.nanmean(betweenCluster_dist)
        
        else:
            
            sys.exit()
            
    if avg_type == 'v1':
            
        total_within_minus_between_cluster_dist = np.sum(avg_within_between_cluster_dist)/np.sum(sizeClusters)
            
    else:
            
        total_within_minus_between_cluster_dist = np.nanmean(avg_within_cluster_dist) - np.nanmean(avg_between_cluster_dist)


    return total_within_minus_between_cluster_dist




#%%




#%% compare true cluster quality at k clusters to cluster quality of randomized correlation matrix at k clusters

def fcn_withinCluster_distance_vs_nClusters(corr_matrix, Z, withinDist_type, include_selfDist):
    
    
    # distances
    distances = fcn_compute_dissimilarity(corr_matrix)
    
    # max number of clusters
    max_nClu = np.size(corr_matrix, 0)
    
    
    # initialize within cluster distances
    W = np.zeros((max_nClu))
    nClusters = np.zeros((max_nClu))
    
    # loop over number of clusters and compare partition
    for ind_maxClu, n_maxClu in enumerate(np.arange(1, max_nClu + 1)):
        
        # partition
        labels = fcn_threshold_hierarchical_clustering(Z, n_maxClu, fcluster_criterion='maxclust')         
        
        # within cluster distance        
        if withinDist_type == 'v1':
            W[ind_maxClu] = fcn_withinCluster_distance_v1(labels, distances, include_selfDist)
        elif withinDist_type == 'v2':
            W[ind_maxClu] = fcn_withinCluster_distance_v2(labels, distances, include_selfDist)
        else:
            sys.exit('invalid withinDist_type')

        
        nClusters[ind_maxClu] = np.size(np.unique(labels))
        

    return nClusters, W





#%%
    
def fcn_within_between_cluster_distance_vs_nClusters(corr_matrix, Z, withinDist_type):
    
    
    # distances
    distances = fcn_compute_dissimilarity(corr_matrix)
    
    # max number of clusters
    max_nClu = np.size(corr_matrix, 0)
    
    
    # initialize within cluster distances
    W = np.zeros((max_nClu))
    nClusters = np.zeros((max_nClu))
    
    # loop over number of clusters and compare partition
    for ind_maxClu, n_maxClu in enumerate(np.arange(1, max_nClu + 1)):
        
        # partition
        labels = fcn_threshold_hierarchical_clustering(Z, n_maxClu, fcluster_criterion='maxclust')         
        
        # within, between cluster distance        
        W[ind_maxClu] = fcn_within_minus_between_cluster_distance(labels, distances, withinDist_type)
    
        # number of clusters
        nClusters[ind_maxClu] = np.size(np.unique(labels))
        

        
    return nClusters, W








#%%



#%% convert nClusters to threshold


def fcn_nClusters_to_threshold(nClusters_select, nClusters_vs_threshold, threshold_vals):
    
    ind_thresh = np.nonzero(nClusters_vs_threshold == nClusters_select)[0]
    
    if np.size(ind_thresh) == 1:
        
        thresh_use = threshold_vals[ind_thresh]
        print('perfect')
    
    if np.size(ind_thresh) > 0:
        
        ind_thresh = np.min(ind_thresh)
        thresh_use = threshold_vals[ind_thresh]

    else:
        
        ind_thresh = np.argmin(np.abs(nClusters_vs_threshold - nClusters_select))
      
        thresh_use = threshold_vals[ind_thresh]
        
    return thresh_use


    

def fcn_maxClusterSize_vs_threshold(Z, threshold_vals, fcluster_criterion = 'distance', plot=True):
    
    
    nThresh = np.size(threshold_vals)
    
    maxClusterSize_vs_threshold = np.zeros((nThresh))
    
    for iThresh in range(0, nThresh):
        
        thresh = threshold_vals[iThresh]
        
        labels = fcn_threshold_hierarchical_clustering(Z, thresh, fcluster_criterion)
        
        clusterSizes = fcn_clusterSizes(labels)
        
        maxClusterSize_vs_threshold[iThresh] = np.max(clusterSizes)
    
    
    fig, axs = plt.subplots(1)
    ax = axs
    ax.plot(threshold_vals, maxClusterSize_vs_threshold)
    ax.set_xlabel('threshold')
    ax.set_ylabel('max cluster size')
    
    return maxClusterSize_vs_threshold
    


def fcn_nClusters_vs_threshold(Z, threshold_vals, fcluster_criterion = 'distance', plot=True):
    
    nThresh = np.size(threshold_vals)
    
    nClusters_vs_threshold = np.zeros((nThresh))
    
    for iThresh in range(0, nThresh):
        
        thresh = threshold_vals[iThresh]
        
        labels = fcn_threshold_hierarchical_clustering(Z, thresh, fcluster_criterion)
                        
        nClusters_vs_threshold[iThresh] = np.size(np.unique(labels))
        
    if plot == True:
        fig, axs = plt.subplots(1)
        ax = axs
        ax.plot(threshold_vals, nClusters_vs_threshold)
        ax.set_xlabel('threshold')
        ax.set_ylabel('# clusters')
        
    
    return nClusters_vs_threshold



#%% OTHER WAYS OF DETERMINING OPTIMAL CLUSTERS



#%%

#%% stability indices


def fcn_optThresh_partitionStabilityIndex(Z, plot=True):
    
    n_branches = np.size(Z,0)
    n = n_branches + 1
    
    d_of_n = np.zeros((n_branches))
    d_of_n_rev = np.zeros((n_branches))
    
    d_of_n[:] = Z[:,2]
    
    d_of_n_rev[:] = np.flip(d_of_n)
    
        
    log_d_of_n_rev = np.log(d_of_n_rev)
    
    diff_log_d_rev = -np.diff(log_d_of_n_rev)
    
    norm_const = np.log(d_of_n_rev[0]) - np.log(d_of_n_rev[-1])
    
    stabilityIndex = diff_log_d_rev/norm_const
    
    argmax_stabilityIndex = np.argmax(stabilityIndex)
    
    dThresh_high = d_of_n[n-2-argmax_stabilityIndex]
    dThresh_low = d_of_n[n-3-argmax_stabilityIndex]
    
    optThresh = np.mean(np.array([dThresh_high, dThresh_low]))
    
    thresh_vals = np.zeros((len(stabilityIndex)))
    for i in range(0, len(thresh_vals)):
        thresh_vals[i] = np.mean([d_of_n[n-2-i], d_of_n[n-3-i]])
    
    if plot==True:
    
        fig, axs = plt.subplots(1)
        ax = axs
        ax.plot(thresh_vals, stabilityIndex, '-o')
        ax.set_xlabel('threshold')
        ax.set_ylabel('stability index')
        ax.set_xlim([0,1])
    

    return stabilityIndex, optThresh
        



def fcn_optThresh_nCluster_stability(nClusters_vs_threshold, threshold_vals, max_nClusters, plot=True):
    
    
    # check that nClusters decreases with threshold
    diff_nClusters = np.diff(nClusters_vs_threshold)
    
    if np.any(diff_nClusters > 0):
        
        sys.exit('nClusters not a decreasing function of threshold')
    
    unique_nClusters = np.unique(nClusters_vs_threshold)
    num_nClusters = np.size(unique_nClusters)
    
    length_nCluster_partition = np.zeros((num_nClusters))
    
    for ind_nClu in range(0, num_nClusters):
        
        if unique_nClusters[ind_nClu] == max_nClusters:
            
            length_nCluster_partition[ind_nClu] = np.nan
            
        elif unique_nClusters[ind_nClu] == 1:
            
            length_nCluster_partition[ind_nClu] = 0
            
        else:
            
            length_nCluster_partition[ind_nClu] = np.size(np.nonzero(nClusters_vs_threshold == unique_nClusters[ind_nClu])[0])
        
    
    argmax_length_nCluster_partition = np.nanargmax(length_nCluster_partition)
    
    indThresh_longest_nCluster_partition = np.nonzero(nClusters_vs_threshold == unique_nClusters[argmax_length_nCluster_partition])[0]
    thresh_longest_nCluster_partition = threshold_vals[indThresh_longest_nCluster_partition]
    
    optThresh = np.mean(np.array([thresh_longest_nCluster_partition[0], thresh_longest_nCluster_partition[-1]]))
    
    if plot == True:
        fig, axs = plt.subplots(1)
        ax = axs
        ax.plot(threshold_vals, nClusters_vs_threshold, '-o')
        ax.plot([optThresh, optThresh], [0, np.max(unique_nClusters)], '-x', color='red')
        ax.set_xlim([threshold_vals[0], threshold_vals[-1]])
        ax.set_xlabel('threshold')
        ax.set_ylabel('number of clusters')

        plt.close()    

        return nClusters_vs_threshold, optThresh, fig
    
    else:
        
        return nClusters_vs_threshold, optThresh
    


#%%

def fcn_optThresh_withinClusterDistance_true_vs_permutedLabels(corr_matrix, Z, threshold_vals, withinDist_type, include_selfDist, optParam_rule = 'one_SE', nShuffle = 500, plot=True, sig_level = 0.01):
    
    
    distances = fcn_compute_dissimilarity(corr_matrix)

    nThresh = np.size(threshold_vals)

    Wtrue = np.zeros((nThresh))
    Wnull = np.zeros((nThresh, nShuffle))
    Wnull_minus_Wtrue = np.zeros((nThresh, nShuffle))
    pval = np.ones((nThresh))*np.nan

    for iThresh in range(0, nThresh):
        
        thresh = threshold_vals[iThresh]
        
        labels = fcn_threshold_hierarchical_clustering(Z, thresh, fcluster_criterion='distance')
        
        if withinDist_type == 'v1':
            Wtrue[iThresh] = fcn_withinCluster_distance_v1(labels, distances, include_selfDist)

        elif withinDist_type == 'v2':
            Wtrue[iThresh] = fcn_withinCluster_distance_v2(labels, distances, include_selfDist)
        else:
            sys.exit('invalid withinDist_type')
        
        
        for indRand in range(0, nShuffle):
            
            labelsRand = np.random.permutation(labels)
        
            if withinDist_type == 'v1':
                Wnull[iThresh, indRand] = fcn_withinCluster_distance_v1(labelsRand, distances, include_selfDist)                
            elif withinDist_type == 'v2':
                Wnull[iThresh, indRand] = fcn_withinCluster_distance_v2(labelsRand, distances, include_selfDist)
            else:
                sys.exit('invalid withinDist_type')
                
            Wnull_minus_Wtrue[iThresh, indRand] = Wnull[iThresh, indRand] - Wtrue[iThresh]

        pval[iThresh] = fcn_oneSided_pval(Wtrue[iThresh], Wnull[iThresh, :], 'less')

    pval[pval > sig_level] = np.nan

    
    # test statistic
    mean_stat = np.mean(Wnull_minus_Wtrue, axis=1)
    std_stat = np.std(Wnull_minus_Wtrue, axis=1)
    
    null_lowerBound = np.percentile(Wnull, 2.5, axis=1)
    null_upperBound = np.percentile(Wnull, 97.5, axis=1)
    
    if optParam_rule == 'one_SE':
        
        indOptThresh = fcn_oneSE_rule(Wtrue, Wnull, 'min', 'largest')
        
    if optParam_rule == 'gap':
        
        indOptThresh = fcn_gap_rule(Wtrue, Wnull, 'min', 'largest')
        
    elif optParam_rule == 'global_extrema':
        
        indOptThresh = fcn_globalExtrema_rule(Wtrue, Wnull, 'min')
        
    optThresh = threshold_vals[indOptThresh]
    
    if plot == True:
        fig1, axs = plt.subplots(1)
        ax1 = axs
        ax1.plot(threshold_vals, Wtrue, '-o', color='k', label='true')
        ax1.fill_between(threshold_vals, null_lowerBound, null_upperBound, alpha=0.3, color='gray', label='permuted labels')
        ax1.plot([optThresh, optThresh], [np.nanmin(Wtrue), np.nanmax(null_upperBound)], '-x', color='red')
        ax1.set_xlim([threshold_vals[0], threshold_vals[-1]])
        ax1.set_xlabel('threshold')
        ax1.set_ylabel('W')
        ax1.legend()

        fig2, axs = plt.subplots(1)
        ax2 = axs
        ax2.plot(threshold_vals, mean_stat, '-o', color='blue')
        ax2.fill_between(threshold_vals, mean_stat-std_stat, mean_stat + std_stat, alpha=0.3, color='blue')
        ax2.plot([optThresh, optThresh], [np.nanmin(mean_stat), np.nanmax(mean_stat)], '-x', color='red')
        ax2.plot(threshold_vals, pval*np.nanmin(mean_stat), 'o', color='gray')
        ax2.set_xlim([threshold_vals[0], threshold_vals[-1]])
        ax2.set_xlabel('threshold')
        ax2.set_ylabel('Wnull - Wtrue')

        plt.close()
            
        return Wtrue, Wnull, optThresh, pval[indOptThresh], fig1, fig2
    
    else:
        
        return Wtrue, Wnull, optThresh, pval[indOptThresh]



    
def fcn_optThresh_silhouette_true_vs_permutedLabels(corr_matrix, Z, threshold_vals, optParam_rule = 'one_SE', nShuffle = 500, plot=True, sig_level=0.01):
    
    
    distances = fcn_compute_dissimilarity(corr_matrix)

    nThresh = np.size(threshold_vals)

    Strue = np.ones((nThresh))*np.nan
    Snull = np.ones((nThresh, nShuffle))*np.nan
    Strue_minus_Snull = np.ones((nThresh, nShuffle))*np.nan
    pval = np.ones((nThresh))*np.nan

    for iThresh in range(0, nThresh):
        
        thresh = threshold_vals[iThresh]
        
        labels = fcn_threshold_hierarchical_clustering(Z, thresh, fcluster_criterion = 'distance')
        
        Strue[iThresh] = fcn_silhouette(labels, distances)

        for indRand in range(0, nShuffle):
            
            labelsRand = np.random.permutation(labels)
        
            Snull[iThresh, indRand] = fcn_silhouette(labelsRand, distances)              

            Strue_minus_Snull[iThresh, indRand] = Strue[iThresh] - Snull[iThresh, indRand]
            
        pval[iThresh] = fcn_oneSided_pval(Strue[iThresh], Snull[iThresh, :], 'greater')

    
    pval[pval > sig_level] = np.nan
    
    mean_null_minus_true = np.mean(Strue_minus_Snull, axis=1)
    std_null_minus_true = np.std(Strue_minus_Snull, axis=1)

    null_lowerBound = np.percentile(Snull, 2.5, axis=1)
    null_upperBound = np.percentile(Snull, 97.5, axis=1)
    
    if optParam_rule == 'one_SE':
        
        indOptThresh = fcn_oneSE_rule(Strue, Snull, 'max', 'largest')
        
    elif optParam_rule == 'gap':
        
        indOptThresh = fcn_gap_rule(Strue, Snull, 'max', 'largest')
        
    elif optParam_rule == 'global_extrema':
        
        indOptThresh = fcn_globalExtrema_rule(Strue, Snull, 'max')
    
    optThresh = threshold_vals[indOptThresh]
    

    
    if plot == True:
        fig1, axs = plt.subplots(1)
        ax1 = axs
        ax1.plot(threshold_vals, Strue, '-o', color='k', label='true')
        ax1.fill_between(threshold_vals, null_lowerBound, null_upperBound, alpha=0.3, color='gray', label='permuted labels')
        ax1.plot([optThresh, optThresh], [np.nanmin(null_lowerBound), np.nanmax(Strue)], '-x', color='red')
        ax1.set_xlim([threshold_vals[0], threshold_vals[-1]])
        ax1.set_xlabel('threshold')
        ax1.set_ylabel('silhouette score')
        ax1.legend()

        fig2, axs = plt.subplots(1)
        ax2 = axs
        ax2.plot(threshold_vals, mean_null_minus_true, '-o', color='blue')
        ax2.fill_between(threshold_vals, mean_null_minus_true-std_null_minus_true, mean_null_minus_true + std_null_minus_true, alpha=0.3, color='blue')
        ax2.plot([optThresh, optThresh], [np.nanmin(mean_null_minus_true), np.nanmax(mean_null_minus_true)], '-x', color='red')
        ax2.plot(threshold_vals, pval*np.nanmin(mean_null_minus_true), 'o', color='gray')
        ax2.set_xlim([threshold_vals[0], threshold_vals[-1]])
        ax2.set_xlabel('threshold')
        ax2.set_ylabel('Strue - Snull')
     
            
        return Strue, Snull, optThresh, pval[indOptThresh], fig1, fig2
    
    else:
        
        return Strue, Snull, optThresh, pval[indOptThresh]






#%% compute quality metric vs number of clusters for true and shuffle data


def fcn_clusterQuality_vs_nClusters_true_shuffled(corr_true, Ztrue, corr_shuffle, Zshuffle, \
                                                  quality_metric, optParam_rule, plot_results, sig_level, \
                                                  withinDist_type = 'v1', include_selfDist = True):
    
    nCells = np.size(corr_true,0)
    
    nShuffles = np.size(Zshuffle)        
    quality_null = np.ones((nShuffles), dtype='object')*np.nan
    nClusters_null = np.zeros((nShuffles), dtype='object')
    
    if quality_metric == 'contrast':
        
        min_or_max = 'max'
        greater_or_less = 'greater'
        nClusters_true, quality_true = fcn_contrast_vs_nClusters(corr_true, Ztrue, include_selfDist)
        
        for indShuf in range(0, nShuffles):
            nClusters_null[indShuf], quality_null[indShuf] = fcn_contrast_vs_nClusters(corr_shuffle[:,:,indShuf], Zshuffle[indShuf], include_selfDist)
            print(indShuf)
        
    elif quality_metric == 'silhouette':
    
        min_or_max = 'max'
        greater_or_less = 'greater'
        nClusters_true, quality_true = fcn_silhouette_vs_nClusters(corr_true, Ztrue)

        for indShuf in range(0, nShuffles):
            nClusters_null[indShuf], quality_null[indShuf] = fcn_silhouette_vs_nClusters(corr_shuffle[:,:,indShuf], Zshuffle[indShuf])

        
    elif quality_metric == 'withinDist':
        
        min_or_max = 'min'
        greater_or_less = 'less'
        nClusters_true, quality_true = fcn_withinCluster_distance_vs_nClusters(corr_true, Ztrue, withinDist_type, include_selfDist)
        
        for indShuf in range(0, nShuffles):
            nClusters_null[indShuf], quality_null[indShuf] = fcn_withinCluster_distance_vs_nClusters(corr_shuffle[:,:,indShuf], Zshuffle[indShuf], withinDist_type, include_selfDist)

    elif quality_metric == 'within_between_dist':
        
        min_or_max = 'min'
        greater_or_less = 'less'
        nClusters_true, quality_true = fcn_within_between_cluster_distance_vs_nClusters(corr_true, Ztrue, withinDist_type)
        
        for indShuf in range(0, nShuffles):
            nClusters_null[indShuf], quality_null[indShuf] = fcn_within_between_cluster_distance_vs_nClusters(corr_shuffle[:,:,indShuf], Zshuffle[indShuf], withinDist_type)
            print(indShuf)

        
    else:
        
        sys.exit('unknown quality metric')
    
    
    # average metric across each possible number of clusters
    nClusters_inPartition = np.arange(1, nCells+1, 1)
    quality_true_updated = np.ones((len(nClusters_inPartition)))*np.nan
    quality_null_updated = np.ones((len(nClusters_inPartition),nShuffles))*np.nan    
    pval = np.ones((len(nClusters_inPartition)))*np.nan

    
    for indCluster in range(0, len(nClusters_inPartition)):
        
        indTrue_clu = np.nonzero(nClusters_true == indCluster+1)[0]
        if np.size(indTrue_clu) == 0:
            quality_true_updated[indCluster] = np.nan
        else:
            quality_true_updated[indCluster] = np.mean(quality_true[indTrue_clu])
        
        
        for indShuffle in range(0, nShuffles):

            indNull_clu = np.nonzero(nClusters_null[indShuffle] == indCluster+1)[0]
            if np.size(indNull_clu) == 0:
                quality_null_updated[indCluster, indShuffle] = np.nan
            else:
                quality_null_updated[indCluster, indShuffle] = np.mean(quality_null[indShuffle][indNull_clu])
    

        pval[indCluster] = fcn_oneSided_pval(quality_true_updated[indCluster], quality_null_updated[indCluster, :], greater_or_less)

    pval[pval > sig_level] = np.nan


    true_minus_null = np.zeros((len(nClusters_inPartition),nShuffles))
    for indShuf in range(0, nShuffles):
        true_minus_null[:, indShuf] = quality_true_updated - quality_null_updated[:, indShuf]

    mean_stat = np.nanmean(true_minus_null, axis=1)
    std_stat = np.nanstd(true_minus_null, axis=1)
    lowerBound_null = np.nanpercentile(quality_null_updated, 2.5, axis=1)
    upperBound_null = np.nanpercentile(quality_null_updated, 97.5, axis=1)  

    if optParam_rule == 'one_SE':
        
        indOpt = fcn_oneSE_rule(quality_true_updated, quality_null_updated, min_or_max, 'smallest')
        opt_nClu = nClusters_inPartition[indOpt]
        
    elif optParam_rule == 'gap':
        
        indOpt = fcn_gap_rule(quality_true_updated, quality_null_updated, min_or_max, 'smallest')
        opt_nClu = nClusters_inPartition[indOpt]
        
    elif optParam_rule == 'global_extrema':
        
        indOpt = fcn_globalExtrema_rule(quality_true_updated, quality_null_updated, min_or_max)
        opt_nClu = nClusters_inPartition[indOpt]
        
    elif optParam_rule == 'global_extrema_raw':
        if min_or_max == 'min':
            indOpt = np.nanargmin(quality_true_updated)
        elif min_or_max == 'max':
            indOpt = np.nanargmax(quality_true_updated)
        else:
            sys.exit()
        opt_nClu = nClusters_true[indOpt]

    else:
        sys.exit()

    
    
    if plot_results == True:

        fig1, ax1 = plt.subplots(1,1, figsize=(3,3))
        ax1.fill_between(nClusters_inPartition, lowerBound_null, upperBound_null, color='gray', alpha=0.3, label='shuffle corr')
        ax1.plot(nClusters_true, quality_true_updated, color='k', label='true')
        ax1.plot([opt_nClu, opt_nClu], [np.nanmin(lowerBound_null), np.nanmax(quality_true_updated)], '-x', color='red')
        ax1.set_xlabel('# clusters')
        ax1.set_ylabel('quality')       
        ax1.legend()
        plt.tight_layout()
        
        fig2, ax2 = plt.subplots(1,1, figsize=(3,3))
        ax2.plot(nClusters_inPartition, mean_stat, '-o', color='blue')
        ax2.fill_between(nClusters_inPartition, mean_stat-std_stat, mean_stat + std_stat, alpha=0.3, color='blue')
        ax2.plot([opt_nClu, opt_nClu], [np.nanmin(mean_stat), np.nanmax(mean_stat)], '-x', color='red')
        ax2.plot(nClusters_inPartition, pval*np.nanmin(mean_stat), 'o', color='gray')
        ax2.set_xlabel('# clusters')
        ax2.set_ylabel('true - null') 
        
        plt.tight_layout()
    
        return  quality_true_updated, quality_null_updated, opt_nClu, nClusters_inPartition, pval[indOpt], fig1, fig2
    
    else:
        
        return quality_true_updated, quality_null_updated, opt_nClu, nClusters_inPartition, pval[indOpt]
    
    
    