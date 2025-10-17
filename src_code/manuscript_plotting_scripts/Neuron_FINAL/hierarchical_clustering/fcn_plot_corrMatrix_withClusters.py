
import numpy as np
import matplotlib.pyplot as plt

def fcn_plot_corrMatrix_withClusters(ax, corr_matrix, clusterID, clustersHighlight, plotColor, cbar_label, cmap, cbar_lim, linewidth=0.5, noTicks=False, shrink=0.8):
    
    nCells = np.size(corr_matrix,0)
    
    sorted_inds = np.argsort(clusterID)
    
    corr_mat_sort = np.zeros(np.shape(corr_matrix))
    corr_mat_sort[:,:] = corr_matrix[sorted_inds, :][:,sorted_inds]
    
    if np.size(clustersHighlight) == 1:
        clustersHighlight = np.array([clustersHighlight])
    

    if np.size(clustersHighlight) != 0:

        min_indx = np.zeros(len(clustersHighlight))
        max_indx = np.zeros(len(clustersHighlight))        
    
        for indClu, clu in enumerate(clustersHighlight):
            
            cells_inClu = np.nonzero(clusterID == clu)[0]
            
            sortedInd_inClu = np.zeros(len(cells_inClu))
            
            for indx, cell in enumerate(cells_inClu):
            
                sortedInd_inClu[indx] = np.nonzero(sorted_inds == cell)[0][0]
                
            min_indx[indClu] = np.min(sortedInd_inClu) - 0.5 + 1
            max_indx[indClu] = np.max(sortedInd_inClu) + 0.5 + 1
            
    
    x = np.arange(1,nCells+1)
    y = np.arange(1,nCells+1)

    ticks = [1, nCells]
    if noTicks:
        ticks=np.array([])
    
    if cbar_lim == 'corr_based':
        f = ax.pcolormesh(x, y, corr_mat_sort, shading='nearest', vmin=-np.nanmax(np.abs(corr_mat_sort)), vmax=np.nanmax(np.abs(corr_mat_sort)), cmap=cmap)
        cbar = plt.colorbar(f, ax=ax, shrink = shrink, label=cbar_label)
    else:
        f = ax.pcolormesh(x, y, corr_mat_sort, shading='nearest', vmin=-cbar_lim, vmax=cbar_lim, cmap=cmap)   
        cbar = plt.colorbar(f, ax=ax, shrink = shrink, label=cbar_label, ticks = [-cbar_lim, 0, cbar_lim])
        cbar.ax.set_yticklabels([('< %0.1f' % -cbar_lim), '0', ('> %0.1f' % cbar_lim)], fontsize=6) 

    for indClu in range(0, len(clustersHighlight)):
        ax.plot([min_indx[indClu], min_indx[indClu]], [min_indx[indClu],max_indx[indClu]], color=plotColor, linewidth=linewidth)
        ax.plot([max_indx[indClu], max_indx[indClu]], [min_indx[indClu],max_indx[indClu]], color=plotColor, linewidth=linewidth)
        ax.plot([min_indx[indClu], max_indx[indClu]], [min_indx[indClu],min_indx[indClu]], color=plotColor, linewidth=linewidth)
        ax.plot([min_indx[indClu], max_indx[indClu]], [max_indx[indClu],max_indx[indClu]], color=plotColor, linewidth=linewidth)
    
    ax.set_aspect('equal','box')
    ax.set_ylim(y.max()+0.5, y.min()-0.5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    
    ax.set_xlabel('cells', labelpad = 1)
    ax.set_ylabel('cells', labelpad = 1)
