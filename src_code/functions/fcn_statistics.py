
"""
Created on Sat Jun 26 12:19:41 2021

@author: liapapadopoulos
"""

#%%

import numpy as np
import sys
import scipy.stats
import statsmodels.stats.descriptivestats


def fcn_partialCorr(x, y, z, method='Spearman'):
    
    if method == 'Spearman':
    
        res = scipy.stats.spearmanr(x, y)
        rxy = res.statistic
        
        res = scipy.stats.spearmanr(x, z)
        rxz = res.statistic
        
        res = scipy.stats.spearmanr(y, z)
        ryz = res.statistic
    
    elif method == 'Pearson':
        
        res = scipy.stats.spearmanr(x, y)
        rxy = res.statistic
        
        res = scipy.stats.spearmanr(x, z)
        rxz = res.statistic
        
        res = scipy.stats.spearmanr(y, z)
        ryz = res.statistic
        
    else:
        sys.exit()
        
    partial_corr = (rxy - rxz*ryz)/np.sqrt( (1-rxz**2)*(1-ryz**2) )

    return partial_corr


#%% CODE FOR IMPLEMENTING STATISTICAL TESTS

def fcn_2sample_paired_permTest(diff_vec, nPerms, test_stat):
    
    # difference vector
    diff_vec = diff_vec[~np.isnan(diff_vec)]
    
    # n samples
    nSamples = np.size(diff_vec)
    
    perm_diff = np.zeros(nPerms)    
    
    # compute actual test statistic
    
    if test_stat == 'mean':
        
        true_diff = np.mean(diff_vec)
        
    elif test_stat == 'median':
        
        true_diff = np.median(diff_vec)
    
    else:
        
        sys.exit('error: unknown test statistic given')    
        
        
    # permutations
    for indPerm in range(0, nPerms, 1):
        
        # randomly flip sign of differenc vectors
        flipVec = np.random.choice([-1,1], nSamples)
                       
        if test_stat == 'mean':
            
            perm_diff[indPerm] = np.mean(diff_vec*flipVec)
            
        elif test_stat == 'median':
        
            perm_diff[indPerm] = np.median(diff_vec*flipVec)
            
        else:
            
            sys.exit('error: unknown test statistic given')
            
    
    # pvalue (2-sided only)
    pVal = np.size(np.where(np.abs(perm_diff)>=np.abs(true_diff))[0])/nPerms
    
    # return        
    return true_diff, pVal
    
    
    

def fcn_2sample_ind_permTest(x, y, nPerms, test_stat):
    
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    
    all_data = np.concatenate((x, y))
    
    nx = np.size(x)
                
    perm_diff = np.zeros(nPerms)
    
    # compute actual test statistic
    
    if test_stat == 'mean':
        
        true_diff = np.mean(x) - np.mean(y)
        
    elif test_stat == 'median':
        
        true_diff = np.median(x) - np.median(y)
    
    else:
        
        sys.exit('error: unknown test statistic given')
        

    
    for indPerm in range(0, nPerms, 1):
        
        perm_all_data = np.random.permutation(all_data)
        xperm = perm_all_data[:nx]
        yperm = perm_all_data[nx:]
        
        if test_stat == 'mean':
        
            perm_diff[indPerm] = np.mean(xperm) - np.mean(yperm)
            
        elif test_stat == 'median':
            
            perm_diff[indPerm] = np.median(xperm) - np.median(yperm)
            
        else:
            
            sys.exit('error: unknown test statistic given')
            
    # pvalue (2-sided only)
    pVal = np.size(np.where(np.abs(perm_diff)>=np.abs(true_diff))[0])/nPerms
            
    return true_diff, pVal
            
 #%%   
def fcn_MannWhitney_twoSided(x, y):
    
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
        
    if ( (np.size(x) == 0) or (np.size(y) == 0) ):
          
          test_stat = np.nan
          pVal = np.nan
          
    elif np.array_equal(x, y):
        
        test_stat = 0.
        pVal = 1.

    else:
    
        Ux, pVal = scipy.stats.mannwhitneyu(x, y, alternative='two-sided')   
        Uy = np.size(x)*np.size(y)-Ux
        test_stat = Ux - Uy
    
    return test_stat, pVal



def fcn_Wilcoxon(x, y):
    
    results_dict = {}
    
    nanVals = np.nonzero((np.isnan(x)) | np.isnan(y))
    x = np.delete(x, nanVals)
    y = np.delete(y, nanVals)
    
    if (np.size(x) == 0):
        stat_2sided = np.nan
        pVal_2sided = np.nan
        stat_1sided_xGreater = np.nan
        pVal_1sided_xGreater = np.nan
        stat_1sided_xSmaller = np.nan
        pVal_1sided_xSmaller = np.nan
        
    elif np.array_equal(x, y):
        stat_2sided = np.nan
        pVal_2sided = 1.
        stat_1sided_xGreater = np.nan
        pVal_1sided_xGreater = 1.
        stat_1sided_xSmaller = np.nan
        pVal_1sided_xSmaller = 1.
        
    else:
    
        # 2 sided test to look for a statistical difference
        stat_2sided, pVal_2sided = scipy.stats.wilcoxon(x,y,alternative='two-sided', correction=False)
        
        # 1 sided test to check if x > y
        stat_1sided_xGreater, pVal_1sided_xGreater = scipy.stats.wilcoxon(x,y,alternative='greater', correction=False)
        stat_1sided_xSmaller, pVal_1sided_xSmaller = scipy.stats.wilcoxon(x,y,alternative='less', correction=False)

    # results_dict
    results_dict['sampleSize'] = np.size(x)
    results_dict['stat_2sided'] = stat_2sided
    results_dict['pVal_2sided'] = pVal_2sided
    results_dict['stat_1sided_xGreater'] = stat_1sided_xGreater
    results_dict['pVal_1sided_xGreater'] = pVal_1sided_xGreater
    results_dict['stat_1sided_xSmaller'] = stat_1sided_xSmaller
    results_dict['pVal_1sided_xSmaller'] = pVal_1sided_xSmaller
       
    return results_dict



def fcn_SignTest_twoSided(x, y):
    
    nanVals = np.nonzero((np.isnan(x)) | np.isnan(y))
    x = np.delete(x, nanVals)
    y = np.delete(y, nanVals)
    
    sample = x - y
    test_stat, pVal = statsmodels.stats.descriptivestats.sign_test(sample)

    return test_stat, pVal


def fcn_paired_tTest_twoSided(x, y):
    
    nanVals = np.nonzero((np.isnan(x)) | np.isnan(y))
    x = np.delete(x, nanVals)
    y = np.delete(y, nanVals)
        
    test_stat, pVal = scipy.stats.ttest_rel(x,y,alternative='two-sided',nan_policy='omit')
    return test_stat, pVal


def fcn_2samp_tTest_twoSided(x, y):
    
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    
    test_stat, pVal = scipy.stats.ttest_ind(x,y, equal_var=False, alternative='two-sided')
    return test_stat, pVal


#%% min max normalize a quantity

def fcn_minmax_norm(x):
    
    norm_x = (x-np.min(x))/(np.max(x)-np.min(x))
    
    return norm_x


#%% min max normalize a quantity

def fcn_zscore(x):
    
    norm_x = (x-np.mean(x))/np.std(x)
    
    return norm_x


#%% normalize as percent change relative to maximum

def fcn_pctChange_max(x):
    
    norm_x = ((x-np.max(x))/np.max(x))*100

    return norm_x



#%% normalize as percent change relative to mean

def fcn_pctChange_mean(x):
    
    norm_x = ((x-np.mean(x))/np.mean(x))*100

    return norm_x
