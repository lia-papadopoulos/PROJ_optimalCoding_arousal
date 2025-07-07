
"""
SET OF FUNCTIONS TO AID IN DECODING ANALYSIS
Would be better to make a class so we aren't passing so many variables around
"""

# standard imports
import numpy as np
import sys

# decoding package
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


#%% CROSS VALIDATION 


#%% 

# STRATIFIED K-FOLD CROSS VALIDATION
def fcn_stratified_kFold_crossVal(X, classLabels, classifier, nKFolds, \
                                  lda_solver, compute_shuffleDist, nShuffles, shuffle_percentile):

    # number of classes
    nClasses = np.size(np.unique(classLabels))
    
    # number of samples
    nSamples = np.size(classLabels)
    inds = np.arange(0,nSamples,1)
    
    # initialize arrays to output
    accuracy = np.zeros(nKFolds)
    confusion_mat = np.zeros((nClasses, nClasses, nKFolds))*np.nan

    
    # initialize class that generates splits
    k_folds = StratifiedKFold(n_splits=nKFolds, shuffle=True)
    
    # loop over folds
    for foldInd, (trainInds, testInds) in enumerate(k_folds.split(X, classLabels)):
        
        # decoding of data
        confusion_mat[:,:,foldInd], accuracy[foldInd] = fcn_decode(X, classLabels, \
                                                                   classifier, \
                                                                   trainInds, testInds, \
                                                                   lda_solver)


    # compute averages over folds
    splitAvg_accuracy = np.mean(accuracy)
    splitAvg_confusionMat = np.mean(confusion_mat,axis=2)
    
    

    # shuffle distribution of accuracy            
    if (compute_shuffleDist == True):
        
        # accuracy for each shuffle
        accuracy_shuf = np.zeros(nShuffles)
        p_accuracy = np.zeros(nKFolds)*np.nan
               
        # loop over shuffles and compute one estimate of decoding accuracy
        for shuffInd in range(0, nShuffles, 1):
        
            # shuffle the classLabels
            classLabels_shuffle = np.random.permutation(classLabels)
            
            # get random training/testing indices
            trainInds, testInds = train_test_split(inds, test_size=1/nKFolds, \
                                                   shuffle=True, \
                                                   stratify = classLabels_shuffle)
            
        
            # decoding of data
            _, accuracy_shuf[shuffInd] = fcn_decode(X, classLabels_shuffle, \
                                                    classifier, \
                                                    trainInds, testInds, \
                                                    lda_solver)
                
            
        # compute significance of each true accuracy value
        for splitInd in range(0, nKFolds, 1):
                      
            p_accuracy[splitInd], \
            mean_accuracy_shuffle, \
            lowPercentile_accuracy_shuffle, \
            highPercentile_accuracy_shuffle, sd_accuracy_shuffle = \
            fcn_decodeAccuracy_significance(accuracy[splitInd], \
                                            accuracy_shuf, \
                                            shuffle_percentile)  
                
            
    
        # compute average p-value over splits
        splitAvg_p_accuracy = np.mean(p_accuracy)
        
    
    else:
        
        splitAvg_p_accuracy = np.nan
        mean_accuracy_shuffle = np.nan
        lowPercentile_accuracy_shuffle = np.nan
        highPercentile_accuracy_shuffle = np.nan
        sd_accuracy_shuffle = np.nan
        
                 
           
    # return
    return splitAvg_accuracy, mean_accuracy_shuffle, sd_accuracy_shuffle, \
           lowPercentile_accuracy_shuffle, highPercentile_accuracy_shuffle, \
           splitAvg_p_accuracy, splitAvg_confusionMat


#%%           

# LEAVE ONE OUT CROSS VALIDATION
# SHUFFLE DISTRIBUTION TAKES FOREVER HERE -- DO NOT RECOMMEND

def fcn_LeaveOneOut_crossVal(X, classLabels, classifier, \
                            compute_shuffleDist, nShuffles, shuffle_percentile, lda_solver):           
     
   
    # number of classes
    nClasses = np.size(np.unique(classLabels))
    
    # initialize loo class
    loo = LeaveOneOut()
    
    # number of splits
    nSplits = loo.get_n_splits(X)
    
    # initialize arrays to output
    accuracy = np.zeros(nSplits)
    foldAvg_accuracy_shuf = np.zeros(nShuffles)*np.nan    
    confusion_mat = np.zeros((nClasses, nClasses, nSplits))*np.nan
    
        
    # loop over splits
    for foldInd, (trainInds, testInds) in enumerate(loo.split(X, classLabels)):
        
        # decoding of data
        confusion_mat[:,:,foldInd], accuracy[foldInd] = fcn_decode(X, classLabels, \
                                                               classifier, \
                                                               trainInds, testInds, \
                                                               lda_solver)
            
    # compute averages over folds
    foldAvg_accuracy = np.mean(accuracy)
    foldAvg_confusionMat = np.mean(confusion_mat,axis=2)

     
    # shuffle distribution of accuracy            
    if (compute_shuffleDist == True):
        
        # loop over shuffles
        for shuffInd in range(0, nShuffles, 1):
            
            # shuffle accuracy of each split
            accuracy_shuf = np.zeros(nSplits)
        
            # shuffle the classLabels
            classLabels_shuffle = np.random.permutation(classLabels)
            
            # loop over splits
            for foldInd, (trainInds, testInds) in enumerate(loo.split(X, classLabels_shuffle)):
        
                # decoding of data
               _, accuracy_shuf[foldInd] = fcn_decode(X, classLabels_shuffle, \
                                                       classifier, \
                                                       trainInds, testInds, \
                                                       lda_solver)
                    
            # average across folds
            foldAvg_accuracy_shuf[shuffInd] = np.mean(accuracy_shuf)
            
            
        # compute significance of true accuracy
        p_foldAvg_accuracy, \
        mean_foldAvg_accuracy_shuffle, \
        lowPercentile_foldAvg_accuracy_shuffle, \
        highPercentile_foldAvg_accuracy_shuffle, sd_accuracy_shuffle = \
            fcn_decodeAccuracy_significance(foldAvg_accuracy, \
                                            foldAvg_accuracy_shuf, \
                                            shuffle_percentile)
                
    else:
        
        p_foldAvg_accuracy = np.nan
        mean_foldAvg_accuracy_shuffle = np.nan
        lowPercentile_foldAvg_accuracy_shuffle = np.nan
        highPercentile_foldAvg_accuracy_shuffle = np.nan  
        sd_accuracy_shuffle = np.nan
           
    # return
    return foldAvg_accuracy, mean_foldAvg_accuracy_shuffle, sd_accuracy_shuffle, \
           lowPercentile_foldAvg_accuracy_shuffle, \
           highPercentile_foldAvg_accuracy_shuffle, p_foldAvg_accuracy, \
           foldAvg_confusionMat
    
           
#%%    
# REPEATED STRATIFIED K-FOLD CROSS VALIDATION
def fcn_repeated_stratified_kFold_crossVal(X, classLabels, classifier, \
                                           nKFolds, nReps, \
                                           compute_shuffleDist, nShuffles, \
                                           shuffle_percentile, lda_solver):
    
    
    # number of classes
    nClasses = np.size(np.unique(classLabels))
    
    # number of samples
    nSamples = np.size(classLabels)
    inds = np.arange(0,nSamples,1)

    # initialize class that generates splits
    rskf = RepeatedStratifiedKFold(n_splits=nKFolds, n_repeats = nReps)
    
    # total number of splits
    nSplits = rskf.get_n_splits(X,classLabels)
        
    # initialize arrays
    accuracy = np.zeros(nSplits) # accuracy for each split
    confusion_mat = np.zeros((nClasses, nClasses, nSplits))*np.nan
     
    # loop over splits into training/testing sets
    for splitInd, (trainInds, testInds) in enumerate(rskf.split(X, classLabels)):
        
        # decoding of data
        confusion_mat[:,:,splitInd], accuracy[splitInd] = fcn_decode(X, classLabels, classifier, \
                                                                     trainInds, testInds, \
                                                                     lda_solver) 

               
    # compute average over splits
    splitAvg_accuracy = np.mean(accuracy)
    foldAvg_confusionMat = np.mean(confusion_mat,axis=2)
    
    
    # shuffle distribution of accuracy            
    if (compute_shuffleDist == True):
        
        # accuracy for each shuffle
        accuracy_shuf = np.zeros(nShuffles)
        p_accuracy = np.zeros(nSplits)
               
        # loop over shuffles and compute one estimate of decoding accuracy
        for shuffInd in range(0, nShuffles, 1):
        
            # shuffle the classLabels
            classLabels_shuffle = np.random.permutation(classLabels)
            
            # get random training/testing indices
            trainInds, testInds = train_test_split(inds, test_size=1/nKFolds, \
                                                   shuffle=True, \
                                                   stratify = classLabels_shuffle)
            
        
            # decoding of data
            _, accuracy_shuf[shuffInd] = fcn_decode(X, classLabels_shuffle, \
                                                    classifier, \
                                                    trainInds, testInds, \
                                                    lda_solver)
                
            
        # compute significance of each true accuracy value
        for splitInd in range(0, nSplits, 1):
                      
            p_accuracy[splitInd], \
            mean_accuracy_shuffle, \
            lowPercentile_accuracy_shuffle, \
            highPercentile_accuracy_shuffle, sd_accuracy_shuffle = \
            fcn_decodeAccuracy_significance(accuracy[splitInd], \
                                            accuracy_shuf, \
                                            shuffle_percentile)  
                
            
    
        # compute average p-value over splits
        splitAvg_p_accuracy = np.mean(p_accuracy)
        
    
    else:
        
        splitAvg_p_accuracy = np.nan
        mean_accuracy_shuffle = np.nan
        lowPercentile_accuracy_shuffle = np.nan
        highPercentile_accuracy_shuffle = np.nan
        sd_accuracy_shuffle = np.nan
        
        
    # return
    return splitAvg_accuracy, mean_accuracy_shuffle, sd_accuracy_shuffle, \
           lowPercentile_accuracy_shuffle, highPercentile_accuracy_shuffle, \
           splitAvg_p_accuracy, foldAvg_confusionMat
    
               
#%% DECODE TRUE DATA
    
# done for each fold
def fcn_decode(X, classLabels, classifier, trainInds, testInds, \
               lda_solver, \
               penalty_linSVC = 'l2', C_linSVC=0.1, loss_linSVC='squared_hinge', dual_linSVC=False, standardize=False, tol_lda=1e-8, \
               tol_linSVC=1e-4, maxiter_linSVC=1000):
    
    
    #---------------TRAINING AND TEST DATA------------------------------------#
    
    # training set + labels
    X_train = X[trainInds,:].copy()
    classLabels_train = classLabels[trainInds].copy()
    
    # testing set + labels
    X_test = X[testInds,:].copy()
    classLabels_test = classLabels[testInds].copy()  
    
    
    #---------------TRAIN/TEST ON TRUE DATA-----------------------------------#
    
    # initialize classifier
    if classifier == 'LDA':          
        if lda_solver=='lsqr':
            clf = LinearDiscriminantAnalysis(solver=lda_solver, shrinkage='auto')
        elif lda_solver == 'svd':
            clf = LinearDiscriminantAnalysis(solver=lda_solver, tol=tol_lda)

    elif classifier == 'LinearSVC':
        clf = LinearSVC(penalty=penalty_linSVC, C=C_linSVC, loss=loss_linSVC, dual=dual_linSVC, tol=tol_linSVC, max_iter=maxiter_linSVC)
        if standardize == True:
            scaler=preprocessing.StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
    else:      
        sys.exit('only LDA & LinearSVC classification supported for now.')
    
    # fit model to training data
    clf.fit(X_train, classLabels_train)
                  
    # predict labels of test data
    classLabels_predict = clf.predict(X_test)
        
    # compute accuracy
    accuracy = accuracy_score(classLabels_test, classLabels_predict)
            
    # compute confusion matrix
    confuse_mat = confusion_matrix(classLabels_test, classLabels_predict)
    
    # normalize
    confuse_mat = confuse_mat.astype('float') / confuse_mat.sum(axis=1)[:, np.newaxis]
    
                  
    # return
    return confuse_mat, accuracy


#%% DECODING WRAPPER FUNCTION
# ****** NEED TO MAKE lda and svc keyword args inputs to cross val functions [perhaps as dictionary] ******
# ****** currently they aren't used here; only values defined in fcn_decode matter ****

def fcn_decode_master(X, classLabels, classifier, crossVal_type, \
                     compute_shuffleDist, nShuffles, shuffle_percentile, \
                     lda_solver, \
                     nKFolds=5, nReps=10, shuffle_type='per_split'):
    
    
    if crossVal_type == 'fcn_repeated_stratified_kFold_crossVal':
        
        accuracy, mean_accuracy_shuffle, sd_accuracy_shuffle, \
        lowPercentile_accuracy_shuffle, highPercentile_accuracy_shuffle, \
        p_accuracy, confusionMat = \
            fcn_repeated_stratified_kFold_crossVal(X, classLabels, classifier, \
                                                   nKFolds, nReps, \
                                                   compute_shuffleDist, nShuffles, \
                                                   shuffle_percentile, lda_solver)
                
        
    elif crossVal_type == 'fcn_LeaveOneOut_crossVal':
        
        accuracy, mean_accuracy_shuffle, sd_accuracy_shuffle, \
        lowPercentile_accuracy_shuffle, highPercentile_accuracy_shuffle, \
        p_accuracy, confusionMat = \
            fcn_LeaveOneOut_crossVal(X, classLabels, classifier, \
                                     compute_shuffleDist, nShuffles, shuffle_percentile, lda_solver)
                
        
    elif crossVal_type == 'fcn_stratified_kFold_crossVal':
        
        accuracy, mean_accuracy_shuffle, sd_accuracy_shuffle, \
        lowPercentile_accuracy_shuffle, highPercentile_accuracy_shuffle, \
        p_accuracy, confusionMat = \
            fcn_stratified_kFold_crossVal(X, classLabels, classifier, nKFolds, \
                                          compute_shuffleDist, nShuffles, shuffle_percentile, lda_solver)
                
    else:
        
        sys.exit('unsupported cross validation type entered')
    
    
    # return
    return accuracy, mean_accuracy_shuffle, \
           lowPercentile_accuracy_shuffle, highPercentile_accuracy_shuffle, \
           p_accuracy, confusionMat    
    
    
    
#%% COMPUTE DECODING SIGNIFICANCE

def fcn_decodeAccuracy_significance(accuracy_true, accuracy_shuffle, prctile):

    # number of shuffles
    nShuffles = np.size(accuracy_shuffle)    
    
    # average shuffle accuracy
    mean_accuracy_shuffle = np.mean(accuracy_shuffle)
    
    # sd shuffle accuracy
    sd_accuracy_shuffle = np.std(accuracy_shuffle)
    
    # lower and upper percentiles of shuffle accuracy
    lowPercentile_accuracy_shuffle = np.percentile(accuracy_shuffle,100-prctile)
    highPercentile_accuracy_shuffle = np.percentile(accuracy_shuffle,prctile)
    
    # p-value = fraction of shuffle accuracies >= true accuracy
    p_accuracy = (1 + np.size(np.where(accuracy_shuffle >= accuracy_true)[0]))/nShuffles
    
   
    # return
    return p_accuracy, mean_accuracy_shuffle, \
           lowPercentile_accuracy_shuffle, highPercentile_accuracy_shuffle, sd_accuracy_shuffle



#%% COMPUTE TIME AVERAGE OF DECODING ACCURACY DURING STIMULATION PERIOD

def fcn_avgAccuracy_duringStim(time, accuracy_vs_time, stimOn, stimOff):
    
    # find first index >= stim onset time
    ind_stimOnset = np.min(np.where( time >= stimOn )[0])
    
    # find first index >= stim offset time
    ind_stimOffset = np.min(np.where( time >= stimOff )[0])
    
    # get decoding accuracies during thos times
    accuracy_duringStim = accuracy_vs_time[ind_stimOnset:ind_stimOffset+1]
    
    # average accuracy across stimulation period
    avgAccuracy_duringStim = np.mean(accuracy_duringStim)
    
    return avgAccuracy_duringStim

    
#%% COMPUTE MAXIMUM DECODING ACCURACY DURING STIM

def fcn_maxAccuracy_duringStim(time, accuracy_vs_time, stimOn, stimOff):
    
    # find first index >= stim onset time
    ind_stimOnset = np.min(np.where( time >= stimOn )[0])
    
    # find first index >= stim offset time
    ind_stimOffset = np.min(np.where( time >= stimOff )[0])
    
    # get decoding accuracies during thos times
    accuracy_duringStim = accuracy_vs_time[ind_stimOnset:ind_stimOffset+1]
    
    # max accuracy across stimulation period
    maxAccuracy_duringStim = np.max(accuracy_duringStim)
    
    return maxAccuracy_duringStim




#%% DRAW NEURONS

def fcn_draw_neurons(cellIDs, clusterIDs, ensembleSize, seed, draw_equalPerCluster):

    # seed
    if seed == 'random':
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)
                
    # number of pops
    nPops = np.size(np.unique(clusterIDs))
    
    # cells per pop
    cells_per_pop = int(np.floor(ensembleSize/nPops))
    
    # leftovers
    n_leftover_cells = np.mod(ensembleSize, nPops)
    
    # if we want to draw neurons randomly, rather than equally per cluster
    if draw_equalPerCluster == False:
        ensembleInds = rng.choice(cellIDs, ensembleSize, replace=False)
        
    else:

        # ensemble inds
        ensembleInds = np.array([])
        
        for indPop in range(0,nPops):
            
            cells_inPop = np.nonzero(clusterIDs == indPop)[0]
            cells_inPop_andStim = np.intersect1d(cells_inPop, cellIDs)
            
            if np.size(cells_inPop_andStim) < cells_per_pop:
                drawnCell = rng.choice(cells_inPop, cells_per_pop, replace=False)
            else:
                drawnCell = rng.choice(cells_inPop_andStim, cells_per_pop, replace=False)
                
            ensembleInds = np.append(ensembleInds, drawnCell)
            
        # randomly select populations to draw more cells from
        pops_for_leftovers = np.random.choice(nPops, n_leftover_cells, replace=False)
        
        for indPop in pops_for_leftovers:
            
            cells_inPop = np.nonzero(clusterIDs == indPop)[0]
            cells_inPop_andStim = np.intersect1d(cells_inPop, cellIDs)
            
            cells_inPop = np.setdiff1d(cells_inPop, ensembleInds)
            cells_inPop_andStim = np.setdiff1d(cells_inPop_andStim, ensembleInds)
            
            if np.size(cells_inPop_andStim) == 0:
                drawnCell = rng.choice(cells_inPop, 1, replace=False)
            else:
                drawnCell = rng.choice(cells_inPop_andStim, 1, replace=False)
            
            ensembleInds = np.append(ensembleInds, drawnCell)
        
            
    ensembleInds = ensembleInds.astype(int)
    
    return ensembleInds

    
    
    