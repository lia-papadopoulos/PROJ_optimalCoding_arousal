
#%% BASIC IMPORTS

import numpy as np
import numpy.matlib
import sys
from nitime.utils import dpss_windows
import scipy.fft
import scipy

#%% Turns a dictionary my_dict into a class

class Dict2Class:
      
    def __init__(self, my_dict):
          
        for key in my_dict:
            setattr(self, key, my_dict[key])
            

#%%
# compute average firing rate of each neuron
def fcn_compute_firingRates(sim_params, spikes, burnTime):
    
    # number of E and I neurons
    N = sim_params.N
    Ne = sim_params.N_e
    
    # length of simulation
    T = sim_params.TF - sim_params.T0
    
    # amount of time over which to compute the firing rate
    Tavg = T - burnTime
    
    # time at which to start recording spikes
    Tb = sim_params.T0 + burnTime
    
    # neuron IDs
    neuron_IDs = spikes[1,:].astype(int)
    
    # spike times
    spike_times = spikes[0,:]
    
    # spike counts
    spike_cnts = np.zeros(N)
    
    # loop over spikes array and record spikes
    for i in range(0,len(spike_times),1):
        
        t = spike_times[i]
        
        # increment loop variable if current time is less than Tb
        if t < Tb:
            continue
        
        # neuron id of current index
        n = neuron_IDs[i]
        
        # update spike count of n
        spike_cnts[n] += 1
    
    # firing rates = # counts/window
    firing_rates = spike_cnts/Tavg
    
    # E and I rates
    E_rates = firing_rates[:Ne]
    I_rates = firing_rates[Ne:] 
    
    # return
    return E_rates, I_rates

#%% autocorrelation


def fcn_binned_spikeTrain(spike_train, minTime, maxTime, dt):
    
    bins = np.arange(minTime, maxTime + dt, dt)
    bin_times = bins[:-1] + dt/2
    binned_counts, _ = np.histogram(spike_train, bins)
    
    return binned_counts, bin_times

    

# binned_spikes:    (nBins,) array of binned spike counts (0s and 1s)
# max_lag:          maximum lag in seconds
# dt:               bin size in seconds
# norm_type:        type of normalization to use
#                   regular: divide by N
#                   corrected: divide by N-lag


def fcn_compute_autoCorr(binned_spikes, max_lag, dt, norm_type='regular'):
    
    # number of time samples
    nSamples = np.size(binned_spikes)
        
    # average # spikes/bin
    mean_spks_per_bin = np.mean(binned_spikes)
    
    # mean-subtracted spike count in each bin
    mean_subtracted = binned_spikes - mean_spks_per_bin
    
    # correlate mean-subtracted spike counts with itself
    # len(xcorr) = 2*nSamples - 1
    xcorr = np.correlate(mean_subtracted, mean_subtracted, 'full')
    
    # number of overlapping samples at each lag (including zero)
    # biggest lag has only 1 overlapping sample
    # [1, 2, ... , nSamples, nSamples-1, ..., 1]
    nSamples_eachLag = np.arange(1, nSamples + 1, 1)
    nSamples_eachLag = np.append(nSamples_eachLag, np.flip(nSamples_eachLag[:-1]))


    # divide by # samples used for each lag
    if norm_type == 'corrected':
        xcorr = xcorr/nSamples_eachLag
    # or just the number of samples
    else:
        xcorr = xcorr/nSamples
    
    # correlation at zero lag
    xcorr_0 = xcorr[nSamples-1]
    
    # divide by correlation at zero lag
    xcorr = xcorr/xcorr_0
    
    # positive lags (inluding zero)
    xcorr = xcorr[nSamples-1:]
    
    # only go up to max-lag
    max_lag_ind = round(max_lag/dt)
    xcorr = xcorr[:max_lag_ind+1]
        
    # lag time
    lag_times = np.arange(0, max_lag_ind+1) * dt
    
    # return
    return xcorr, lag_times, mean_spks_per_bin



def fcn_lagged_covariance(x, y, max_lag, dt):
    """
    Calculates the lagged covariance between two time series.

    Args:
        x (np.ndarray): The first time series.
        y (np.ndarray): The second time series.
        lag (int): The lag value. Positive lag means y is shifted forward, 
                   negative lag means y is shifted backward.

    Returns:
        float: The lagged covariance.
    """

    nLags = round(max_lag/dt)

    if len(x) != len(y):
        raise ValueError("Time series must have the same length")

    if nLags >= len(x):
         raise ValueError("Lag value is too large")

    
    cov_xy = np.ones(nLags + 1)*np.nan
    lag_times = np.ones(nLags + 1)*np.nan
    
    for lag in range(0, nLags + 1):
        
        if lag > 0:
            x_aligned = x[:-lag]
            y_aligned = y[lag:]
        else:
            x_aligned = x
            y_aligned = y
            
        cov_xy[lag] = np.cov(x_aligned, y_aligned)[0,1]
        
    corr_xy = cov_xy/cov_xy[0]
    lag_times = np.arange(0, nLags+1)*dt

    return corr_xy, lag_times
    

#%%

## apply gaussian kernal to spikes
def fcn_apply_gaussianKernel(t,ts,sigma):
    
    conv_spike = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-((t-ts)**2)/(2*sigma**2))
    
    return conv_spike


# compute average firing rate of each neuron using a gaussian window function
# r(t) = sum_i w(t-ti); 
# w=gaussian window function, ti=ith spike time of particular neuron
# window_std:  standard deviation of gaussian
# window_step: time between centers of adjacent windows
def fcn_compute_time_resolved_rate_gaussian(sim_params, spikes, To, Tf, \
                                            window_std, window_step):
    
    
    
    # number of E and I neurons
    N = sim_params.N
    Ne = sim_params.N_e

        
    # times at which time resolved firing rate will be computed
    t = np.arange(To,Tf+window_step,window_step)
        
    
    # spike times
    ts = spikes[0,:]  
    n_spks = len(ts)
    
    # neuron IDs
    neuron_IDs = spikes[1,:].astype(int)
    
    # intitialize firing rate array
    rate = np.zeros((N,len(t)))
    
    # loop over all spike times
    for ts_ind in range(0,n_spks,1):
        
        # neuron id of current index
        n = neuron_IDs[ts_ind]
        
        # convolve spike with gaussian at all times t
        conv_spike = fcn_apply_gaussianKernel(t,ts[ts_ind],window_std)
        
        # update estimate of rate of neuron n at all times t by adding 
        # contribution from the current spike
        rate[n,:] = rate[n,:] + conv_spike
        
    # separate E and I neurons
    rateE = rate[:Ne,:]
    rateI = rate[Ne:,:]
    
    # return
    return t, rateE, rateI


# compute firing rate of each neuron via binning spikes        
def fcn_compute_time_resolved_rate_bins(sim_params, spikes, bin_width):
    
    # number of E and I neurons
    N = sim_params.N
    Ne = sim_params.N_e

    # beginning and end times of simulation
    To = sim_params.T0
    Tf = sim_params.TF
        
    # spike times
    ts = spikes[0,:]  
    
    # neuron IDs
    neuron_IDs = spikes[1,:].astype(int)
        
    # spike count bins
    bins = np.arange(To, Tf+bin_width, bin_width)
    bin_times = bins[:-1] + bin_width/2
    n_bins = np.size(bin_times)
    
    # rate vs time for each cell
    rate = np.zeros((N, n_bins))
    
    # bin spike counts
    for cellInd in range(0, N):
        
        # spike time inds of this cell
        spike_inds_cell = np.nonzero(neuron_IDs == cellInd)[0]
        spike_times_cell = ts[spike_inds_cell]
        rate[cellInd, :], _ = np.histogram(spike_times_cell, bins)
        rate[cellInd, :] = rate[cellInd, :]/bin_width
        
        
    # separate E and I neurons
    rateE = rate[:Ne,:]
    rateI = rate[Ne:,:]
    
    # return
    return bin_times, rateE, rateI

        

# compute time-resolved firing rates of clusters (i.e. avg across all neurons in given cluster)
# includes background population if popSize includes background population
def fcn_compute_clusterRates_vs_time(popSize, neuronRates):
    
    # get cluster assignments
    cluLabels = 0
    cluLabels = np.append(cluLabels, np.cumsum(popSize))
    nClu = len(cluLabels)-1 
    
    # initialize cluster rates vs time
    rates_clu = np.zeros((nClu,np.size(neuronRates,1)))

    # average rates over neurons in the same cluster
    for cluInd in range(0,nClu,1):
    
        # cluster start and end
        a = cluLabels[cluInd]
        b = cluLabels[cluInd+1]

        # average rate
        rates_clu[cluInd,:] = np.mean(neuronRates[a:b,:],0)
        
    return rates_clu


#%%
# compute cluster activation times given cluster rates and threshold for activation
def fcn_compute_cluster_activationTimes(tRate, clu_rates, thresh):
    
    nClu = np.size(clu_rates,0)
    
    clu_on_off = [np.nan*np.zeros((2,1))]*nClu
    
    # append one time point
    dt = tRate[1]-tRate[0]
    tRate = np.append(tRate, tRate[-1] + dt)

    
    for cluInd in range(0,nClu,1):
        
        # binarize cluster rate vector into active, inactive periods
        clu_binary = (clu_rates[cluInd,:] > thresh).astype(int)
        
        # pad either end with zero so we always get a start time and end time
        clu_binary_padded = np.append(0, clu_binary)
        clu_binary_padded = np.append(clu_binary_padded, 0)
        
        # check if unactivated the whole time
        if np.all(clu_binary_padded == 0):
            tStart = np.array([])
            tEnd = np.array([])

        else:
    
            # take difference to find start/end times of cluster activation
            clu_binary_diff = np.diff(clu_binary_padded)
            
            # start and end times of cluster activation
            tStart_ind = np.nonzero(clu_binary_diff == 1)[0]
            tEnd_ind = np.nonzero(clu_binary_diff == -1)[0]
            
            # check times
            if np.size(tStart_ind) != np.size(tEnd_ind):
                sys.exit('start and end times not equal size')
            if np.any((tStart_ind - tEnd_ind) > 0):
                sys.exit('some start times > end times')

            
            # get corresponding times in seconds
            tStart = tRate[tStart_ind]
            tEnd = tRate[tEnd_ind]
        
        # store start/end times
        clu_on_off[cluInd] = np.vstack((tStart, tEnd))
    
    return clu_on_off




#%% compute length of cluster activation periods

def fcn_compute_cluster_activation_lifetimes(tRate, clu_rates, thresh):
    
    nClu = np.size(clu_rates,0)
    
    activeLifetimes = np.zeros(nClu, dtype='object')*np.nan
    
    clu_on_off = fcn_compute_cluster_activationTimes(tRate, clu_rates, thresh)
    
    for indClu in range(0, nClu):
        
        cluster_activation_times = clu_on_off[indClu]
        
        activeLifetimes[indClu] = cluster_activation_times[1,:]-cluster_activation_times[0,:]
        

    return activeLifetimes


#%%
# compute average cluster timescale from cluster activation times
def fcn_compute_avg_clusterTimescale(cluster_activation_times):
    
    clusterLifetimes = cluster_activation_times[1,:]-cluster_activation_times[0,:]
        
    avg_clusterLifetime = np.nanmean(clusterLifetimes) # will still be nan if cluster never became active
    
    return avg_clusterLifetime


#%%
# compute average interactivation timescale from cluster activation times
def fcn_compute_avg_interactivationTimescale(cluster_activation_times):
    
    # intervval between cluster activation onset
    cluster_IAI = np.diff(cluster_activation_times[0,:])
        
    avg_cluster_IAI = np.nanmean(cluster_IAI) # will still be nan if cluster never became active
    
    return avg_cluster_IAI

#%%
# compute number of of active clusters per time
def fcn_compute_num_activeClusters(clu_rates, thresh):
    
    # binarize cluster rates to determine whether active/inactive at a given time
    clu_activity = (clu_rates > thresh)*1
    
    # sum across clusters to determine number of active clusters at each point in time
    num_active_clusters = (np.sum(clu_activity,0)).astype(int)
    
    return num_active_clusters



# compute fraction of time across simulation that m clusters are active
# INPUT: num_active_clusters = 1xT array of number of active clusters at each time
#        nMax                = largest possible number of active clusters 
def fcn_compute_freq_XActiveClusters(num_active_clusters, nMax):
    
    count_XActiveClusters = np.zeros(nMax+1)
    
    T = len(num_active_clusters)
    
    for tInd in range(0,T,1):
        
        nActive = num_active_clusters[tInd]
        
        count_XActiveClusters[nActive] += 1
        
    freq_XActiveClusters = count_XActiveClusters/T
    
    return freq_XActiveClusters


# compute which clusters are active at each time point
# compute average rate of active clusters at each time point
def fcn_compute_activeCluster_IDs_rates(tRate, clu_rates, thresh):
    
    # binarize cluster rates to determine whether active/inactive at a given time
    clu_activity = (clu_rates > thresh)*1    
    
    # list of active clusters
    ids_activeClus = []
    
    # average rate of active clusters
    avgRate_activeClus = np.zeros(len(tRate))
    
    # loop across time
    for tInd in range(0,len(tRate),1):
        
        ids = np.where(clu_activity[:,tInd]==1)[0]
        
        ids_activeClus.append(ids)
        
        # if active clusters
        if (np.size(ids) != 0):
            avgRate_activeClus[tInd] = np.mean(clu_rates[ids,tInd])
        # otherwise
        else:
            avgRate_activeClus[tInd] = np.nan
        
    return ids_activeClus, avgRate_activeClus


# compute which clusters are inactive at each time point
# compute average rate of inactive clusters at each time point
def fcn_compute_inactiveCluster_IDs_rates(tRate, clu_rates, thresh):
    
    # binarize cluster rates to determine whether active/inactive at a given time
    clu_activity = (clu_rates > thresh)*1    
    
    # list of inactive clusters
    ids_inactiveClus = []
    
    # average rate of active clusters
    avgRate_inactiveClus = np.zeros(len(tRate))
    
    # loop across time
    for tInd in range(0,len(tRate),1):
        
        ids = np.nonzero(clu_activity[:,tInd]==0)[0]
        
        ids_inactiveClus.append(ids)
        
        # if inactive clusters
        if (np.size(ids) != 0):
            avgRate_inactiveClus[tInd] = np.mean(clu_rates[ids,tInd])
        # otherwise
        else:
            avgRate_inactiveClus[tInd] = np.nan
        
    return ids_inactiveClus, avgRate_inactiveClus
 
    
#%%

def fcn_compute_clusterActivation(clu_rates, thresh):
    
    # binarize cluster rates to determine whether active/inactive at a given time
    clu_binarized = (clu_rates > thresh)*1  
    
    return clu_binarized


# compute average rate of active clusters at each time point
# given cluster activation array
def fcn_compute_activeCluster_rates_givenBinarized(tRate, clu_rates, clu_binarized):
      
    
    # list of active clusters
    ids_activeClus = []
    
    # average rate of active clusters
    avgRate_activeClus = np.zeros(len(tRate))
    
    # loop across time
    for tInd in range(0,len(tRate),1):
        
        ids = np.nonzero(clu_binarized[:,tInd]==1)[0]
        
        ids_activeClus.append(ids)
        
        # if active clusters
        if (np.size(ids) != 0):
            avgRate_activeClus[tInd] = np.mean(clu_rates[ids,tInd])
        # otherwise
        else:
            avgRate_activeClus[tInd] = np.nan
        
    return avgRate_activeClus

# compute average rate of inactive clusters at each time point
# given cluster activation array
def fcn_compute_inactiveCluster_rates_givenBinarized(tRate, clu_rates, clu_binarized):
    
    
    # list of inactive clusters
    ids_inactiveClus = []
    
    # average rate of active clusters
    avgRate_inactiveClus = np.zeros(len(tRate))
    
    # loop across time
    for tInd in range(0,len(tRate),1):
        
        ids = np.nonzero(clu_binarized[:,tInd]==0)[0]
        
        ids_inactiveClus.append(ids)
        
        # if inactive clusters
        if (np.size(ids) != 0):
            avgRate_inactiveClus[tInd] = np.mean(clu_rates[ids,tInd])
        # otherwise
        else:
            avgRate_inactiveClus[tInd] = np.nan
        
    return avgRate_inactiveClus



#%%

# compute the average rate of active clusters when X clusters are active
# INPUTS
#       avgRate_activeClus = vector of average rate of active clusters at each time point
#       num_active_clusters = vector of number of active clusters at each time point
#       maxActive = maximum # active clusters to consider
def fcn_compute_clusterRate_XClustersActive(avgRate_activeClus, num_active_clusters, maxActive):
    
    # initialize
    clusterRate_XActive = np.zeros(maxActive+1)
    
    # loop over each choice of # simultaneously active clusters
    for nActive in range(0,maxActive+1,1):
        
        # time points where nActive clusters were active
        tInd = np.where(num_active_clusters == nActive)[0]
        
        # average rates of active clusters across those time points
        clusterRate_XActive[nActive] = np.mean(avgRate_activeClus[tInd])
        
        
    return clusterRate_XActive


# compute the average rate of inactive clusters when X clusters are active
# INPUTS
#       avgRate_inactiveClus = vector of average rate of inactive clusters at each time point
#       num_active_clusters = vector of number of active clusters at each time point
#       maxActive = maximum # active clusters to consider
def fcn_compute_inactive_clusterRate_XClustersActive(avgRate_inactiveClus, num_active_clusters, maxActive):
    
    # initialize
    inactive_clusterRate_XActive = np.zeros(maxActive+1)
    
    # loop over each choice of # simultaneously active clusters
    for nActive in range(0,maxActive+1,1):
        
        # time points where nActive clusters were active
        tInd = np.nonzero(num_active_clusters == nActive)[0]
        
        # average rates of active clusters across those time points
        inactive_clusterRate_XActive[nActive] = np.mean(avgRate_inactiveClus[tInd])
        
        
    return inactive_clusterRate_XActive


# compute the average rate of background population when X clusters are active
# INPUTS
#       avgRate_background = vector of average rate of background neurons at each time point
#       num_active_clusters = vector of number of active clusters at each time point
#       maxActive = maximum # active clusters to consider
def fcn_compute_backgroundRate_XClustersActive(avgRate_background, num_active_clusters, maxActive):
    
    # initialize
    backgroundRate_XActive = np.zeros(maxActive+1)
    
    # loop over each choice of # simultaneously active clusters
    for nActive in range(0,maxActive+1,1):
        
        # time points where nActive clusters were active
        tInd = np.nonzero(num_active_clusters == nActive)[0]
        
        # average rates of active clusters across those time points
        backgroundRate_XActive[nActive] = np.mean(avgRate_background[tInd])
        
        
    return backgroundRate_XActive
        


#%%

# compute average rate of each cluster when its active vs. inactive
def fcn_compute_clusterRate_active_inactive(clu_rates, thresh):
    
    # number of clusters
    nClu = np.size(clu_rates,0)
    
    # initialize array
    avgRate_active = np.zeros(nClu)
    avgRate_inactive = np.zeros(nClu) 
    
    # binarize cluster rates to determine whether active/inactive at a given time
    clu_activity = (clu_rates > thresh)*1   
    
    # loop over clusters and compute average rate across active/inactive times
    for cluInd in range(0,nClu,1):
        
        activeInds = np.where(clu_activity[cluInd,:]==1)[0]
        inactiveInds = np.where(clu_activity[cluInd,:]==0)[0]
        
        avgRate_active[cluInd] = np.mean(clu_rates[cluInd,activeInds])
        avgRate_inactive[cluInd] = np.mean(clu_rates[cluInd,inactiveInds])
    
    
    return avgRate_active, avgRate_inactive
    
    
#%% compute cells in targeted and nontargeted clusters

def fcn_compute_Ecells_in_targeted_nontargeted_Eclusters(sim_params, clust_sizeE):

        clusters_start_end = np.cumsum(np.concatenate(([0],clust_sizeE)))
        targeted_clusters = sim_params.selectiveClusters
        neurons_in_targetedClus = []
        for count, indClu in enumerate(targeted_clusters):
    
            neurons_in_targetedClus = np.concatenate((neurons_in_targetedClus, \
                                                      np.arange(clusters_start_end[indClu],clusters_start_end[indClu+1])))
                                                                                                                            
        neurons_in_targetedClus = neurons_in_targetedClus.astype(int)
        
        
        # non targeted clusters
        nClu = sim_params.p
        nonTargeted_clusters = np.setdiff1d(np.arange(0,nClu),targeted_clusters)
        
        # neurons in non-targeted clusters
        neurons_in_nontargetedClus = []
        for count,indClu in enumerate(nonTargeted_clusters):
    
            neurons_in_nontargetedClus = np.concatenate((neurons_in_nontargetedClus, \
                                                      np.arange(clusters_start_end[indClu],clusters_start_end[indClu+1])))                                                                  
                
        neurons_in_nontargetedClus = neurons_in_nontargetedClus.astype(int)
        
        # return
        return neurons_in_targetedClus, neurons_in_nontargetedClus


    
#%%
# compute avg firing rates across population of E and I neurons
# inputs
#   sim_params  class that contains all model parameters
#   spikes      numpy array (first row = spike times, second row = neuron ID)
#   burnTime    amount of time to discard before firing rate calculation
# outputs
#   avg_Erate   population and time averaged firing rate of E neurons
#   avg_Irate   population and time averaged firing rate of I neurons
def fcn_compute_popAvg_firingRate(sim_params, spikes, burnTime):
    
    # number of E and I neurons
    Ne = sim_params.N_e
    Ni = sim_params.N_i
    
    # length of simulation
    T = sim_params.TF - sim_params.T0
    
    # amount of time over which to compute the firing rate
    Tavg = T - burnTime
    
    # neuron IDs
    neuron_IDs = spikes[1,:]
    
    # spike times
    spike_times = spikes[0,:]
    
    # find number of E and I spikes
    Espikes = np.where((neuron_IDs < Ne) & (spike_times >= burnTime))
    Ispikes = np.where((neuron_IDs >= Ne) & (spike_times >= burnTime))  
    num_Espikes = np.size(Espikes)
    num_Ispikes = np.size(Ispikes)
    
    # average rates based on length of simulation and number of neurons
    avg_Erate = num_Espikes/(Tavg * Ne)
    avg_Irate = num_Ispikes/(Tavg * Ni)
    
    # return
    return avg_Erate, avg_Irate
    

def fcn_compute_total_spkCount(sim_params, spikes):
    
    # number of E and I neurons
    N = sim_params.N
    
    # neuron IDs
    neuron_IDs = spikes[1,:].astype(int)
    
    # spike times
    spike_times = spikes[0,:]
    
    # spike counts
    spike_cnts = np.zeros(N)
    
    # loop over spikes array and record spikes
    for i in range(0,len(spike_times),1):
                
        # neuron id of current index
        n = neuron_IDs[i]
        
        # update spike count of n
        spike_cnts[n] += 1 

    return spike_cnts


#%% 
def fcn_compute_spkCounts(sim_params, spikes, burnTime, windLength, windStep):
    
    # number of neurons
    N = sim_params.N
    Ne = sim_params.N_e
    
    # time at end of simulation
    Tf = sim_params.TF
    
    # time at which to begin computing FF
    Tb = sim_params.T0 + burnTime
    
    # number of windows given simulation length, window length, window increment
    Nw = int(np.floor((Tf - Tb - windLength)/windStep)) + 1
    
    # initialize spkCounts array (size = # windows x number of neurons)
    spkCounts = np.zeros((Nw, N)) 
    
    # initialize array to hold window times
    t_window = np.zeros(Nw)
          
    # times and neuron ids
    times = spikes[0,:].copy()
    ids = spikes[1,:].copy().astype(int)
           
    # loop over windows
    for windInd in range(0,Nw,1):
        
        # beginning of window
        Tbegin = Tb + windInd*windStep 
        
        # end time of current window
        Tend = Tb + windInd*windStep + windLength
      
        # spikes in this time window
        indSpks_window = np.nonzero(((times >= Tbegin) & (times < Tend)))[0]
        spkID_window = ids[indSpks_window].copy()
        
        # loop over spk ids and add to count
        for i in range(0, len(spkID_window)):
        
            # spike id
            n = spkID_window[i]
        
            # update spike count array
            spkCounts[windInd, n] += 1
          
        
        # time of window end
        t_window[windInd] = Tend
                
    
    # output spike counts for E and I neurons
    spkCounts_E = spkCounts[:,:Ne].copy()
    spkCounts_I = spkCounts[:,Ne:].copy()
    
    return spkCounts_E, spkCounts_I, t_window

#%%
# compute fano factor of each neuron in a network

# INPUTS
#   spikeTimes_trialList --  list of "spikes" array
#                        len(spikeTimes_array) = # trials
#                        "spikes": np array with row1 = spike times, row2 = neuron ID)

def fcn_compute_raw_fanoFactor(sim_params, spikeTimes_trialList, burnTime, \
                               windLength, windStep):
        
    
    # number of neurons
    Ne = sim_params.N_e
    Ni = sim_params.N_i
    
    # time at end of simulation
    Tf = sim_params.TF
    
    # time at which to begin computing FF
    Tb = sim_params.T0 + burnTime
    
    # number of windows given simulation length, window length, window increment
    Nw = int(np.floor((Tf - Tb - windLength)/windStep)) + 1
    
    # number of trials
    Nt = len(spikeTimes_trialList)
    
    # intialize
    spkCnts_e = np.zeros((Nt, Nw, Ne))
    spkCnts_i = np.zeros((Nt, Nw, Ni))
    t_FF = np.zeros(Nw)
    
    # compute spike counts of each neuron for each trial, time window
    
    # loop over trials
    for trialInd in range(0,Nt,1):
        
        # spike time array for given trial
        spikes = spikeTimes_trialList[trialInd] 
        
        spkCnts_e[trialInd,:,:], spkCnts_i[trialInd,:,:], t_FF = \
            fcn_compute_spkCounts(sim_params, spikes, burnTime, windLength, windStep)
          
                  
    # compute mean and variance of spike count array across trials
    mean_spike_cnts_e = np.mean(spkCnts_e,axis=0)
    var_spike_cnts_e = np.var(spkCnts_e,axis=0)
    
    mean_spike_cnts_i = np.mean(spkCnts_i, axis=0)
    var_spike_cnts_i = np.var(spkCnts_i, axis=0)    
        
    # compute fano factor as ratio of variance to mean of spike counts
    FF_E = var_spike_cnts_e/mean_spike_cnts_e
    FF_I = var_spike_cnts_i/mean_spike_cnts_i
    
    # return fano factors for E and I neurons
    return FF_E, FF_I, t_FF
                  
    
#%% power spectrum for spike train data

# data = nTrials x nTpts (binned spike counts)

def mt_specpb_lp(data, Fs=1000, NW=4, trial_ave=True, type = 0):
    
    nTrials = np.size(data, 0)
    nTpts = np.size(data,1)
    dt = 1/Fs
    
    tapers, _ = dpss_windows(nTpts, NW, 2*NW-1)             # Compute the tapers,
    tapers *= np.sqrt(Fs)                                   # ... and scale them.
    
    nTapers = np.size(tapers, 0)

    dataT = np.ones((nTrials, nTapers, nTpts ))*np.nan
    
    for indTrial in range(0, nTrials):
        
        trial_data = data[indTrial, :].copy()
        
        for indTaper in range(0, nTapers):
            
            dataT[indTrial, indTaper, :] = trial_data*tapers[indTaper,:]
            
            
    T = scipy.fft.rfft(tapers)                                 # Compute the fft of the tapers.
    J = scipy.fft.rfft(dataT)                                  # Compute the fft of the tapered data.

    for indTrial in range(0, nTrials):
        
        if type == 0:
            J[indTrial, :, :] = J[indTrial, :, :] - T*np.mean(data[indTrial, :])
        else:
            J[indTrial, :, :] = J[indTrial, :, :] - T*np.mean(data)
    
    
    J *= np.conj(J)                                        # Compute the spectrum
    S = np.real(np.mean(J,1))

    # normalize by rate in this trial
    Snorm = np.ones(np.shape(S))*np.nan
    
    for indTrial in range(0, nTrials):
        if type == 0:
            Snorm[indTrial, :] = S[indTrial, :]/(np.sum(data[indTrial,:])/(dt*nTpts))
        else:
            Snorm[indTrial, :] = S[indTrial, :]/(np.sum(data)/(dt*nTpts*nTrials))

    f = scipy.fft.rfftfreq(nTpts, 1 / Fs)
    
    if trial_ave: 
        S = np.mean(S,0)                        # Average across trials.
        Snorm = np.mean(Snorm, 0)
        
    return f, S, Snorm



# raw spectrum
def raw_specpb_lp(data, Fs=1000, trial_ave=True, type = 0):
    
    nTrials = np.size(data, 0)
    nTpts = np.size(data,1)
    dt = 1/Fs
    
    window_func = np.ones(nTpts)
    window_func = window_func*1/np.sqrt(dt*nTpts)
    
    dataT = np.ones((nTrials, nTpts ))*np.nan
    
    for indTrial in range(0, nTrials):
        
        trial_data = data[indTrial, :].copy()
        
        dataT[indTrial, :] = trial_data*window_func
    
    J = scipy.fft.rfft(dataT)
    R = scipy.fft.rfft(window_func)

    for indTrial in range(0, nTrials):
        
        if type == 0:
            J[indTrial, :] = J[indTrial, :] - R*np.mean(data[indTrial, :])
        else:
            J[indTrial, :] = J[indTrial, :] - R*np.mean(data)
    
    J *= np.conj(J)                                        # Compute the spectrum
    S = np.real(J)
    
    # normalize by rate in this trial
    Snorm = np.ones(np.shape(S))*np.nan
    
    for indTrial in range(0, nTrials):
        if type == 0:
            Snorm[indTrial, :] = S[indTrial, :]/(np.sum(data[indTrial,:])/(dt*nTpts))
        else:
            Snorm[indTrial, :] = S[indTrial, :]/(np.sum(data)/(dt*nTpts*nTrials))

    f = scipy.fft.rfftfreq(nTpts, 1 / Fs)
    
    if trial_ave: 
        S = np.mean(S,0)                        # Average across trials.
        Snorm = np.mean(Snorm, 0)
        
    return f, S, Snorm


#%% 
# compute pairwise spike correlations for spontaneous runs of the simulation

# INPUTS
#   spikeTimes_trialList --  list of "spikes" array
#                            len(spikeTimes_array) = # trials
#                            "spikes": np array with shape (2,ns)
#                                      row1 = spike times
#                                      row2 = neuron ID

def fcn_compute_spike_correlations(sim_params, spikeTimes_trialList, burnTime, \
                                   windLength, windStep, nSamples = 100):    
    
    
    # checks
    # make sure spikeTimes_trialList is a list
    if isinstance(spikeTimes_trialList, list) == False:
       
        sys.exit('error: spikeTimes_trialList must be a list')
    
    
    # Number of E and I cells
    Ne = sim_params.N_e
    Ni = sim_params.N_i
    
    
    # number of trials
    Nt = len(spikeTimes_trialList)
    
    # intialize
    spike_cnts_e = np.transpose([[] for i in range(Ne)])
    spike_cnts_i = np.transpose([[] for i in range(Ni)])
    
    # compute spike counts of each neuron for each trial, time window
    
    # loop over trials
    for trialInd in range(0,Nt,1):
        
        # spike time array for given trial
        spikes = spikeTimes_trialList[trialInd] 
        
        # shape = (Nw, N)
        sc_e, sc_i, _ = \
            fcn_compute_spkCounts(sim_params, spikes, burnTime, windLength, windStep)
            
   
        # append
        spike_cnts_e = np.vstack((spike_cnts_e, sc_e))
        spike_cnts_i = np.vstack((spike_cnts_i, sc_i))
    
    
    # subsample
    inds_sample = np.random.choice(np.shape(spike_cnts_e)[0], nSamples, replace=False)
    
    spike_cnts_e = spike_cnts_e[inds_sample,:]
    spike_cnts_i = spike_cnts_i[inds_sample,:]
    
    # compute sample variance of spike counts
    var_spk_cnts_e = np.var(spike_cnts_e, axis=0, ddof=1)
    var_spk_cnts_i = np.var(spike_cnts_i, axis=0, ddof=1)
        
    # compute covariance of spike counts
    cov_spk_cnts_e = np.cov(spike_cnts_e, rowvar=False)
    cov_spk_cnts_i = np.cov(spike_cnts_i, rowvar=False)
                
    # compute pearson correlation
    corr_spk_cnts_e = np.corrcoef(spike_cnts_e, rowvar=False)
    corr_spk_cnts_i = np.corrcoef(spike_cnts_i, rowvar=False)
    
    
    
    return var_spk_cnts_e, cov_spk_cnts_e, corr_spk_cnts_e, \
           var_spk_cnts_i, cov_spk_cnts_i, corr_spk_cnts_i, \
           spike_cnts_e, spike_cnts_i


#%% dimensionality
#   compute dimensionality measure from Mazzucato 2016

def fcn_dimensionality(cov_matrix):
       
    tr_cov = np.trace(cov_matrix)
    tr_cov_sq = np.trace(np.matmul(cov_matrix,cov_matrix))
    dimensionality = (tr_cov**2)/tr_cov_sq 
    
    
    return dimensionality


#%% response efficacy for a single stimulus

'''
trialAvg_rate_resp                  (n_respCells, n_tPts)
trialAvg_rate_nonresp               (n_nonrespCells, n_tPts)
trialVar_rate_resp                  (n_respCells, n_tPts)
trialVar_rate_nonresp               (n_nonrespCells, n_tPts)
'''

def fcn_compute_response_efficacy(trialAvg_rate_resp, trialAvg_rate_nonresp, trialVar_rate_resp, trialVar_rate_nonresp):
    
    
    # cell avg rate of response cells
    cellAvg_trialAvg_rate_respCells = np.mean(trialAvg_rate_resp,0)    

    # cell avg rate of nonresponse cells
    cellAvg_trialAvg_rate_nonrespCells = np.mean(trialAvg_rate_nonresp,0)   
    
    # response signal
    resp_signal_vs_time = cellAvg_trialAvg_rate_respCells - cellAvg_trialAvg_rate_nonrespCells   
    
    # cell avg var rate of response cells
    cellAvg_trialVar_rate_respCells = np.nanmean(trialVar_rate_resp,0)        
     
    # cell avg var rate of nonresponse cells
    cellAvg_trialVar_rate_nonrespCells = np.nanmean(trialVar_rate_nonresp,0)  
     
    # total trial sd
    resp_var_vs_time = np.sqrt( (1/2)*(cellAvg_trialVar_rate_respCells + cellAvg_trialVar_rate_nonrespCells))
                          
    # response efficacy
    resp_efficacy_vs_time = resp_signal_vs_time/resp_var_vs_time 
    
    return resp_signal_vs_time, resp_var_vs_time, resp_efficacy_vs_time





#%% significantly responding cells


### compare pre vs post stim responses for a single stimulus
'''
pval:           (nCells,)
sigLevel:       threshold for significant response

sigCells:       array of cell ids that responded significantly 
'''
def fcn_sigCells_pre_vs_post_stim(pval, sigLevel):
    
    sigCells = np.nonzero(pval < sigLevel)[0]
    
    return sigCells
    

### set of cells that respond significantly to at least one tone
'''
sigCells_eachStim:       (nStim,) object array
                         each stimulus index holds list of cells that responded to that input
                

allSig_cells:            array of cell ids that respond to at least one tone
'''
def fcn_sigCells_anyStimulus(sigCells_eachStim):
    
    nStim = np.size(sigCells_eachStim)
    allSig_cells = np.array([])
    
    for indStim in range(0, nStim):
        allSig_cells = np.append(allSig_cells, sigCells_eachStim[indStim])
        
    
    allSig_cells = np.unique(allSig_cells)
    
    return allSig_cells
    

### function to determine response amplitude of each cell for a given stimulus
'''
trialAvg_gain:          (nCells, nTpts)

respAmp:                (nCells,)
'''
def fcn_responseAmp_eachStim(trialAvg_gain, tInd_response):
        
    respAmp = trialAvg_gain[:, tInd_response].copy()
    
    return respAmp

### function to convert object array of response amplitudes for each stimulus to (nCells, nStim) array
'''
respAmp_eachStim:       (nStim,) object array of response amplitudes
                        each stimulus index holds (nCells,) response amplitude
'''
def fcn_responseAmp_allStim(respAmp_eachStim):
    
    nCells = np.size(respAmp_eachStim[0])
    nStim = np.size(respAmp_eachStim)
    
    respAmp_allStim = np.zeros((nCells, nStim))
    
    for indStim in range(0, nStim):
        
        respAmp_allStim[:, indStim] = respAmp_eachStim[indStim].copy()
        
    return respAmp_allStim
    

    
    
### wrapper function for computing significant cells and response amplitude from psth data
def fcn_compute_sigCells_respAmp(psth_data, sigLevel):
    
    # unpack data
    pval = psth_data['pval_preStim_vs_postStim_allBaseMod'].copy()
    trialAvg_gain = psth_data['trialAvg_gain_allBaseMod'].copy()
    tInd_postStim = psth_data['postStim_wind']
    
    # significant cells for each stimulus
    sigCells_eachStim = fcn_sigCells_pre_vs_post_stim(pval, sigLevel)
    
    # response to each stimulus
    resp_eachStim =  fcn_responseAmp_eachStim(trialAvg_gain, tInd_postStim)
    
    
    return sigCells_eachStim, resp_eachStim
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    