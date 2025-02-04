import numpy as np


#---------- FUNCTION FOR SETTING STIMULATED CLUSTERS + NEURONS -----------#
# INPUTS
#    s_params
#    random_seed: determines which clusters are stimulated for the different stimuli
# OUTPUTS
#    selectiveClusters: list of length nStim, where each element gives the set
#                       clusters that will be stimulated for the given 
#                       stimulation number


class Dict2Class:
      
    def __init__(self, my_dict):
          
        for key in my_dict:
            setattr(self, key, my_dict[key])
            

def get_stimulated_clusters(params, random_seed):
    
    if isinstance(params, dict):
        sim_params = Dict2Class(params)
    else:
        sim_params = params
    
    # unpack sim_params
    p = sim_params.p
    f_selectiveClus = sim_params.f_selectiveClus
    mixed_selectivity = sim_params.mixed_selectivity
    nStim = sim_params.nStim
    
    # list to hold selective clusters for each stim condition
    selectiveClusters = ['NONE']*nStim
     
    # set random number generator using the specified seed
    if random_seed == 'random':
        random_seed = np.random.choice(10000,1)
        rng = np.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng(random_seed)

    # get number of selective clusters
    n_selectiveClus = np.round(f_selectiveClus*p, 0).astype(int)

    # get selective cluster ids
    
    # if mixed_selectivity allowed
    if mixed_selectivity:
        # loop over stimuli
        for stim_ind in range(0,nStim,1):
            selectiveClusters[stim_ind] = rng.choice(p, size=n_selectiveClus, replace=False)
            
    # else if non-overlapping
    else:
        # all selective clusters (#conditions x # clusters per condition)
        all_stim_clus = rng.choice(p, size=n_selectiveClus*nStim, replace=False)
        # loop over stimuli
        for stim_ind in range(0,nStim,1):
            indBegin = stim_ind*n_selectiveClus
            indEnd = indBegin + n_selectiveClus
            selectiveClusters[stim_ind] = all_stim_clus[indBegin:indEnd]
     
    # return    
    return selectiveClusters
   

#%%

'''
#---------- FUNCTION FOR GENERATING STIMULUS AT EACH POINT OF SIMULATION -----------#
#
# NOTE: ONLY DOES BOX STIMULUS FOR NOW
# NOTE: SETUP TO WORK WITH THE FUNCTION 'fcn_simulation.py' in PROJ_VariabilityGainMod

# inputs
#    stim_type
#    onset_time
#    duration
#    ampltidue:           either a scalar or an Nx1 array indicating stim amplitude for each cell
#    stimulated_cells:    Nx1 array indicating whether each neuron receives stimulus
#    current_time:        current simulation time
#    
#    potentially other parameters involving stimulus

def fcn_generate_stimulus(stim_type, onset_time, duration, amplitude, stimulated_cells, current_time):
    
    # number of cells
    N = np.size(stimulated_cells)
    
    # compute offset time
    offset_time = onset_time + duration
    
    # check if current time is >= stim onset time and <= stim offset time
    if ( (current_time >= onset_time) and (current_time <= offset_time) ):
        
        if stim_type == 'box':
            
            stim_current = amplitude*stimulated_cells
            
        elif stim_type == 'linear':
            
            # equation for a straight line that satisfies:
            # y(offset_time) = amplitude
            # y(onset_time) = 0
            
            # slope
            m = amplitude/duration
            # intercept
            b = -m*onset_time
            
            stim_current = (m*current_time + b)*stimulated_cells
            

        else:
            
            sys.exit('error: this function only supports box stimulus right now.')
            
            
    else:
            
        stim_current = np.zeros(N)
            
        


    # return
    return stim_current

'''

#%% DIFFERENT TYPES OF STIMULI

def fcn_box_stimulus(onset_time, duration, amplitude, current_time):
    
    # compute offset time
    offset_time = onset_time + duration
    
    # determine if stim should be on
    if ((current_time >= onset_time) and (current_time <= offset_time)):
        
        stim_current = amplitude
        
    else:
        
        stim_current = 0.*amplitude
        
    return stim_current


def fcn_linear_stimulus(onset_time, duration, amplitude, current_time):
    
    # compute offset time
    offset_time = onset_time + duration
    
    # determine if stim should be on
    if ((current_time >= onset_time) and (current_time <= offset_time)):
        
        # equation for a straight line that satisfies:
        # y(offset_time) = amplitude
        # y(onset_time) = 0
        
        # slope
        m = amplitude/duration
        # intercept
        b = -m*onset_time
        # current
        stim_current = (m*current_time + b)
    
    else:
        
        stim_current = 0.*amplitude
        
    return stim_current


def fcn_diff2exp_stimulus(onset_time, peak_amplitude, taur, taud, current_time):
    
    # normalization
    alpha = 1/((taur/taud)**(taur/(taud-taur)) - (taur/taud)**(taud/(taud-taur)))
    
    # should stimulus be on?
    if current_time < onset_time:
        
        stim_current = 0.*peak_amplitude
    
    else:
        
        stim_current = alpha*peak_amplitude*( np.exp(-(current_time - onset_time)/taud) - \
                                              np.exp(-(current_time - onset_time)/taur) )
        
    return stim_current


