
import sys
import numpy as np

class Dict2Class:
      
    def __init__(self, my_dict):
          
        for key in my_dict:
            setattr(self, key, my_dict[key])


def fcn_basic_setup(params_dict, args=None):

    if args is not None:
        # argparse vals
        params_dict = get_argparse_vals(params_dict, args)

    # set dependent variables
    params_dict = set_dependent_vars(params_dict)
    
    params_dict = update_JplusAB(params_dict)

    # update cluster size std
    params_dict = fcn_update_clusterSize_std(params_dict)
    
    # update stim params
    params_dict = update_stim_params(params_dict)
    
    # baseline external inputs
    params_dict = set_baseline_external_inputs_ei(params_dict)
    
    # stimulus rate
    params_dict = set_max_stim_rate(params_dict)

    # return
    return params_dict


#---------------------------------------------------------------------------------
# GET ANY VALUES THAT COULD HAVE BEEN PASSED THROUGH ARGPARSE
#---------------------------------------------------------------------------------
def get_argparse_vals(params_dict, args):
    
    # get any values from argparser

    # network type
    params_dict['net_type'] = args.net_type

    # perturbations
    params_dict['pert_mean_nu_ext_ee'] = args.pert_mean_nu_ext_ee
    params_dict['pert_mean_nu_ext_ie'] = args.pert_mean_nu_ext_ie
    params_dict['pert_mean_nu_ext_ii'] = args.pert_mean_nu_ext_ii
    params_dict['pert_mean_nu_ext_ei'] = args.pert_mean_nu_ext_ei

    # beta inputs
    params_dict['nu_ext_ee_beta_spread'] = args.nu_ext_ee_beta_spread
    params_dict['nu_ext_ie_beta_spread'] = args.nu_ext_ie_beta_spread
    params_dict['nu_ext_ei_beta_spread'] = args.nu_ext_ei_beta_spread
    params_dict['nu_ext_ii_beta_spread'] = args.nu_ext_ii_beta_spread
    
    # uniform inputs
    params_dict['nu_ext_ee_uniform_spread'] = args.nu_ext_ee_uniform_spread
    params_dict['nu_ext_ie_uniform_spread'] = args.nu_ext_ie_uniform_spread
    params_dict['nu_ext_ei_uniform_spread'] = args.nu_ext_ei_uniform_spread
    params_dict['nu_ext_ii_uniform_spread'] = args.nu_ext_ii_uniform_spread
    
    # synaptic reduction
    if 'Jee_reduction' in params_dict:
        params_dict['Jee_reduction'] = args.Jee_reduction
    
    if 'Jie_reduction' in params_dict:
        params_dict['Jie_reduction'] = args.Jie_reduction
    
    # JeePlus variation
    if 'JplusEE_sweep' in params_dict: 
        params_dict['JplusEE_sweep'] = args.JplusEE_sweep
    
    # zero mean sd_nu_ext
    if 'zeroMean_sd_nu_ext_ee' in params_dict:
        params_dict['zeroMean_sd_nu_ext_ee'] = args.zeroMean_sd_nu_ext_ee

    
    return params_dict


def set_dependent_vars(params_dict):
                
    # variables that depend on main inputs

    params_dict['N_e'] = int(params_dict['N']*params_dict['ne'])                   
    params_dict['N_i'] = int(params_dict['N'] - params_dict['N_e'])       

    params_dict['Cee'] = params_dict['pee']*params_dict['N_e']
    params_dict['Cei'] = params_dict['pei']*params_dict['N_i']
    params_dict['Cii'] = params_dict['pii']*params_dict['N_i']
    params_dict['Cie'] = params_dict['pie']*params_dict['N_e']
    
    params_dict['Cext_ee'] = params_dict['pext_ee']*params_dict['N_e']
    params_dict['Cext_ie'] = params_dict['pext_ie']*params_dict['N_e']
    params_dict['Cext_ei'] = params_dict['pext_ei']*params_dict['N_i']
    params_dict['Cext_ii'] = params_dict['pext_ii']*params_dict['N_i']    
    
    # baseline Js
    params_dict['Jee_base'] = params_dict['jee']/np.sqrt(params_dict['N'])
    params_dict['Jie_base'] = params_dict['jie']/np.sqrt(params_dict['N'])
    params_dict['Jei_base'] = params_dict['jei']/np.sqrt(params_dict['N'])
    params_dict['Jii_base'] = params_dict['jii']/np.sqrt(params_dict['N'])
    
    # default Js set to baseline values, but can get changed by arousal
    params_dict['Jee'] = params_dict['Jee_base']
    params_dict['Jie'] = params_dict['Jie_base']
    params_dict['Jei'] = params_dict['Jei_base']
    params_dict['Jii'] = params_dict['Jii_base']
        
    params_dict['Jie_ext'] = params_dict['jie_ext']/np.sqrt(params_dict['N']) 
    params_dict['Jee_ext'] = params_dict['jee_ext']/np.sqrt(params_dict['N']) 
    params_dict['Jii_ext'] = params_dict['jii_ext']/np.sqrt(params_dict['N']) 
    params_dict['Jei_ext'] = params_dict['jei_ext']/np.sqrt(params_dict['N'])        
        
    return params_dict


def update_JplusAB(params_dict):

    if params_dict['net_type'] == 'hom':
        
        params_dict['JplusEE'] = 1.0       # EE intra-cluster potentiation factor
        params_dict['JplusII'] = 1.0       # II intra-cluster potentiation factor
        params_dict['JplusEI'] = 1.0       # EI intra-cluster potentiation factor
        params_dict['JplusIE'] = 1.0       # IE intra-cluster potentiation factor

    return params_dict



def fcn_update_clusterSize_std(params_dict):
    
    if ( (params_dict['clustE'] == 'hom') or (params_dict['clustI'] == 'hom')):
        
        # std of cluster size (as a fraction of mean)
        params_dict['clust_std'] = 0.0 
        
    return params_dict


def update_stim_params(params_dict):
    
    if ((params_dict['stim_shape'] == 'diff2exp')):
        
        params_dict['stim_duration'] = []
        
    return params_dict


def set_baseline_external_inputs_ei(params_dict):
    
    
    nu_ext_ee_base = params_dict['mean_nu_ext_ee']
    nu_ext_ie_base = params_dict['mean_nu_ext_ie']
    nu_ext_ei_base = params_dict['mean_nu_ext_ei']
    nu_ext_ii_base = params_dict['mean_nu_ext_ii']
    
    params_dict['nu_ext_ee'] = (nu_ext_ee_base)*np.ones(params_dict['N_e'])
    params_dict['nu_ext_ie'] = (nu_ext_ie_base)*np.ones(params_dict['N_i'])
    params_dict['nu_ext_ei'] = (nu_ext_ei_base)*np.ones(params_dict['N_e'])
    params_dict['nu_ext_ii'] = (nu_ext_ii_base)**np.ones(params_dict['N_i'])

    return params_dict


def set_max_stim_rate(params_dict):
        
    params_dict['stimRate_E'] = params_dict['stim_rel_amp']*params_dict['mean_nu_ext_ee']
    params_dict['stimRate_I'] = params_dict['stim_rel_amp']*params_dict['mean_nu_ext_ie']
    
    return params_dict


def fcn_set_popSizes(params_dict, popSize_E, popSize_I):
    
    params_dict['popSize_E'] = popSize_E.copy()
    params_dict['popSize_I'] = popSize_I.copy()

    return params_dict


#---------------------------------------------------------------------------------
# SET INITIAL VOLTAGE
#---------------------------------------------------------------------------------       
def fcn_set_initialVoltage(params_dict):
    
    print('initial voltages uniformly distributed between reset and threshold; could add rng seed as input here')
    iVe = np.random.uniform(params_dict['Vr_e'], params_dict['Vth_e'], params_dict['N_e'])
    iVi = np.random.uniform(params_dict['Vr_i'], params_dict['Vth_i'], params_dict['N_i'])
    params_dict['iV'] = np.append(iVe, iVi)

    return params_dict


#---------------------------------------------------------------------------------
# AROUSAL MODULATIONS
#---------------------------------------------------------------------------------  

def fcn_sigmoidArousal(x, L, k, xo):
    y = L/(1 + np.exp(-k*(x-xo)))
    return y

def fcn_BoundedSigmoidArousal(x, L, k, xo):
    '''
    L: y[1]
    k: steepness
    x0: y[0.5]/L
    '''
    if xo > L:
        sys.exit('xo cant be greater than L')
    alpha = np.log2( 1/( 1 + ((1-xo)/xo)**(1/k) ) )
    y = L/ ( 1 + ( x**alpha -1 )**k )
    return y

def fcn_linearArousal(n_arousalSamples, ymin, ymax):
    y = np.linspace(ymin, ymax, n_arousalSamples)
    return y


def fcn_define_arousalSweep(params_dict):
    
    
    arousal_variation = params_dict['arousal_variation']
    nParams_sweep = params_dict['nParams_sweep']
    arousal_levels = params_dict['arousal_levels']
    
    swept_params_dict = {}
    
    for iParam in range(0, nParams_sweep):
        
        dict_key_str = ( ('param_vals%d') % (iParam+1) )
        
    
        if arousal_variation == 'linear':
            
            linear_arousal_dict = params_dict['linear_arousal_dict']
            ymin, ymax = linear_arousal_dict[dict_key_str]
            y = np.interp(arousal_levels, np.array([0,1]), np.array([ymin,ymax]))
                
        elif arousal_variation == 'sigmoid':
            
            sigmoid_arousal_dict = params_dict['sigmoid_arousal_dict']
            xmin, xmax = sigmoid_arousal_dict['xrange'] 
            xvals = np.interp(arousal_levels, np.array([0,1]), np.array([xmin,xmax]))            
            L,k,xo = sigmoid_arousal_dict[dict_key_str]
            y = fcn_sigmoidArousal(xvals, L, k, xo)
            
        elif arousal_variation == 'bounded_sigmoid':

            sigmoid_arousal_dict = params_dict['sigmoid_arousal_dict']
            L,k,xo = sigmoid_arousal_dict[dict_key_str]
            xvals = arousal_levels.copy()
            y = fcn_BoundedSigmoidArousal(xvals, L, k, xo)            
            
        else:
            
            sys.exit('unknown arousal variation')
            
        
        swept_params_dict[dict_key_str] = y.copy()  

        
    params_dict['swept_params_dict'] = swept_params_dict.copy()


    return params_dict



#---------------------------------------------------------------------------------
# UPDATE PARAMETERS BASED ON AROUSAL
#---------------------------------------------------------------------------------  


def fcn_updateParams_givenArousal(params_dict, seed):
    
    params_dict = fcn_set_Je_reduction(params_dict)
    params_dict = set_perturbation_external_inputs_ei(params_dict, seed)
    params_dict = fcn_set_JeePlus_sweep(params_dict)
    
    return params_dict


def fcn_set_Je_reduction(params_dict):
    
    if 'Jee_reduction' in params_dict:
        params_dict['Jee'] = params_dict['Jee_base'] - params_dict['Jee_base']*params_dict['Jee_reduction']
    else:
        params_dict['Jee'] = params_dict['Jee_base'] 
        
    if 'Jie_reduction' in params_dict:
        params_dict['Jie'] = params_dict['Jie_base'] - params_dict['Jie_base']*params_dict['Jie_reduction']
    else:
        params_dict['Jie'] = params_dict['Jie_base']

    return params_dict


def set_perturbation_external_inputs_ei(params_dict, seed):

        
    if 'pert_inputType' in params_dict:
    
        if params_dict['pert_inputType'] =='homogeneous':
            
            params_dict['pert_nu_ext_ee'] = params_dict['pert_mean_nu_ext_ee']*np.ones(params_dict['N_e'])
            params_dict['pert_nu_ext_ie'] = params_dict['pert_mean_nu_ext_ie']*np.ones(params_dict['N_i'])
            params_dict['pert_nu_ext_ei'] = params_dict['pert_mean_nu_ext_ei']*np.ones(params_dict['N_e'])
            params_dict['pert_nu_ext_ii'] = params_dict['pert_mean_nu_ext_ii']*np.ones(params_dict['N_i'])
    
        elif params_dict['pert_inputType'] == 'beta':
            
            rng = np.random.default_rng(seed)

            params_dict['pert_nu_ext_ee'] = rng.beta(params_dict['beta_a'], params_dict['beta_b'], size = params_dict['N_e'])*params_dict['nu_ext_ee_beta_spread']
            params_dict['pert_nu_ext_ie'] = rng.beta(params_dict['beta_a'], params_dict['beta_b'], size = params_dict['N_i'])*params_dict['nu_ext_ie_beta_spread']
            params_dict['pert_nu_ext_ei'] = rng.beta(params_dict['beta_a'], params_dict['beta_b'], size = params_dict['N_e'])*params_dict['nu_ext_ei_beta_spread']
            params_dict['pert_nu_ext_ii'] = rng.beta(params_dict['beta_a'], params_dict['beta_b'], size = params_dict['N_i'])*params_dict['nu_ext_ii_beta_spread']
            
    
        elif params_dict['pert_inputType'] == 'uniform':
    
            rng = np.random.default_rng(seed)

            params_dict['pert_nu_ext_ee'] = rng.uniform(0, 1, size = params_dict['N_e'])*params_dict['nu_ext_ee_uniform_spread']
            params_dict['pert_nu_ext_ie'] = rng.uniform(0, 1, size = params_dict['N_i'])*params_dict['nu_ext_ie_uniform_spread']
            params_dict['pert_nu_ext_ei'] = rng.uniform(0, 1, size = params_dict['N_e'])*params_dict['nu_ext_ei_uniform_spread']
            params_dict['pert_nu_ext_ii'] = rng.uniform(0, 1, size = params_dict['N_i'])*params_dict['nu_ext_ii_uniform_spread']
            
        elif params_dict['pert_inputType'] == 'zeroMean_sd_pert':
            
            params_dict = fcn_set_zeroMean_sdPert(params_dict, seed)
            
        else:
            sys.exit('unknown pert_inputType')
            
    else:
        
        params_dict['pert_nu_ext_ee'] = np.zeros(params_dict['N_e'])
        params_dict['pert_nu_ext_ie'] = np.zeros(params_dict['N_i'])
        params_dict['pert_nu_ext_ei'] = np.zeros(params_dict['N_e'])
        params_dict['pert_nu_ext_ii'] = np.zeros(params_dict['N_i'])

   
    return params_dict


def fcn_set_JeePlus_sweep(params_dict):
    
    if 'JplusEE_sweep' in params_dict:    
        params_dict['JplusEE'] = params_dict['JplusEE_sweep']
        
    return params_dict
        

#%% function to set inputs for zeroMean_sdPert

def fcn_set_zeroMean_sdPert(params_dict, seed):
    
    rng = np.random.default_rng(seed)
    
    
    # setup perturbations-----------------------------------------------------
    
    if 'zeroMean_sd_nu_ext_ee' in params_dict:
        zee = rng.normal(loc = 0, scale = 1, size = params_dict['N_e'])*params_dict['zeroMean_sd_nu_ext_ee']
    else:
        zee = np.zeros((params_dict['N_e']))
    
    if 'zeroMean_sd_nu_ext_ie' in params_dict:
        zie = rng.normal(loc = 0, scale = 1, size = params_dict['N_i'])*params_dict['zeroMean_sd_nu_ext_ie']
    else:
        zie = np.zeros((params_dict['N_i']))
        
    if 'zeroMean_sd_nu_ext_ei' in params_dict:
        zei = rng.normal(loc = 0, scale = 1, size = params_dict['N_e'])*params_dict['zeroMean_sd_nu_ext_ei']
    else:
        zei = np.zeros((params_dict['N_e']))
        
    if 'zeroMean_sd_nu_ext_ii' in params_dict:
        zii = rng.normal(loc = 0, scale = 1, size = params_dict['N_i'])*params_dict['zeroMean_sd_nu_ext_ii']
    else:
        zii = np.zeros((params_dict['N_i']))
        

    # if we want the perturbation to be the same in each cluster---------------

    if 'zeroMean_sd_pert_sameAllCluster' in params_dict:
        
        if params_dict['zeroMean_sd_pert_sameAllCluster'] == True:
            
            N_e = params_dict['N_e']
            N_i = params_dict['N_i']
            popSize_I = params_dict['popSize_I'].copy()
            popSize_E = params_dict['popSize_E'].copy()
            p = params_dict['p']
            
            # EE--------------------------------------------------------------
            
            cluSize = popSize_E[0]
            cluBoundaries = np.append(0, np.cumsum( popSize_E ))
            zee_new = np.zeros(N_e)
            
            for indClu in range(0,p):
                zee_new[cluBoundaries[indClu]:cluBoundaries[indClu+1]] = zee[:cluSize]
                
            indClu = p
            zee_new[cluBoundaries[indClu]:cluBoundaries[indClu+1]] = zee[cluBoundaries[indClu]:cluBoundaries[indClu+1]]
            
            zee = zee_new.copy()
            
            # IE--------------------------------------------------------------
            
            cluSize = popSize_I[0]
            cluBoundaries = np.append(0, np.cumsum( popSize_I ))
            zie_new = np.zeros(N_i)
            
            for indClu in range(0,p):
                zie_new[cluBoundaries[indClu]:cluBoundaries[indClu+1]] = zie[:cluSize]
                
            indClu = p
            zie_new[cluBoundaries[indClu]:cluBoundaries[indClu+1]] = zie[cluBoundaries[indClu]:cluBoundaries[indClu+1]]
            
            zie = zie_new.copy()            
            
            # EI--------------------------------------------------------------
            
            cluSize = popSize_E[0]
            cluBoundaries = np.append(0, np.cumsum( popSize_E ))
            zei_new = np.zeros(N_e)
            
            for indClu in range(0,p):
                zei_new[cluBoundaries[indClu]:cluBoundaries[indClu+1]] = zei[:cluSize]
                
            indClu = p
            zei_new[cluBoundaries[indClu]:cluBoundaries[indClu+1]] = zei[cluBoundaries[indClu]:cluBoundaries[indClu+1]]
            
            zei = zei_new.copy()            
            

            # II--------------------------------------------------------------
            
            cluSize = popSize_I[0]
            cluBoundaries = np.append(0, np.cumsum( popSize_I ))
            zii_new = np.zeros(N_i)
            
            for indClu in range(0,p):
                zii_new[cluBoundaries[indClu]:cluBoundaries[indClu+1]] = zii[:cluSize]
                
            indClu = p
            zii_new[cluBoundaries[indClu]:cluBoundaries[indClu+1]] = zii[cluBoundaries[indClu]:cluBoundaries[indClu+1]]
            
            zii = zii_new.copy()    
            
    
    # perturbation external inputs---------------------------------------------
    pert_nu_ext_ee = zee*params_dict['mean_nu_ext_ee']
    pert_nu_ext_ie = zie*params_dict['mean_nu_ext_ie']
    pert_nu_ext_ei = zei*params_dict['mean_nu_ext_ei']
    pert_nu_ext_ii = zii*params_dict['mean_nu_ext_ii']
    
    # don't allow total inputs to go negative----------------------------------
    total_input_ee = params_dict['mean_nu_ext_ee'] + pert_nu_ext_ee
    neg_inds = np.nonzero(total_input_ee < 0)[0]
    pert_nu_ext_ee[neg_inds] = -params_dict['mean_nu_ext_ee']
    
    total_input_ie = params_dict['mean_nu_ext_ie'] + pert_nu_ext_ie 
    neg_inds = np.nonzero(total_input_ie < 0)[0]
    pert_nu_ext_ie[neg_inds] = -params_dict['mean_nu_ext_ie']
                                              
    total_input_ei = params_dict['mean_nu_ext_ei'] + pert_nu_ext_ei 
    neg_inds = np.nonzero(total_input_ei < 0)[0]
    pert_nu_ext_ei[neg_inds] = -params_dict['mean_nu_ext_ei']    

    total_input_ii = params_dict['mean_nu_ext_ii'] + pert_nu_ext_ii 
    neg_inds = np.nonzero(total_input_ii < 0)[0]
    pert_nu_ext_ii[neg_inds] = -params_dict['mean_nu_ext_ii']    
    
    # save perturbation inputs
    params_dict['pert_nu_ext_ee'] = pert_nu_ext_ee
    params_dict['pert_nu_ext_ie'] = pert_nu_ext_ie
    params_dict['pert_nu_ext_ei'] = pert_nu_ext_ei
    params_dict['pert_nu_ext_ii'] = pert_nu_ext_ii
    
    
    return params_dict
    
    

#---------------------------------------------------------------------------------
# COMPUTE WHICH NEURONS ARE STIMULATED 
#---------------------------------------------------------------------------------    

def fcn_compute_cluster_assignments(popsizeE, popsizeI):
    
    # number of E and I neurons
    Ne = np.sum(popsizeE)
    Ni = np.sum(popsizeI)
    
    # initialize outputs
    Ecluster_ids = np.zeros(Ne)
    Icluster_ids = np.zeros(Ni)
    
    # population start and end indices [E]
    pops_start_end = np.append(0, np.cumsum(popsizeE))
    
    # number of populations
    npops = np.size(pops_start_end)-1
        
    # loop over populations
    for popInd in range(0,npops,1):
    
        # cluster start and end
        startID = pops_start_end[popInd]
        endID = pops_start_end[popInd+1]
        
        Ecluster_ids[startID:endID] = popInd
        
    # population start and end indices [I]
    pops_start_end = np.append(0, np.cumsum(popsizeI))
    
    # number of populations
    npops = np.size(pops_start_end)-1
    
    # loop over populations
    for popInd in range(0,npops,1):
    
        # cluster start and end
        startID = pops_start_end[popInd]
        endID = pops_start_end[popInd+1]
        
        Icluster_ids[startID:endID] = popInd       
        
    return Ecluster_ids, Icluster_ids
        


def fcn_get_stimulated_neurons(params_dict, random_seed):       
    
    
    # set random number generator using the specified seed
    if random_seed == 'random':
        random_seed = np.random.choice(10000,1)
        rng = np.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng(random_seed)
        
    # boolean arrays that denote which neurons receive stimulus
    params_dict['stim_Ecells'] = np.zeros(params_dict['N_e'])
    params_dict['stim_Icells'] = np.zeros(params_dict['N_i'])
    
    
    # if homogeneous network
    if params_dict['net_type'] == 'hom':
        
        fracStim = params_dict['f_Ecells_target']*params_dict['f_selectiveClus']
        nStim = np.round(params_dict['N_e']*fracStim).astype(int)
        stim_cells = rng.choice(params_dict['N_e'], nStim, replace=False)
        params_dict['stim_Ecells'][stim_cells] = True

        fracStim = params_dict['f_Icells_target']*params_dict['f_selectiveClus']
        nStim = np.round(params_dict['N_i']*fracStim).astype(int)
        stim_cells = rng.choice(params_dict['N_i'], nStim, replace=False)
        params_dict['stim_Icells'][stim_cells] = True
    
    
    # else if clustered network
    elif params_dict['net_type'] == 'cluster':
        
        
        clust_sizeE = params_dict['popSize_E']
        clust_sizeI = params_dict['popSize_I']
        
    
        # get selective cluster ids
        selectiveClusters = params_dict['selectiveClusters']      
        
        # get assignment of neurons to clusters
        Ecluster_inds, Icluster_inds = fcn_compute_cluster_assignments(clust_sizeE,clust_sizeI)
        
        # loop over selective clusters
        for cluInd in selectiveClusters:
            
            #---------- Ecells -----------#
            
            # cells in this cluster
            cells_in_clu = np.nonzero(Ecluster_inds == cluInd)[0]
            
            # number to select
            nstim = np.round(params_dict['f_Ecells_target']*np.size(cells_in_clu),0).astype(int)
            
            # randomly select fraction of them
            stim_cells = rng.choice(cells_in_clu, \
                                    size = nstim, \
                                    replace=False)
            
            # update array
            params_dict['stim_Ecells'][stim_cells] = True
            
            
            #---------- Icells -----------#
            
            # cells in this cluster
            cells_in_clu = np.nonzero(Icluster_inds == cluInd)[0]
            
            # number to select
            nstim = np.round(params_dict['f_Icells_target']*np.size(cells_in_clu),0).astype(int)
            
            # randomly select fraction of them
            stim_cells = rng.choice(cells_in_clu, \
                                    size = nstim, \
                                    replace=False)
            
            # update array
            params_dict['stim_Icells'][stim_cells] = True       

            
    else:
        sys.exit('unknown network type')
       
        
    return params_dict


def fcn_setup_one_stimulus(params_dict, selectiveClusters_allStim, stim_ind, seed):
    
    params_dict['selectiveClusters'] = selectiveClusters_allStim[stim_ind].copy()
    params_dict = fcn_get_stimulated_neurons(params_dict, seed)
    
    return params_dict


