
import sys
import numpy as np
sys.path.append('../functions') 
from fcn_make_network_cluster import fcn_compute_cluster_assignments

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
    params_dict['Jee_reduction'] = args.Jee_reduction
    params_dict['Jie_reduction'] = args.Jie_reduction
        
    
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

    params_dict['Jee'] = params_dict['jee']/np.sqrt(params_dict['N'])
    params_dict['Jie'] = params_dict['jie']/np.sqrt(params_dict['N'])
    params_dict['Jei'] = params_dict['jei']/np.sqrt(params_dict['N'])
    params_dict['Jii'] = params_dict['jii']/np.sqrt(params_dict['N'])
        
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
    
    if params_dict['stim_type'] == 'noStim':      # set stim strength to zero if stim type is 'noStim'
        
        params_dict['nStim'] = 1
        params_dict['stim_rel_amp'] = 0
        
        
    if ((params_dict['stim_shape'] == 'diff2exp')):
        
        params_dict['stim_duration'] = []
        
        
    return params_dict


def set_baseline_external_inputs_ei(params_dict):
    
    
    nu_ext_ee_base = params_dict['mean_nu_ext_ee']
    nu_ext_ie_base = params_dict['mean_nu_ext_ie']
    nu_ext_ei_base = params_dict['mean_nu_ext_ei']
    nu_ext_ii_base = params_dict['mean_nu_ext_ii']
    
    params_dict['nu_ext_ee'] = nu_ext_ee_base*np.ones(params_dict['N_e'])
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



def fcn_set_arousal(params_dict, seed):
    
    
    params_dict = fcn_set_Je_reduction(params_dict)
    params_dict = set_perturbation_external_inputs_ei(params_dict, seed)
    
    return params_dict


def fcn_set_Je_reduction(params_dict):
    
    params_dict['Jee'] = params_dict['Jee'] - params_dict['Jee']*params_dict['Jee_reduction']
    params_dict['Jie'] = params_dict['Jie'] - params_dict['Jie']*params_dict['Jie_reduction']

    return params_dict


def set_perturbation_external_inputs_ei(params_dict, seed):

    
    rng = np.random.default_rng(seed)
    
    if params_dict['pert_inputType'] =='homogeneous':
        
        params_dict['pert_nu_ext_ee'] = params_dict['pert_mean_nu_ext_ee']*np.ones(params_dict['N_e'])
        params_dict['pert_nu_ext_ie'] = params_dict['pert_mean_nu_ext_ie']*np.ones(params_dict['N_i'])
        params_dict['pert_nu_ext_ei'] = params_dict['pert_mean_nu_ext_ei']*np.ones(params_dict['N_e'])
        params_dict['pert_nu_ext_ii'] = params_dict['pert_mean_nu_ext_ii']*np.ones(params_dict['N_i'])


    elif params_dict['pert_inputType'] == 'beta':
        
        params_dict['pert_nu_ext_ee'] = rng.beta(params_dict['beta_a'], params_dict['beta_b'], size = params_dict['N_e'])*params_dict['nu_ext_ee_beta_spread']
        params_dict['pert_nu_ext_ie'] = rng.beta(params_dict['beta_a'], params_dict['beta_b'], size = params_dict['N_i'])*params_dict['nu_ext_ie_beta_spread']
        params_dict['pert_nu_ext_ei'] = rng.beta(params_dict['beta_a'], params_dict['beta_b'], size = params_dict['N_e'])*params_dict['nu_ext_ei_beta_spread']
        params_dict['pert_nu_ext_ii'] = rng.beta(params_dict['beta_a'], params_dict['beta_b'], size = params_dict['N_i'])*params_dict['nu_ext_ii_beta_spread']
        


    elif params_dict['pert_inputType'] == 'uniform':
        
        params_dict['pert_nu_ext_ee'] = rng.uniform(0, 1, size = params_dict['N_e'])*params_dict['nu_ext_ee_uniform_spread']
        params_dict['pert_nu_ext_ie'] = rng.uniform(0, 1, size = params_dict['N_i'])*params_dict['nu_ext_ie_uniform_spread']
        params_dict['pert_nu_ext_ei'] = rng.uniform(0, 1, size = params_dict['N_e'])*params_dict['nu_ext_ei_uniform_spread']
        params_dict['pert_nu_ext_ii'] = rng.uniform(0, 1, size = params_dict['N_i'])*params_dict['nu_ext_ii_uniform_spread']
        

   
    return params_dict
        

#---------------------------------------------------------------------------------
# COMPUTE WHICH NEURONS ARE STIMULATED 
#---------------------------------------------------------------------------------    
def fcn_get_stimulated_neurons(params_dict, random_seed):       
    
    # boolean arrays that denote which neurons receive stimulus
    params_dict['stim_Ecells'] = np.zeros(params_dict['N_e'])
    params_dict['stim_Icells'] = np.zeros(params_dict['N_i'])
    clust_sizeE = params_dict['popSize_E']
    clust_sizeI = params_dict['popSize_I']
    
    # set random number generator using the specified seed
    if random_seed == 'random':
        random_seed = np.random.choice(10000,1)
        rng = np.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng(random_seed)

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
       
        
    return params_dict


def fcn_setup_one_stimulus(params_dict, selectiveClusters_allStim, stim_ind, seed):
    
    params_dict['selectiveClusters'] = selectiveClusters_allStim[stim_ind].copy()
    
    params_dict = fcn_get_stimulated_neurons(params_dict, seed)
    
    return params_dict


