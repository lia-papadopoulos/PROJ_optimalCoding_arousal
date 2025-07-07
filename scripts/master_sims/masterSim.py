

#%% BASIC IMPORTS
import time
import argparse
import sys
import scipy as sp
from scipy.io import savemat
import importlib

import paths_file

# UNPACK PATHS
sim_params_path = paths_file.sim_params_path
sim_params_name = paths_file.sim_params_name
functions_path1 = paths_file.functions_path1

# IMPORT CONFIG FILE FOR SETTING PARAMETERS
sys.path.append(sim_params_path) 
params = importlib.import_module(sim_params_name) 
s_params = params.sim_params
del params

# FROM WORKING DIRECTORY
from fcn_simulation_setup import fcn_basic_setup, fcn_set_popSizes, fcn_set_initialVoltage, fcn_updateParams_givenArousal, fcn_setup_one_stimulus

# IMPORTS FROM FUNCTIONS FOLDER: SPECIFY PATH
sys.path.append(functions_path1)         
import fcn_make_network_cluster
from fcn_simulation_EIextInput import fcn_simulate_expSyn
from fcn_stimulation import get_stimulated_clusters

#%%
# main function
if __name__=='__main__':

        
    #%% SET PARAMETERS THAT CAN BE PASSED IN
    
    # initialize arg parser
    parser = argparse.ArgumentParser() 
    
    # external input perturbations
    parser.add_argument('-pert_mean_nu_ext_ee', '--pert_mean_nu_ext_ee', \
                        type=float, default = 0.)   
        
    parser.add_argument('-pert_mean_nu_ext_ie', '--pert_mean_nu_ext_ie', \
                        type=float, default = 0.)   
       
    parser.add_argument('-pert_mean_nu_ext_ei', '--pert_mean_nu_ext_ei', \
                        type=float, default = 0.)   
        
    parser.add_argument('-pert_mean_nu_ext_ii', '--pert_mean_nu_ext_ii', \
                        type=float, default = 0.)   


    parser.add_argument('-nu_ext_ee_uniform_spread', '--nu_ext_ee_uniform_spread', \
                        type=float, default = 0.)   
        
    parser.add_argument('-nu_ext_ie_uniform_spread', '--nu_ext_ie_uniform_spread', \
                        type=float, default = 0.)   

    parser.add_argument('-nu_ext_ei_uniform_spread', '--nu_ext_ei_uniform_spread', \
                        type=float, default = 0.)   

    parser.add_argument('-nu_ext_ii_uniform_spread', '--nu_ext_ii_uniform_spread', \
                        type=float, default = 0.)   


    parser.add_argument('-nu_ext_ee_beta_spread', '--nu_ext_ee_beta_spread', \
                        type=float, default = 0.)   
        
    parser.add_argument('-nu_ext_ie_beta_spread', '--nu_ext_ie_beta_spread', \
                        type=float, default = 0.)   

    parser.add_argument('-nu_ext_ei_beta_spread', '--nu_ext_ei_beta_spread', \
                        type=float, default = 0.)   

    parser.add_argument('-nu_ext_ii_beta_spread', '--nu_ext_ii_beta_spread', \
                        type=float, default = 0.)   
        
    # synaptic strength reductions    
    parser.add_argument('-Jee_reduction', '--Jee_reduction', type=float, default = 0.)
    parser.add_argument('-Jie_reduction', '--Jie_reduction', type=float, default = 0.)

    # JplusEE
    parser.add_argument('-JplusEE_sweep', '--JplusEE_sweep', type=float, default = 1.)
    
    # zero mean sd_nu_ext
    parser.add_argument('-zeroMean_sd_nu_ext_ee', '--zeroMean_sd_nu_ext_ee', type=float, default = 0.)
    

    # network type    
    parser.add_argument('-net_type', '--net_type', type=str, default=s_params['net_type'])

    # other
    parser.add_argument('-ID','--ID',type=str, default='')
    parser.add_argument('-path_data','--path_data',type=str, default='')
    parser.add_argument('-ind_network', '--ind_network', type=int, default=0)
    parser.add_argument('-ind_IC', '--ind_IC', type=int, default=0)
    
    args = parser.parse_args()
    
    
    #%% SETUP SIMULATION
    
    s_params = fcn_basic_setup(s_params, args)
    
    
    #%% GET RELEVANT ARGS
    
    ID = args.ID
    path_data = args.path_data  
    
    # network index
    ind_network = args.ind_network
    
    # initial condition (trial) index
    ind_IC = args.ind_IC
    
    # network filename
    net_filename = path_data + 'network' + str(ind_network) + '.mat'
    
    
    #%% SET ALL RANDOM SEEDS
    
    # which clusters to stimulate [use network index as random seed so all ICs have same stimuli]
    get_stimulated_clusters_seed = ind_network
    
    # seed for setting external inputs
    # setting to ind_network means that the same realization will be used for all trials [this is what we want]
    extInput_seed = ind_network


    #%% GET POPULATION SIZES
    
    _, popSize_E, popSize_I = fcn_make_network_cluster.fcn_make_network_cluster(s_params, ind_network)   
    s_params = fcn_set_popSizes(s_params, popSize_E, popSize_I)

    
    #%% AROUSAL PARAMETERS
    
    s_params = fcn_updateParams_givenArousal(s_params, extInput_seed)


    #%% MAKE NETWORK

    # make network using network index as random seed
    W, _, _ = fcn_make_network_cluster.fcn_make_network_cluster(s_params, ind_network)   
    print('network constructed')
    
    
    #%% GET STIMULUS SELECTIVE CLUSTERS
    
    selectiveClusters = get_stimulated_clusters(s_params, get_stimulated_clusters_seed)
    

    #%% MAIN SIMULATION 
    
    #---------------------- LOOP OVER STIMULATION -------------------- #
    for stim_ind in range(0, s_params['nStim']):
            
        t0 = time.time()
        
        #---------------------- INITIAL CONDITIONS ---------------------#
        s_params = fcn_set_initialVoltage(s_params)
        
        #---------------------- STIMULATION -------------------------- #
        # set selective clusters
        # determine which neurons are stimulated [using stim_ind as random seed]
        stimNeurons_seed = stim_ind
        s_params = fcn_setup_one_stimulus(s_params, selectiveClusters, stim_ind, stimNeurons_seed)
        
        #---------------------- RUN SIMULATION ----------------------- #
            
        # run simulation
        if s_params['save_voltage'] == 0:
            spikes = fcn_simulate_expSyn(s_params, W)
            s_params['simulator'] = 'fcn_simulate_expSyn'
            
        else:
            sys.exit('do not save voltage for each simulation')
            
        print('simulation done')
           
        #---------------------- TIMING ------------------------------- #
        tf = time.time()
        print('sim time = %0.3f seconds' %(tf-t0)) 
            
        
        #---------------------- TIMING ------------------------------- #
        
        results_dictionary = {'sim_params':                     s_params, \
                              'spikes':                         [], \
                              'popSize_E':                    popSize_E, \
                              'popSize_I':                    popSize_I, \
                              'network_seed':                   ind_network, \
                              'set_external_inputs_seed':       extInput_seed, \
                              'get_stimulated_clusters_seed':   get_stimulated_clusters_seed,\
                              'get_stimulated_neurons_seed':    stim_ind}
            
        if s_params['writeSimulation_to_file']==True:
            results_dictionary['spikes'] = spikes
  
            
        # filename
        filename = ('%s%s_stim%d_stimType_%s_stim_rel_amp%0.3f_%s.mat' % \
                       (path_data, ID, stim_ind, s_params['stim_shape'], \
                       s_params['stim_rel_amp'], s_params['parameters_fileName']))
                
                
        savemat(filename, results_dictionary)
        print('results written to file')
    

    
    #%% WRITE NETWORK TO FILE
    
    if (s_params['writeNetwork_to_file'] == 1):
        W = sp.sparse.csc_matrix(W)
        network_dict = {'W':                W, \
                        'popSize_E':      popSize_E, \
                        'popSize_I':      popSize_I, \
                        'network_seed':     ind_network}
        savemat(net_filename, network_dict)
        print('network written to file')

    print('done!')
    
    
    
    
    
    
