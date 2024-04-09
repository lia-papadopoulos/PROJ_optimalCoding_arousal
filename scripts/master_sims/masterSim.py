#%% MASTER SCRIPT FOR RUNNING SIMULATIONS WHERE NUeo IS VARIED

# THINGS TO POSSIBLY CHANGE

# RANDOM SEEDS
# INTEGRATION TYPE




#%% BASIC IMPORTS
import time
import argparse
import sys
import scipy as sp
from scipy.io import savemat

# IMPORT CONFIG FILE FOR SETTING PARAMETERS
from simParams import sim_params

# IMPORTS FROM FUNCTIONS FOLDER: SPECIFY PATH
sys.path.append('../functions/')    
import fcn_make_network_cluster
from fcn_simulation import fcn_simulate_whitenoise
from fcn_simulation import fcn_simulate_exact_poisson
from fcn_stimulation import get_stimulated_clusters

#%%
# main function
if __name__=='__main__':


    #%% INITIALIZE SIM PARAMS    
    s_params = sim_params()
        
    #%% SET PARAMETERS THAT CAN BE PASSED IN
    
    # initialize arg parser
    parser = argparse.ArgumentParser() 
    
    # perturbations
    parser.add_argument('-mean_nu_ext_e_offset', '--mean_nu_ext_e_offset', \
                        type=float, default = s_params.mean_nu_ext_e_offset)   

    parser.add_argument('-mean_nu_ext_i_offset', '--mean_nu_ext_i_offset', \
                        type=float, default = s_params.mean_nu_ext_i_offset)  
    
    parser.add_argument('-sd_nu_ext_e_pert', '--sd_nu_ext_e_pert', \
                        type=float, default = s_params.sd_nu_ext_e_pert)

    parser.add_argument('-sd_nu_ext_e_type', '--sd_nu_ext_e_type', \
                        type=str, default = '')
        
    parser.add_argument('-sd_nu_ext_e_white_pert', '--sd_nu_ext_e_white_pert', \
                        type=float, default = s_params.sd_nu_ext_e_white_pert)
        
    # stimulation
    parser.add_argument('-stim_rel_amp', '--stim_rel_amp', \
                        type=float, default = s_params.stim_rel_amp)
        
    # Jee+
    parser.add_argument('-JplusEE', '--JplusEE', \
                        type=float, default = s_params.JplusEE)
        
    # network type    
    parser.add_argument('-net_type', '--net_type', type=str, default=s_params.net_type)

    # other
    parser.add_argument('-ID','--ID',type=str, default='')
    parser.add_argument('-path_data','--path_data',type=str, default='')
    parser.add_argument('-ind_network', '--ind_network', type=int, default=0)
    parser.add_argument('-ind_IC', '--ind_IC', type=int, default=0)
    
    args = parser.parse_args()
    
    #%% UPDATE SIM PARAMS
    
    s_params.get_argparse_vals(args)
    s_params.update_JplusAB()
    s_params.set_dependent_vars()
    
    
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
    
    # seed for initial conditions ['random' or int]
    IC_seed = 'random'
    
    # seed for setting external inputs
    # setting to ind_network means that the same realization will be used for all trials [this is what we want!]
    extInput_seed = ind_network
        
    # whitenoise seed
    whiteNoise_seed = ind_network
    
    
    #%% MAKE NETWORK

    # make network using network index as random seed
    W, popSize_E, popSize_I = \
            fcn_make_network_cluster.fcn_make_network_cluster(s_params, ind_network)   
    print('network constructed')
    
    s_params.set_popSizes(popSize_E, popSize_I)
    
    
    #%% GET STIMULUS SELECTIVE CLUSTERS
    
    selectiveClusters = get_stimulated_clusters(s_params, get_stimulated_clusters_seed)
    

    #%% MAIN SIMULATION 
    
    #---------------------- LOOP OVER STIMULATION -------------------- #
    for stim_ind in range(0, s_params.nStim):
            
        #---------------------- TIMING ------------------------------- #
        t0 = time.time()
                
        #---------------------- EXTERNAL INPUTS ---------------------- #
        # set external inputs
        s_params.set_external_inputs(extInput_seed)
        
        #---------------------- STIMULATION -------------------------- #
        
        # set selective clusters
        s_params.selectiveClusters = selectiveClusters[stim_ind]
        
        # determine which neurons are stimulated [using stim_ind as random seed]
        s_params.get_stimulated_neurons(stim_ind, popSize_E, popSize_I)
        
        # determine maximum stimulus strength
        s_params.set_max_stim_rate()
        
        # if we are doing stim vs no stim set stim strength to zero for stim_ind=0
        if s_params.stim_type == 'stim_noStim':
            if s_params.nStim != 2:
                sys.exit('error: only 2 stimulus conditions possible for stim_type = stim_noStim')
            else:           
                s_params.stimRate_E = s_params.stimRate_E*stim_ind
                s_params.stimRate_I = s_params.stimRate_I*stim_ind
        
    
        #---------------------- RUN SIMULATION ----------------------- #
            
        # run simulation
        if s_params.save_voltage == 0:
            if s_params.sd_nu_ext_e_white_pert != 0:
                spikes = fcn_simulate_whitenoise(s_params, W, IC_seed, whiteNoise_seed)
                s_params.simulator = 'fcn_simulate_whitenoise'
            else:
                spikes = fcn_simulate_exact_poisson(s_params, W, IC_seed)
                s_params.simulator = 'fcn_simulate_exact_poisson'
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
                              'IC_seed':                        IC_seed, \
                              'whitenoiseSeed':                 whiteNoise_seed, \
                              'set_external_inputs_seed':       extInput_seed, \
                              'get_stimulated_clusters_seed':   get_stimulated_clusters_seed,\
                              'get_stimulated_neurons_seed':    stim_ind}
            
        if s_params.writeSimulation_to_file==True:
            results_dictionary['spikes'] = spikes
        
        # filename              
        if s_params.stim_type == 'stim_noStim':
            
            filename = ('%s%s_stim%d_stimType_%s_stim_rel_amp%0.3f_stim_noStim_%s.mat' % \
                       (path_data, ID, stim_ind, s_params.stim_shape, \
                       s_params.stim_rel_amp, s_params.parameters_fileName))
                
        elif s_params.stim_type == 'noStim':
            
            filename = ('%s%s_noStim_%s.mat' % (path_data, ID, s_params.parameters_fileName))      
                
        else:
            
            filename = ('%s%s_stim%d_stimType_%s_stim_rel_amp%0.3f_%s.mat' % \
                       (path_data, ID, stim_ind, s_params.stim_shape, \
                       s_params.stim_rel_amp, s_params.parameters_fileName))
                
                
        savemat(filename, results_dictionary)
        print('results written to file')
    

    
    #%% WRITE NETWORK TO FILE
    
    if (s_params.writeNetwork_to_file == 1):
        W = sp.sparse.csc_matrix(W)
        network_dict = {'W':                W, \
                        'popSize_E':      popSize_E, \
                        'popSize_I':      popSize_I, \
                        'network_seed':     ind_network}
        savemat(net_filename, network_dict)
        print('network written to file')

    print('done!')
    
    
    
    
    
    
