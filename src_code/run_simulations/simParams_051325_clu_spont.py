
import numpy as np

sim_params = {}


#-----------------------------------------------------------------------------
# SIMULATION PROPERTIES
#----------------------------------------------------------------------------- 

sim_params['simID'] = '051300002025_clu_spont' 

sim_params['parameters_fileName'] = ('simulationData')
sim_params['network_name'] = 'network'

sim_params['save_voltage'] = False               # whether or not to save membrane potential, input current 
sim_params['writeNetwork_to_file'] = False       # write network to file?    
sim_params['writeSimulation_to_file'] = True     # write simulation to file?

sim_params['T0'] = 0.                           # simulation start time
sim_params['TF'] = 2.7                          # simulation end time
sim_params['dt'] = 0.05e-3                      # time step


#-----------------------------------------------------------------------------
# NEURON PROPERTIES
#----------------------------------------------------------------------------- 

sim_params['Vth_e'] = 1.5                 # excitatory threshold
sim_params['Vth_i'] = 0.75                # inhibitory threshold
sim_params['Vr_e'] = 0.                      # reset potential E
sim_params['Vr_i'] = 0.                      # reset potential I

sim_params['tau_m_e'] = 20e-3                # membrane time constant E
sim_params['tau_m_i'] = 20e-3                # membrane time constant I  
sim_params['tau_s_e'] = 5e-3
sim_params['tau_s_i'] = 5e-3         
sim_params['tau_r'] = 5e-3                   # refractory period
sim_params['t_delay'] = 0e-3              # delay

sim_params['synType'] = 'exp'                # synapse type


#-----------------------------------------------------------------------------
# BASELINE INPUTS
#----------------------------------------------------------------------------- 

sim_params['base_extCurrent_poisson'] = True
sim_params['pert_extCurrent_poisson'] = True
sim_params['pert_toVoltage'] = False

sim_params['mean_nu_ext_ee'] = 7.
sim_params['mean_nu_ext_ie'] = 7.
sim_params['mean_nu_ext_ei'] = 0.
sim_params['mean_nu_ext_ii'] = 0.

sim_params['pext_ee'] = 0.2
sim_params['pext_ie'] = 0.2
sim_params['pext_ei'] = 0.8
sim_params['pext_ii'] = 0.8

sim_params['jee_ext'] = 2.3
sim_params['jie_ext'] = 2.3
sim_params['jei_ext'] = 2.3
sim_params['jii_ext'] = 2.3


#-----------------------------------------------------------------------------
# NETWORK
#----------------------------------------------------------------------------- 

# size
sim_params['N'] = 2000                       # total number of neurons
sim_params['ne'] = 0.8                       # total fraction of excitatory neurons

# cluster or hom
sim_params['net_type'] ='cluster'

# network connection type        
sim_params['connType'] ='fixed_inDegree' 

# whether or not to depress inter-cluster connections
sim_params['depress_interCluster'] = True

sim_params['pee'] = 0.2
sim_params['pei'] = 0.5
sim_params['pii'] = 0.5
sim_params['pie'] = 0.5

sim_params['jee'] = 0.63
sim_params['jie'] = 0.63
sim_params['jei'] = 1.9
sim_params['jii'] = 3.8
        

# clusters
sim_params['p'] = 18
sim_params['bgrE'] = 0.1
sim_params['bgrI'] = 0.1

# which neurons & weights are clustered
sim_params['clusters'] = ['E','I']
sim_params['clusterWeights'] = ['EE','EI','IE','II']

# other cluster properties (probably wont change much)
sim_params['Ecluster_weightSize'] = False

# cluster size heterogeneity ('hom' or'het')
sim_params['clustE'] ='hom' # E clusters 
sim_params['clustI'] ='hom' # I clusters 
sim_params['clust_std'] = 1.0 # if heterogeneous


# cluster depression & potentiation
sim_params['JplusEE'] = 15.75             
sim_params['JplusII'] = 5.            
sim_params['JplusEI'] = 6.25             
sim_params['JplusIE'] = 5.45            

# variance in synaptic weights
sim_params['deltaEE'] = 0
sim_params['deltaEI'] = 0
sim_params['deltaIE'] = 0
sim_params['deltaII'] = 0


#-----------------------------------------------------------------------------
# STIMULUS PROPERTIES
#----------------------------------------------------------------------------- 

sim_params['stim_type'] = ''           # type of stimulus ['' or 'noStim']
sim_params['nStim'] = 1                      # number of different stimuli to run
sim_params['mixed_selectivity'] = True       # allow different stimuli to target same clusters
sim_params['stim_shape'] = 'diff2exp'        # type of stimulus
sim_params['stim_onset'] = sim_params['T0'] + 1.0     # stimulus onset
sim_params['f_selectiveClus'] = 0.5          # fraction of clusters that are selective to each stimulus
sim_params['f_Ecells_target'] = 0.5          # fraction E cells targeted in selective clusters
sim_params['f_Icells_target'] = 0.0          # fraction of I cells targeted in selective clsuters
sim_params['stim_rel_amp'] = 0.0            # relative strength (fraction above baseline)

        
# for box and linear
sim_params['stim_duration'] = 2.             # duration of stimulus in seconds
# for difference of exponentials
sim_params['stim_taur'] = 0.075
sim_params['stim_taud'] = 0.1
        

#-----------------------------------------------------------------------------
# PERTURBATIONS
#----------------------------------------------------------------------------- 

# type of input perturbation (homogeneous, beta, uniform)
sim_params['pert_inputType'] = 'beta'


# homogeneous inputs
sim_params['pert_mean_nu_ext_ee'] = 0.0
sim_params['pert_mean_nu_ext_ie'] = 0.0
sim_params['pert_mean_nu_ext_ei'] = 0.0
sim_params['pert_mean_nu_ext_ii'] = 0.0

# beta inputs
sim_params['beta_a'] = 10.0
sim_params['beta_b'] = 10.0
sim_params['nu_ext_ee_beta_spread'] = 0.0
sim_params['nu_ext_ie_beta_spread'] = 0.0
sim_params['nu_ext_ei_beta_spread'] = 0.0
sim_params['nu_ext_ii_beta_spread'] = 0.0

# uniform inputs
sim_params['nu_ext_ee_uniform_spread'] = 0.0
sim_params['nu_ext_ie_uniform_spread'] = 0.0
sim_params['nu_ext_ei_uniform_spread'] = 0.0
sim_params['nu_ext_ii_uniform_spread'] = 0.0

# synaptic reductions
sim_params['Jee_reduction'] = 0.0
sim_params['Jie_reduction'] = 0.0

# external variance
sim_params['zeroMean_sd_nu_ext_ee'] = 0.0



#-----------------------------------------------------------------------------
# PERTURBATION SWEEP
#----------------------------------------------------------------------------- 

# number of parameters in arousal model
sim_params['nParams_sweep'] = 3

# names of parameters in arousal model
swept_param_name_dict = {}
swept_param_name_dict['param_vals1'] = 'Jee_reduction'
swept_param_name_dict['param_vals2'] = 'nu_ext_ee_beta_spread'
swept_param_name_dict['param_vals3'] = 'nu_ext_ie_beta_spread'
sim_params['swept_param_name_dict'] = swept_param_name_dict

# parameters for arousal variation
sim_params['arousal_variation'] = 'bounded_sigmoid'
sim_params['arousal_levels'] = np.linspace(0,1,11)


sigmoid_arousal_dict = {}
sigmoid_arousal_dict['xrange'] = (0,1)
sigmoid_arousal_dict['param_vals1'] = (0.75, 1.25, 0.2)
sigmoid_arousal_dict['param_vals2'] = (13.125, 1.25, 0.2)
sigmoid_arousal_dict['param_vals3'] = (13.125, 1.25, 0.2)
sim_params['sigmoid_arousal_dict'] = sigmoid_arousal_dict


#%% ONLY CHANGE THESE

#-----------------------------------------------------------------------------
# COMPUTING STUFF
#----------------------------------------------------------------------------- 
sim_params['path_data'] = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/'\
                           'PROJ_VariabilityGainMod/data_files/test_stim_expSyn/')
    
# number of cores to use
sim_params['maxCores'] = 64
sim_params['cores_per_job'] = 4 # needs to be set ahead of time using OMP_NUM_THREADS

# network index at which to start simulations
sim_params['indNet_start'] = 0
# number of networks to run
sim_params['n_networks'] = 2
# number of initial conditions per network
sim_params['n_ICs'] = 30
