
"""
settings file for spont_cvISI_vsPerturbation
"""


sim_params_path = '/home/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/scripts/master_sims/'

func_path0 = '/home/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/scripts/master_sims/'


func_path = '/home/liap/PostdocWork_Oregon/My_Projects/' \
            'PROJ_VariabilityGainMod/scripts/functions/'

load_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/'\
             'PROJ_VariabilityGainMod/data_files/test_stim_expSyn/')
    
load_path_plotting = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/test_stim_expSyn/spont_cvISI_vsPerturbation/')


save_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/test_stim_expSyn/spont_cvISI_vsPerturbation/')

save_path_plotting = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/Figures/test_stim_expSyn/spont_cvISI_vsPerturbation/')
    

    
#%% simulations params
   
simParams_fname = 'simParams_051325_clu_spont'
sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'
net_type = 'baseEIclu'
nNetworks = 2

# window length/steps for computing spike counts
windL = 2500e-3



#%% cluster

maxCores = 48
cores_per_job = 4 # needs to be set ahead of time using OMP_NUM_THREADS

#%% plotting

rate_thresh = 1.
