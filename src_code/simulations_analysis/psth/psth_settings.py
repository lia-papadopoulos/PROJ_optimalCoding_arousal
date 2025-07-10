
"""
psth settings
"""

#%% paths


sim_params_path = '/home/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/scripts/master_sims/'


func_path0 = '/home/liap/PostdocWork_Oregon/My_Projects/' \
             'PROJ_VariabilityGainMod/scripts/master_sims/'

func_path1 = '/home/liap/PostdocWork_Oregon/My_Projects/' \
            'PROJ_VariabilityGainMod/scripts/functions/'

# load path
load_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/'\
             'PROJ_VariabilityGainMod/data_files/test_stim_expSyn/')

save_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/'\
              'PROJ_VariabilityGainMod/data_files/test_stim_expSyn/psth/')
    
save_path_plotting = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/'\
                     'PROJ_VariabilityGainMod/data_files/Figures/test_stim_expSyn/psth/')
    
    
#%% simulations params

simParams_fname = 'simParams_051325_clu'
#simParams_fname = 'simParams_051325_hom'

sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'
net_type = 'baseEIclu'
#net_type = 'baseHOM'
nNetworks = 10
nStim = 5

# baseline window
base_window = [-0.8, 0]

# stimulus window
stim_window = [0, 0.2]

# bin size
binSize = 100e-3

# step size
stepSize = 5e-3

# burn time
burnTime = 0

#%% cluster

maxCores = 48
cores_per_job = 4 # needs to be set ahead of time using OMP_NUM_THREADS

