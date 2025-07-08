

# path to functions

sim_params_path = '/home/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/scripts/master_sims/'


func_path0 = '/home/liap/PostdocWork_Oregon/My_Projects/' \
             'PROJ_VariabilityGainMod/scripts/master_sims/'

func_path1 = '/home/liap/PostdocWork_Oregon/My_Projects/' \
            'PROJ_VariabilityGainMod/scripts/functions/'
            
func_path2 = '/home/liap/PostdocWork_Oregon/My_Projects/' \
            'PROJ_VariabilityGainMod/scripts/analysis_SuData/'
            
# load path
load_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/'\
             'PROJ_VariabilityGainMod/data_files/test_stim_expSyn/')
    
# decode path
decode_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/test_stim_expSyn/decoding_analysis/')


# save path
save_path = ('/mnt/data0/liap/PostdocWork_Oregon/My_Projects/'\
             'PROJ_VariabilityGainMod/data_files/test_stim_expSyn/evoked_corr/')
    

    
#%% simulations params
   
simParams_fname = 'simParams_051325_clu'
#simParams_fname = 'simParams_051325_hom'
sweep_param_name = 'Jee_reduction_nu_ext_ee_beta_spread_nu_ext_ie_beta_spread'
net_type = 'baseEIclu'
#net_type = 'baseHOM'
nNetworks = 10


# window length/steps for computing spike counts
windL = 100e-3
windStep = windL
baseWind_burn = 0.2

# number of cells to subsample
nCells_sample = 144


# nShuffles
nShuffles = 100
nSamples = 10


# decoding info
decode_windL = 100e-3
decode_ensembleSize = 160
decode_rateThresh = 0.
decode_classifier = 'LinearSVC'

# decode window or after stim window
use_decode_window = False


#%% cluster job submission

# for task spooler
maxCores = 48
cores_per_job = 4 # needs to be set ahead of time using OMP_NUM_THREADS

