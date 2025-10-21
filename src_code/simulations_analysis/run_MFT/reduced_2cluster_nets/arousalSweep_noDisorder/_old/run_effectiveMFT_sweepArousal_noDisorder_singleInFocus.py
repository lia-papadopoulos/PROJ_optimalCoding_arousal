
"""
EFFECTIVE MFT FOR NETWORKS WITH 2 E CLUSTERS
"""

#%% imports

# import settings
import effectiveMFT_sweepArousal_noDisorder_settings as settings

# basic imports
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('agg')
from scipy.io import savemat
from scipy.io import loadmat
import argparse

#%% unpack settings

func_path1 = settings.func_path1
func_path2 = settings.func_path2
func_path3 = settings.func_path3
simParams_path = settings.simParams_path
fig_outpath = settings.fig_outpath
data_outpath = settings.data_outpath
fName_begin = settings.fName_begin

# for arousal
arousalParams_path = settings.arousalParams_path
arousalParams_fname = settings.arousalParams_fname
sweep_param_name = settings.sweep_param_name
n_sweepParams = settings.n_sweepParams
arousal_multFactor = settings.arousal_multFactor
load_arousalParams_from_settings = settings.load_arousalParams_from_settings



#%% load my own functions

from mftParams_arousalSweep_noDisorder_singleInFocus import mft_params

sys.path.append(func_path1)     
from fcn_make_network_2cluster import fcn_make_network_cluster
from fcn_simulation_loading import fcn_set_sweepParam_string

sys.path.append(func_path2) 
import fcn_MFT_fixedInDeg_clusNet

sys.path.append(func_path3) 
import fcn_effectiveMFT_fixedInDeg_clusNet

sys.path.append(simParams_path)
from simParams_arousalSweep_noDisorder import sim_params

#%% argparser
# parser
parser = argparse.ArgumentParser() 
# possible swept parameters used in launch jobs
parser.add_argument('-param_indx', '--param_indx', type=int, required=True)
# arguments of parser
args = parser.parse_args()

#-------------------- argparser values for later use -------------------------#
param_indx = args.param_indx

#%% arousal parameters
if load_arousalParams_from_settings == True:
    ### load arousal parameters from settings file
    swept_params_dict = settings.swept_params_dict
else:
    ### load araousal parameters from file
    arousalParams = loadmat(arousalParams_path + arousalParams_fname, simplify_cells = True)
    swept_params_dict = arousalParams['swept_params_dict_mft']
    ### update arousal parameters based on multiplication factor for reduced network mft
    ### factor should be calibrated st arousal parameters are varied within relevant range for reduced network
    for i in range(1, n_sweepParams+1):
        dict_key = 'param_vals%d' % i
        swept_params_dict[dict_key] = swept_params_dict[dict_key]*arousal_multFactor


#%% exit if param_indx is greater than number of arousal samples
if param_indx > len(swept_params_dict['param_vals1']) - 1:
    sys.exit(0)


#%% initialize simulation and analysis parameters classes
params = sim_params()
m_params = mft_params()    

#%% setting up

# sweep param string
sweep_param_str_val = fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, param_indx) 
# set arousal variables
params.fcn_set_arousalVars(sweep_param_name, swept_params_dict, param_indx)
# set other dependent variables
params.set_dependent_vars()
# make network
W, popsizeE, popsizeI = fcn_make_network_cluster(params)    
# set population sizes 
params.popsizeE = popsizeE
params.popsizeI = popsizeI

print('setup done')


#%% MFT

#%%

print('selective fixed point with root solver:')
# selective fixed point
m_params.nu_vec = m_params.nu_vec_cluster.copy()
results  = fcn_MFT_fixedInDeg_clusNet.fcn_master_MFT(params, m_params)
selective_fixedPoint = np.append(results['nu_e'], results['nu_i']) 
selective_fixedPoint_inFocus_e = results['nu_e'][m_params.inFocus_pops_e]
print(selective_fixedPoint)


#%% non-selective fixed point

print('saddle point with root solver:')
m_params.nu_vec = m_params.nu_vec_uniform.copy()
results = fcn_MFT_fixedInDeg_clusNet.fcn_master_MFT(params, m_params)

saddlePoint = np.append(results['nu_e'], results['nu_i']) 
saddlePoint_inFocus_e = results['nu_e'][m_params.inFocus_pops_e]

print(saddlePoint)


#%% verify the answer for the selective fixed point is the same with the dynamical equations

print('selective fixed point with dyn eqs:')
m_params.nu_vec = selective_fixedPoint + np.array([0.1,0,0,0])
results  = fcn_MFT_fixedInDeg_clusNet.fcn_master_MFT_DynEqs(params, m_params)
selective_fixedPoint_dyn = np.append(results['nu_e'], results['nu_i']) 
print(selective_fixedPoint_dyn)


#%% verify the answer for the saddle point is the same with the dynamical equations

print('saddle point with dyn eqs:')
m_params.nu_vec = saddlePoint + np.array([0.1,0,0,0])
results  = fcn_MFT_fixedInDeg_clusNet.fcn_master_MFT_DynEqs(params, m_params)
saddlePoint_dyn = np.append(results['nu_e'], results['nu_i']) 
print(saddlePoint_dyn)


#%% check effective theory with dynamical equations: selective

print('selective fixed point with effective dynamical:')
m_params.nu_vec = selective_fixedPoint.copy()
nu_vec_in = m_params.nu_vec.copy()

inFocus_nu_e_out, inFocus_nu_i_out, outFocus_nu_e, outFocus_nu_i = \
    fcn_effectiveMFT_fixedInDeg_clusNet.fcn_master_MFT_DynEqs(params, m_params)

print('rate out - rate in')
print(inFocus_nu_e_out[0] - nu_vec_in[0])


print('selective fixed point with effective dynamical, start away from fp:')
m_params.nu_vec = selective_fixedPoint + np.array([0.1,0,0,0])
nu_vec_in = m_params.nu_vec.copy()

inFocus_nu_e_out, inFocus_nu_i_out, outFocus_nu_e, outFocus_nu_i = \
    fcn_effectiveMFT_fixedInDeg_clusNet.fcn_master_MFT_DynEqs(params, m_params)

print('rate out - rate in')
print(inFocus_nu_e_out[0] - nu_vec_in[0])

#%% check effective theory with dynamical equations: saddle

print('saddle fixed point with effective dynamical:')
m_params.nu_vec = saddlePoint.copy()
nu_vec_in = m_params.nu_vec.copy()

inFocus_nu_e_out, inFocus_nu_i_out, outFocus_nu_e, outFocus_nu_i = \
    fcn_effectiveMFT_fixedInDeg_clusNet.fcn_master_MFT_DynEqs(params, m_params)

print('rate out - rate in')
print(inFocus_nu_e_out[0] - nu_vec_in[0])


print('saddle fixed point with effective dynamical, start away:')
m_params.nu_vec = saddlePoint + np.array([0.1,0,0,0])
nu_vec_in = m_params.nu_vec.copy()

inFocus_nu_e_out, inFocus_nu_i_out, outFocus_nu_e, outFocus_nu_i = \
    fcn_effectiveMFT_fixedInDeg_clusNet.fcn_master_MFT_DynEqs(params, m_params)

print('rate out - rate in')
print(inFocus_nu_e_out[0] - nu_vec_in[0])

#%% check effective theory with root solver: selective

print('selective fixed point with effective root solver:')
m_params.nu_vec = selective_fixedPoint.copy()
nu_vec_in = m_params.nu_vec.copy()

inFocus_nu_e_out, inFocus_nu_i_out, outFocus_nu_e, outFocus_nu_i = \
    fcn_effectiveMFT_fixedInDeg_clusNet.fcn_master_MFT(params, m_params)

print('rate out - rate in')
print(inFocus_nu_e_out[0] - nu_vec_in[0])


print('selective fixed point with effective root solver, start away from fp:')
m_params.nu_vec = selective_fixedPoint + np.array([0.1,0,0,0])
nu_vec_in = m_params.nu_vec.copy()

inFocus_nu_e_out, inFocus_nu_i_out, outFocus_nu_e, outFocus_nu_i = \
    fcn_effectiveMFT_fixedInDeg_clusNet.fcn_master_MFT(params, m_params)

print('rate out - rate in')
print(inFocus_nu_e_out[0] - nu_vec_in[0])



#%% check effective theory with root solver: saddle

print('saddle fixed point with effective root solver:')
m_params.nu_vec = saddlePoint.copy()
nu_vec_in = m_params.nu_vec.copy()

inFocus_nu_e_out, inFocus_nu_i_out, outFocus_nu_e, outFocus_nu_i = \
    fcn_effectiveMFT_fixedInDeg_clusNet.fcn_master_MFT(params, m_params)

print('rate out - rate in')
print(inFocus_nu_e_out[0] - nu_vec_in[0])


print('saddle fixed point with effective root solver, start away from fp:')
m_params.nu_vec = saddlePoint + np.array([0.1,0,0,0])
nu_vec_in = m_params.nu_vec.copy()

inFocus_nu_e_out, inFocus_nu_i_out, outFocus_nu_e, outFocus_nu_i = \
    fcn_effectiveMFT_fixedInDeg_clusNet.fcn_master_MFT(params, m_params)

print('rate out - rate in')
print(inFocus_nu_e_out[0] - nu_vec_in[0])




#%% EFFECTIVE THEORY [WRITE FUNCTION FOR THIS]

# min and maximum rates to consider for in-focuse populations
rate_min = 0.
rate_max = 58.


# values of pop1 rates at which to compute effective force
pop1_rates_A = np.linspace(saddlePoint_inFocus_e[0], rate_min, 100)
pop1_rates_B = np.linspace(saddlePoint_inFocus_e[0], rate_max, 100)

pop1_rates_outA = np.zeros((len(pop1_rates_A)))
pop1_rates_outB = np.zeros((len(pop1_rates_B)))


outFocus_nu_e = np.zeros(2)
outFocus_nu_i = np.zeros(1)
outFocus_nu_e[0] = saddlePoint[1]
outFocus_nu_e[1] = saddlePoint[2]
outFocus_nu_i[0] = saddlePoint[3]

# start loop over nu1-nu2 grid
for ind_nu1 in range(0, len(pop1_rates_A)):
    
        
    # use new in-focus population rates, and use previous of out-of-focus rates for initial guesses         
    m_params.nu_vec = [pop1_rates_A[ind_nu1], outFocus_nu_e[0], outFocus_nu_e[1], outFocus_nu_i[0]]
    
    
    # solve for output in-focus population rate at this grid location
    inFocus_nu_e, inFocus_nu_i, outFocus_nu_e, outFocus_nu_i = \
        fcn_effectiveMFT_fixedInDeg_clusNet.fcn_master_MFT(params, m_params)


    # normalized force vector at this location
    pop1_rates_outA[ind_nu1] =  inFocus_nu_e[0]
    


outFocus_nu_e = np.zeros(2)
outFocus_nu_i = np.zeros(1)
outFocus_nu_e[0] = saddlePoint[1]
outFocus_nu_e[1] = saddlePoint[2]
outFocus_nu_i[0] = saddlePoint[3]

# start loop over nu1-nu2 grid
for ind_nu1 in range(0, len(pop1_rates_B)):
    
        
    # use new in-focus population rates, and use previous of out-of-focus rates for initial guesses         
    m_params.nu_vec = [pop1_rates_B[ind_nu1], outFocus_nu_e[0], outFocus_nu_e[1], outFocus_nu_i[0]]
    
    
    # solve for output in-focus population rate at this grid location
    inFocus_nu_e, inFocus_nu_i, outFocus_nu_e, outFocus_nu_i = \
        fcn_effectiveMFT_fixedInDeg_clusNet.fcn_master_MFT(params, m_params)


    # normalized force vector at this location
    pop1_rates_outB[ind_nu1] =  inFocus_nu_e[0]
    
    

pop1_rates_in = np.append(np.flip(pop1_rates_A), pop1_rates_B)
pop1_rates_out = np.append(np.flip(pop1_rates_outA), pop1_rates_outB)



#%%COMPUTE POTENTIAL ENERGY AS A FUNCTION OF DISTANCE AWAY FROM SADDLE POINT


F_of_r = pop1_rates_out - pop1_rates_in

U_of_r = np.zeros(len(F_of_r))



U_of_r = U_of_r + np.abs(np.min(U_of_r))


# transfer function
transfer_of_r = F_project_dr + path_distance



#%% PLOT FORCE VS POSITION

plt.figure()
plt.plot(path_distance, F_of_r, '-o', color='k')
plt.ylabel('F(r)')
plt.xlabel('position r')
plt.tight_layout()
plt.savefig(fig_outpath + fName_begin + '_force_vs_path_%s.pdf' % (sweep_param_str_val))


        
#%% PLOT POTENTIAL ENERGY VS POSITION

plt.figure()
plt.plot(path_distance, U_of_r, '-o', color='k')
plt.ylabel('potential U(x)')
plt.xlabel('position x')
plt.tight_layout()
plt.savefig(fig_outpath + fName_begin + '_potentialEnergy_vs_path_%s.pdf' % (sweep_param_str_val))

plt.close('all')



#%% PLOT TRANSFER FUNCTION VS POSITION

plt.figure()
plt.plot(path_distance, transfer_of_r, '-', color='r')
plt.plot([-40,40], [-40,40], '-', color='k', linewidth=1)
plt.ylabel('transfer function')
plt.xlabel('position r')
plt.tight_layout()
plt.savefig(fig_outpath + fName_begin + '_transferFunction_vs_path_%s.pdf' % (sweep_param_str_val))

plt.close('all')


#%% SAVE DATA


settings_dictionary = {'fName_begin':               fName_begin, \
                       'sweep_param_name':          sweep_param_name, \
                       'swept_params_dict':          swept_params_dict, \
                       'func_path1':                func_path1, \
                       'func_path2':                func_path2, \
                       'func_path3':                func_path3, \
                       'fig_outpath':               fig_outpath, \
                       'data_outpath':              data_outpath, \
                       'simParams_path':           simParams_path, 
                       'arousalParams_path':        arousalParams_path, \
                       'arousalParams_fname':      arousalParams_fname, \
                       'n_sweepParams':             n_sweepParams}        
    
    
results_dictionary = {'sim_params':                     params, \
                      'mft_params':                     m_params, \
                      'settings_dictionary':            settings_dictionary, \
                      'saddlePoint':                    saddlePoint, \
                      'selective_fixedPoint':           selective_fixedPoint, \
                      'rate_min':                       rate_min, \
                      'rate_max':                       rate_max, \
                      'd_nu':                           d_nu, \
                      'nBins':                          nBins, \
                      'nu1_vec':                        nu1_vec, \
                      'nu2_vec':                        nu2_vec, \
                      'F_pop1_vec':                     F_pop1_vec, \
                      'F_pop2_vec':                     F_pop2_vec, \
                      'nu1_grid':                       nu1_grid, \
                      'nu2_grid':                       nu2_grid, \
                      'F_pop1':                         F_pop1, \
                      'F_pop2':                         F_pop2, \
                      'F_pop1_norm_vec':                F_pop1_norm_vec, \
                      'F_pop2_norm_vec':                F_pop2_norm_vec, \
                      'force_mag':                      force_mag, \
                      'minEnergy_pathCoords_AtoSaddle': minEnergy_pathCoords_AtoSaddle, \
                      'minEnergy_pathCoords_SaddletoB': minEnergy_pathCoords_SaddletoB, \
                      'path_distance':                  path_distance, \
                      'F_dot_dr':                         F_dot_dr, \
                      'F_project_dr':                         F_project_dr, \
                      'transfer_of_r':                         transfer_of_r, \
                      'U_of_r':                         U_of_r}



fName_end = ('_effectiveMFT_ALT_%s.mat' % (sweep_param_str_val))
save_filename = ( data_outpath +  fName_begin + fName_end)   
savemat(save_filename, results_dictionary)





