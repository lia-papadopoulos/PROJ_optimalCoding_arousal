
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

from mftParams_arousalSweep_noDisorder import mft_params

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
print(inFocus_nu_e_out[0] - nu_vec_in[0], inFocus_nu_e_out[1] - nu_vec_in[1])


print('selective fixed point with effective dynamical, start away from fp:')
m_params.nu_vec = selective_fixedPoint + np.array([0.1,0,0,0])
nu_vec_in = m_params.nu_vec.copy()

inFocus_nu_e_out, inFocus_nu_i_out, outFocus_nu_e, outFocus_nu_i = \
    fcn_effectiveMFT_fixedInDeg_clusNet.fcn_master_MFT_DynEqs(params, m_params)

print('rate out - rate in')
print(inFocus_nu_e_out[0] - nu_vec_in[0], inFocus_nu_e_out[1] - nu_vec_in[1])

#%% check effective theory with dynamical equations: saddle

print('saddle fixed point with effective dynamical:')
m_params.nu_vec = saddlePoint.copy()
nu_vec_in = m_params.nu_vec.copy()

inFocus_nu_e_out, inFocus_nu_i_out, outFocus_nu_e, outFocus_nu_i = \
    fcn_effectiveMFT_fixedInDeg_clusNet.fcn_master_MFT_DynEqs(params, m_params)

print('rate out - rate in')
print(inFocus_nu_e_out[0] - nu_vec_in[0], inFocus_nu_e_out[1] - nu_vec_in[1])


print('saddle fixed point with effective dynamical, start away:')
m_params.nu_vec = saddlePoint + np.array([0.1,0,0,0])
nu_vec_in = m_params.nu_vec.copy()

inFocus_nu_e_out, inFocus_nu_i_out, outFocus_nu_e, outFocus_nu_i = \
    fcn_effectiveMFT_fixedInDeg_clusNet.fcn_master_MFT_DynEqs(params, m_params)

print('rate out - rate in')
print(inFocus_nu_e_out[0] - nu_vec_in[0], inFocus_nu_e_out[1] - nu_vec_in[1])

#%% check effective theory with root solver: selective

print('selective fixed point with effective root solver:')
m_params.nu_vec = selective_fixedPoint.copy()
nu_vec_in = m_params.nu_vec.copy()

inFocus_nu_e_out, inFocus_nu_i_out, outFocus_nu_e, outFocus_nu_i = \
    fcn_effectiveMFT_fixedInDeg_clusNet.fcn_master_MFT(params, m_params)

print('rate out - rate in')
print(inFocus_nu_e_out[0] - nu_vec_in[0], inFocus_nu_e_out[1] - nu_vec_in[1])


print('selective fixed point with effective root solver, start away from fp:')
m_params.nu_vec = selective_fixedPoint + np.array([0.1,0,0,0])
nu_vec_in = m_params.nu_vec.copy()

inFocus_nu_e_out, inFocus_nu_i_out, outFocus_nu_e, outFocus_nu_i = \
    fcn_effectiveMFT_fixedInDeg_clusNet.fcn_master_MFT(params, m_params)

print('rate out - rate in')
print(inFocus_nu_e_out[0] - nu_vec_in[0], inFocus_nu_e_out[1] - nu_vec_in[1])



#%% check effective theory with root solver: saddle

print('saddle fixed point with effective root solver:')
m_params.nu_vec = saddlePoint.copy()
nu_vec_in = m_params.nu_vec.copy()

inFocus_nu_e_out, inFocus_nu_i_out, outFocus_nu_e, outFocus_nu_i = \
    fcn_effectiveMFT_fixedInDeg_clusNet.fcn_master_MFT(params, m_params)

print('rate out - rate in')
print(inFocus_nu_e_out[0] - nu_vec_in[0], inFocus_nu_e_out[1] - nu_vec_in[1])


print('saddle fixed point with effective root solver, start away from fp:')
m_params.nu_vec = saddlePoint + np.array([0.1,0,0,0])
nu_vec_in = m_params.nu_vec.copy()

inFocus_nu_e_out, inFocus_nu_i_out, outFocus_nu_e, outFocus_nu_i = \
    fcn_effectiveMFT_fixedInDeg_clusNet.fcn_master_MFT(params, m_params)

print('rate out - rate in')
print(inFocus_nu_e_out[0] - nu_vec_in[0], inFocus_nu_e_out[1] - nu_vec_in[1])



#%% EFFECTIVE THEORY [WRITE FUNCTION FOR THIS]

# min and maximum rates to consider for in-focuse populations
rate_min = 0.
#rate_max = np.round(np.max(selective_fixedPoint_inFocus_e) + np.min(selective_fixedPoint_inFocus_e))
rate_max = 58.

# step sizes for in-focus population rate locations
d_nu = 0
nBins = 50
while np.mod(d_nu, 2) == 0:
    nBins = nBins+1
    d_nu = np.round((rate_max - rate_min)/nBins,2)
    
print(nBins)

# values of pop1 and pop2 rates at which to compute effective force
pop1_rates = np.append(np.flip(np.arange(saddlePoint_inFocus_e[0] - d_nu, rate_min, -d_nu)), np.arange(saddlePoint_inFocus_e[0], rate_max + d_nu, d_nu))
pop2_rates = np.append(np.flip(np.arange(saddlePoint_inFocus_e[0] - d_nu, rate_min, -d_nu)), np.arange(saddlePoint_inFocus_e[0], rate_max + d_nu, d_nu))

# rate grids
nu1_grid, nu2_grid = np.meshgrid(pop1_rates, pop2_rates)

# intiialize grid of force components in both directions
F_pop1 = np.zeros((len(pop1_rates), len(pop2_rates)))
F_pop2 = np.zeros((len(pop1_rates), len(pop2_rates)))

# initialize vectors of pop1 and pop2 rates, force field components
nu1_vec = np.array([])
nu2_vec = np.array([])
F_pop1_vec = np.array([])
F_pop2_vec = np.array([])
F_pop1_norm_vec = np.array([])
F_pop2_norm_vec = np.array([])

# set initial values of nu1, nu2
# (bottom right corner)
nu1_ind = len(pop1_rates)-1
nu2_ind = 0
nu1 = nu1_grid[nu2_ind , nu1_ind]
nu2 = nu2_grid[nu2_ind , nu1_ind]

# initial rate vector [use known fixed point values for out-of-focus]
m_params.nu_vec = np.zeros(4)
m_params.nu_vec[0] = pop1_rates[nu1_ind]
m_params.nu_vec[1] = pop2_rates[nu2_ind]
m_params.nu_vec[2] = selective_fixedPoint[2]
m_params.nu_vec[3] = selective_fixedPoint[3]

# create a copy
nu_vec_in = m_params.nu_vec.copy()

# how to scan population 1 values (move left or right)
pop1Rule = 'left'


# start loop over nu1-nu2 grid
count = 0
while (count < np.size(nu1_grid)+10):
    
    print(count/np.size(nu1_grid))
        
    # counter variabile
    count = count + 1
    
    # solve for output in-focus population rates at this grid location
    inFocus_nu_e, inFocus_nu_i, outFocus_nu_e, outFocus_nu_i = \
        fcn_effectiveMFT_fixedInDeg_clusNet.fcn_master_MFT(params, m_params)
                
    
    # vector of sampled nu1 - nu2 locations
    nu1_vec = np.append(nu1_vec, nu1)
    nu2_vec = np.append(nu2_vec, nu2)
    

    # force vector at this location
    Fvec = np.array([inFocus_nu_e[0] - nu_vec_in[0], inFocus_nu_e[1] - nu_vec_in[1]])
    
    # normalized force vector at this location
    Fvec_norm = Fvec/np.linalg.norm(Fvec)    
    
    # update Ffield grids with value of force at this location (difference between input and output in-focus rates)
    F_pop1[nu2_ind, nu1_ind] = Fvec[0]
    F_pop2[nu2_ind, nu1_ind] = Fvec[1]
    
    # update force vectors with value of force at this location
    F_pop1_vec = np.append(F_pop1_vec, Fvec[0])
    F_pop2_vec = np.append(F_pop2_vec, Fvec[1])
    
    F_pop1_norm_vec = np.append(F_pop1_norm_vec, Fvec_norm[0])
    F_pop2_norm_vec = np.append(F_pop2_norm_vec, Fvec_norm[1])
    
        
    # update nu1 grid location    
    if pop1Rule == 'left':
        
        nu1_ind = nu1_ind - 1
        
    if pop1Rule == 'right':
        
        nu1_ind = nu1_ind + 1
                    
        
    # if nu1 goes below minimum value, start moving in the other direction and increase nu2
    if  nu1_ind < 0:
                
        pop1Rule = 'right'
        
        nu1_ind = 0
        
        nu2_ind = nu2_ind + 1
        
        
    # if nu1 goes above maximum value, start moving in the other direction and increase nu2            
    if nu1_ind == len(pop1_rates):
        
        pop1Rule = 'left'
        
        nu1_ind = len(pop1_rates)-1
        
        nu2_ind = nu2_ind + 1
        
        
    # if second population goes above max rate, quit
    if nu2_ind >= len(pop2_rates):
        print('reached end of grid')
        break
    
    # set nu_1, nu_2    
    nu1 = nu1_grid[nu2_ind , nu1_ind]
    nu2 = nu2_grid[nu2_ind , nu1_ind]
        
        
    # update initial rate vector    
    # use new in-focus population rates, and use previous of out-of-focus rates for initial guesses         
    m_params.nu_vec = [nu1, nu2, outFocus_nu_e[0], outFocus_nu_i[0]]
    
    
    # create a copy of the starting vector
    nu_vec_in = m_params.nu_vec.copy()
         

#%% COMPUTE THE POTENTIAL

# FIND INTEGRATION PATH BY LOOPING OVER ROWS OF GRID AND FINDING GRID POINTS AT MINIMUM FORCE

# location of grid point for saddle point
saddlePoint_ind_nu1 = np.argmin(np.abs(nu1_grid[0,:] - saddlePoint_inFocus_e[0]))
saddlePoint_ind_nu2 = np.argmin(np.abs(nu2_grid[:,0] - saddlePoint_inFocus_e[1]))

# initialize vector of points for denoting path of minimum energy
minEnergy_pathCoords_AtoSaddle = np.zeros((saddlePoint_ind_nu2+1, 2))
minEnergy_pathInds_AtoSaddle = np.zeros((saddlePoint_ind_nu2+1, 2))


# increase value of nu2 from bottom row (by lower right fixed point) to saddle point
# find path of least resistance at each point
for rowInd in range(0, saddlePoint_ind_nu2+1):
        
    # energy values of this row
    rowEnergies = np.sqrt(F_pop1**2 + F_pop2**2)[rowInd, :]
        
    # location of minimum energy
    minEnergy_ind = np.argmin(rowEnergies)
        
    minEnergy_pathInds_AtoSaddle[rowInd, 0] = minEnergy_ind
    minEnergy_pathInds_AtoSaddle[rowInd, 1] = rowInd                
    
    minEnergy_pathCoords_AtoSaddle[rowInd, 0] = pop1_rates[minEnergy_ind]
    minEnergy_pathCoords_AtoSaddle[rowInd, 1] = pop2_rates[rowInd]
    
    
    '''
    if rowInd == 0:
        rowEnergies = np.sqrt(F_pop1**2 + F_pop2**2)[rowInd, :]
        minEnergy_ind = np.argmin(rowEnergies)
    
    else:
        
        x1 =  minEnergy_pathCoords_AtoSaddle[rowInd, 0]
        y1 =  pop2_rates[rowInd]
        
        x2 = pop1_rates
        y2 = pop2_rates[rowInd+1]
        
        f_project_r = np.zeros(len(x2))
        for columnInd in range(0, len(x2)):
            dr = np.array([x2[i]- x1, y2-y1])
            fx = F_pop1[rowInd, columnInd]
            fy = F_pop2[rowInd, columnInd] 
            f = np.array([fx,fy])
            f_project_r[columnInd] = np.dot(f, dr/np.linalg.norm(dr))
        minEnergy_ind = np.argmin(f_project_r)
        
    minEnergy_pathInds_AtoSaddle[rowInd, 0] = minEnergy_ind
    minEnergy_pathInds_AtoSaddle[rowInd, 1] = rowInd                
    
    minEnergy_pathCoords_AtoSaddle[rowInd, 0] = pop1_rates[minEnergy_ind]
    minEnergy_pathCoords_AtoSaddle[rowInd, 1] = pop2_rates[rowInd]
    '''


#YOU ARE HERE.


'''
# mirror the path for going from saddle point to fixed_pointB
minEnergy_pathInds_SaddletoB = np.flip(minEnergy_pathInds_AtoSaddle, axis=1)
minEnergy_pathInds_SaddletoB = np.flip(minEnergy_pathInds_SaddletoB, axis=0)

minEnergy_pathCoords_SaddletoB = np.flip(minEnergy_pathCoords_AtoSaddle, axis=1)
minEnergy_pathCoords_SaddletoB = np.flip(minEnergy_pathCoords_SaddletoB, axis=0)
'''

# increase value of nu2 from bottom row (by lower right fixed point) to saddle point
# find path of least resistance at each point

# initialize vector of points for denoting path of minimum energy
minEnergy_pathCoords_SaddletoB = np.zeros((saddlePoint_ind_nu2+1, 2))
minEnergy_pathInds_SaddletoB = np.zeros((saddlePoint_ind_nu2+1, 2))
count = 0
for colInd in range(saddlePoint_ind_nu1, -1,-1):
        
    # energy values of this row
    colEnergies = np.sqrt(F_pop1**2 + F_pop2**2)[:, colInd]
    
    # location of minimum energy
    minEnergy_ind = np.argmin(colEnergies)
        
    minEnergy_pathInds_SaddletoB[count,1] = minEnergy_ind
    minEnergy_pathInds_SaddletoB[count, 0] = colInd                
    
    minEnergy_pathCoords_SaddletoB[count, 1] = pop2_rates[minEnergy_ind]
    minEnergy_pathCoords_SaddletoB[count, 0] = pop1_rates[colInd]
    count +=1

# GET FORCE VECTOR ALONG PATH
# go from A --> Saddle, then B --> Saddle 

force_minEnergy_path_AtoSaddle = np.zeros((np.shape(minEnergy_pathInds_AtoSaddle)[0], 2))
force_minEnergy_path_SaddletoB = np.zeros((np.shape(minEnergy_pathInds_AtoSaddle)[0], 2))

phiIn_minEnergy_path_AtoSaddle = np.zeros((np.shape(minEnergy_pathInds_AtoSaddle)[0], 2))
phiIn_minEnergy_path_SaddletoB = np.zeros((np.shape(minEnergy_pathInds_AtoSaddle)[0], 2))


# A to saddle
for pathInd in range(0, np.shape(minEnergy_pathInds_AtoSaddle)[0]):
    
    xInd = minEnergy_pathInds_AtoSaddle[pathInd, 0].astype(int)
    yInd = minEnergy_pathInds_AtoSaddle[pathInd, 1].astype(int)
    
    force_minEnergy_path_AtoSaddle[pathInd, 0] = F_pop1[yInd, xInd]
    force_minEnergy_path_AtoSaddle[pathInd, 1] = F_pop2[yInd, xInd]    

    phiIn_minEnergy_path_AtoSaddle[pathInd, 0] = nu1_grid[yInd, xInd]
    phiIn_minEnergy_path_AtoSaddle[pathInd, 1] = nu2_grid[yInd, xInd]

    
# Saddle to B
for pathInd in range(0, np.shape(minEnergy_pathInds_SaddletoB)[0]):
    
    xInd = minEnergy_pathInds_SaddletoB[pathInd, 0].astype(int)
    yInd = minEnergy_pathInds_SaddletoB[pathInd, 1].astype(int)
    
    force_minEnergy_path_SaddletoB[pathInd, 0] = F_pop1[yInd, xInd]
    force_minEnergy_path_SaddletoB[pathInd, 1] = F_pop2[yInd, xInd]    
    
    phiIn_minEnergy_path_SaddletoB[pathInd, 0] = nu1_grid[yInd, xInd]
    phiIn_minEnergy_path_SaddletoB[pathInd, 1] = nu2_grid[yInd, xInd]  


# COMPUTE LINE INTEGRAL OF F*dr
# FOR  (A --> Saddle) AND (Saddle --> B)

F_project_dr_AtoSaddle = np.zeros(np.shape(minEnergy_pathInds_AtoSaddle)[0]-1)
F_project_dr_SaddletoB = np.zeros(np.shape(minEnergy_pathInds_AtoSaddle)[0]-1)


F_dot_dr_AtoSaddle = np.zeros(np.shape(minEnergy_pathInds_AtoSaddle)[0]-1)
F_dot_dr_SaddletoB = np.zeros(np.shape(minEnergy_pathInds_SaddletoB)[0]-1)

phiIn_dot_dr_AtoSaddle = np.zeros(np.shape(minEnergy_pathInds_AtoSaddle)[0]-1)
phiIn_dot_dr_SaddletoB = np.zeros(np.shape(minEnergy_pathInds_SaddletoB)[0]-1)

dr_magnitude_AtoSaddle = np.zeros(np.shape(minEnergy_pathInds_AtoSaddle)[0]-1)
dr_magnitude_SaddletoB = np.zeros(np.shape(minEnergy_pathInds_SaddletoB)[0]-1)

# step along path from A to Saddle and store F*dr at each point
count = 0
#for rowInd in range(len(dr_magnitude_SaddletoA), 0, -1):
for rowInd in range(0, len(dr_magnitude_AtoSaddle)):

    # dr vector along step
    dr = minEnergy_pathCoords_AtoSaddle[rowInd + 1, :] - minEnergy_pathCoords_AtoSaddle[rowInd, :] 
       
    # F vector along step
    Fstep = force_minEnergy_path_AtoSaddle[rowInd,:]

    # F project dr
    F_project_dr_AtoSaddle[count] = np.dot(Fstep, dr/np.linalg.norm(dr))


    # F dot dr
    F_dot_dr_AtoSaddle[count] = np.dot(Fstep, dr)

    # phi vector along step
    phiIn_step = phiIn_minEnergy_path_AtoSaddle[rowInd,:]
    phiIn_dot_dr_AtoSaddle[count] = np.dot(phiIn_step, dr)


    # dr magnitude
    dr_magnitude_AtoSaddle[count] = np.sqrt(np.sum(dr**2))
    
    # update step count
    count+=1
        
   
# step along path from B to Saddle and store F*dr at each point
count = 0
for rowInd in range(0, len(dr_magnitude_SaddletoB), 1):
    
    # dr vector along step
    dr = minEnergy_pathCoords_SaddletoB[rowInd+1, :] - minEnergy_pathCoords_SaddletoB[rowInd, :] 
    
    # F vector along step
    Fstep = force_minEnergy_path_SaddletoB[rowInd,:]

    # F project dr
    F_project_dr_SaddletoB[count] = np.dot(Fstep, dr/np.linalg.norm(dr))

    # F dot dr
    F_dot_dr_SaddletoB[count] = np.dot(Fstep, dr)

    phiIn_step = phiIn_minEnergy_path_SaddletoB[rowInd,:]
    phiIn_dot_dr_SaddletoB[count] = np.dot(phiIn_step, dr)
    
    # dr magnitude
    dr_magnitude_SaddletoB[count] = np.sqrt(np.sum(dr**2))
    
    # update step count
    count+=1

#%%COMPUTE POTENTIAL ENERGY AS A FUNCTION OF DISTANCE AWAY FROM SADDLE POINT


# total force project dr
F_project_dr = np.append(F_project_dr_AtoSaddle,F_project_dr_SaddletoB)

# total force dot dr
F_dot_dr = np.append(F_dot_dr_AtoSaddle,F_dot_dr_SaddletoB)

# total potential
U_of_r = np.cumsum(-F_dot_dr)
U_of_r = U_of_r + np.abs(np.min(U_of_r))

# path distance
path_distance_AtoSaddle = np.cumsum(np.append(0,dr_magnitude_AtoSaddle[:-1]))
path_distance_SaddletoB = np.cumsum(np.append(0,dr_magnitude_SaddletoB[:-1])) + path_distance_AtoSaddle[-1]
path_distance = np.append(path_distance_AtoSaddle,path_distance_SaddletoB)


path_distance_AtoSaddle = np.cumsum(np.append(0,dr_magnitude_AtoSaddle))
path_distance_AtoSaddle -= path_distance_AtoSaddle[-1]
path_distance_AtoSaddle = path_distance_AtoSaddle[:-1]
path_distance_SaddletoB = np.cumsum(np.append(0,dr_magnitude_SaddletoB[:-1]))
path_distance = np.append(path_distance_AtoSaddle,path_distance_SaddletoB)

# transfer function
transfer_of_r = F_project_dr + path_distance


#%% BEGIN PLOTTING


#%% PLOT VECTOR FIELDS


# plot the results of the vector field using grid
plt.figure()
plt.quiver(nu1_grid[::1,::1], nu2_grid[::1,::1], F_pop1[::1,::1], F_pop2[::1,::1])
plt.plot(selective_fixedPoint_inFocus_e[0], selective_fixedPoint_inFocus_e[1], '*', markersize=10) 
plt.plot(selective_fixedPoint_inFocus_e[1], selective_fixedPoint_inFocus_e[0], '*', markersize=10) 
plt.plot(saddlePoint_inFocus_e[0], saddlePoint_inFocus_e[1], '*', markersize=10) 
plt.xlabel('nu 1')  
plt.ylabel('nu 2') 
plt.tight_layout()
plt.savefig(fig_outpath + fName_begin + '_forceField_%s.pdf' % (sweep_param_str_val))


# plot the results of the normalized vector field
plt.figure()
plt.quiver(nu1_vec, nu2_vec, F_pop1_norm_vec, F_pop2_norm_vec)
plt.plot(selective_fixedPoint_inFocus_e[0], selective_fixedPoint_inFocus_e[1], '*', markersize=10) 
plt.plot(selective_fixedPoint_inFocus_e[1], selective_fixedPoint_inFocus_e[0], '*', markersize=10) 
plt.plot(saddlePoint_inFocus_e[0], saddlePoint_inFocus_e[1], '*', markersize=10) 
plt.xlabel('nu 1')  
plt.ylabel('nu 2') 
plt.tight_layout()
plt.savefig(fig_outpath + fName_begin + '_forceField_normalized_%s.pdf' % (sweep_param_str_val))



# plot a heat map of the force magnitude
force_mag = np.sqrt(F_pop1**2 + F_pop2**2)
force_mag[saddlePoint_ind_nu2, saddlePoint_ind_nu1] = np.nan

plt.figure()
plt.pcolor(nu1_grid, nu2_grid, np.log(force_mag), shading='auto')        
plt.colorbar()
plt.title('log(|force|)')
plt.xlabel('nu 1')
plt.ylabel('nu 2')
plt.tight_layout()
plt.savefig(fig_outpath + fName_begin + '_forceField_log_magnitude_%s.pdf' % (sweep_param_str_val))



#%% PLOT MINIMUM ENERGY PATH ON HEATMAP OF FORCE MAGNITUDE

force_mag = np.sqrt(F_pop1**2 + F_pop2**2)
force_mag[saddlePoint_ind_nu2, saddlePoint_ind_nu1] = np.nan

plt.figure()
plt.pcolor(nu1_grid, nu2_grid, np.log(force_mag), shading='auto')

for i in range(0,np.shape(minEnergy_pathCoords_AtoSaddle)[0]-1):
    
    plt.plot([minEnergy_pathCoords_AtoSaddle[i,0],minEnergy_pathCoords_AtoSaddle[i+1,0]],\
             [minEnergy_pathCoords_AtoSaddle[i,1],minEnergy_pathCoords_AtoSaddle[i+1,1]],\
             '-o',color='r',markersize=2)
        
    plt.plot([minEnergy_pathCoords_SaddletoB[i,0],minEnergy_pathCoords_SaddletoB[i+1,0]],\
             [minEnergy_pathCoords_SaddletoB[i,1],minEnergy_pathCoords_SaddletoB[i+1,1]],\
             '-o',color='r',markersize=2)
        
plt.colorbar()
plt.title('log(|force|)')
plt.xlabel('nu 1')
plt.ylabel('nu 2')
plt.tight_layout()
plt.savefig(fig_outpath + fName_begin + '_forceField_log_magnitude_integrationPath_%s.pdf' % (sweep_param_str_val))



#%% PLOT MINIMUM ENERGY PATH ON FORCE FIELD

plt.figure()
plt.quiver(nu1_grid, nu2_grid, F_pop1, F_pop2)

for i in range(0,np.shape(minEnergy_pathCoords_AtoSaddle)[0]-1):
    
    plt.plot([minEnergy_pathCoords_AtoSaddle[i,0],minEnergy_pathCoords_AtoSaddle[i+1,0]],\
             [minEnergy_pathCoords_AtoSaddle[i,1],minEnergy_pathCoords_AtoSaddle[i+1,1]],\
             '-o',color='r',markersize=2)
        
    plt.plot([minEnergy_pathCoords_SaddletoB[i,0],minEnergy_pathCoords_SaddletoB[i+1,0]],\
             [minEnergy_pathCoords_SaddletoB[i,1],minEnergy_pathCoords_SaddletoB[i+1,1]],\
             '-o',color='r',markersize=2)
        
plt.title('force field')
plt.xlabel('nu 1')
plt.ylabel('nu 2')
plt.tight_layout()
plt.savefig(fig_outpath + fName_begin + '_forceField_integrationPath_%s.pdf' % (sweep_param_str_val))



#%% PLOT FORCE VS POSITION

plt.figure()
plt.plot(path_distance, F_dot_dr, '-o', color='k')
plt.ylabel('F * dr')
plt.xlabel('position r')
plt.tight_layout()
plt.savefig(fig_outpath + fName_begin + '_force_dot_dr_vs_path_%s.pdf' % (sweep_param_str_val))

plt.figure()
plt.plot(path_distance, F_project_dr, '-o', color='k')
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




