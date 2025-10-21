
"""
EFFECTIVE MFT FOR NETWORKS WITH 2 E CLUSTERS
"""

#%% imports

# import settings
import effectiveMFT_spatialVariance_simplified_settings as settings

# basic imports
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
from scipy.io import savemat
import argparse

#%% unpack settings

maxCores = settings.maxCores
cores_per_job = settings.cores_per_job 

func_path1 = settings.func_path1
func_path2 = settings.func_path2
func_path3 = settings.func_path3
func_path4 = settings.func_path4
simParams_path = settings.simParams_path
fig_outpath = settings.fig_outpath
data_outpath = settings.data_outpath
fName_begin = settings.fName_begin

sweep_param_name = settings.sweep_param_name
param_vals = settings.param_vals


#%% load my own functions

from mftParams_2cluster_sweep_sd_nu_ext_e_effectiveMFT import mft_params
 
sys.path.insert(0,func_path1)     
from fcn_make_network_2cluster import fcn_make_network_cluster

sys.path.insert(0,func_path2) 
import fcn_MFT_spatialVariance_simplified

sys.path.insert(0,func_path3) 
import fcn_effectiveMFT_spatialVariance_simplified

sys.path.append(simParams_path)
from simParams_2cluster_sweep_sd_nu_ext_e import sim_params


#%% initialize simulation and analysis parameters classes


params = sim_params()
m_params = mft_params()


#%% argparser

parser = argparse.ArgumentParser() 


# swept parameter name + value as string
parser.add_argument('-sweep_param_str_val', '--sweep_param_str_val', \
                    type=str, required=True)
    

# possible swept parameters used in launch jobs
parser.add_argument('-sd_nu_ext_e_pert', '--sd_nu_ext_e_pert', \
                     type=float, default = params.sd_nu_ext_e_pert)
    
# arguments of parser
args = parser.parse_args()


#-------------------- argparser values for later use -------------------------#

# name of swept parameter with value as a string
sweep_param_str_val = args.sweep_param_str_val


#%% SETUP FOR MFT

params.temporal_extNoise = params.extCurrent_poisson
params.sd_nu_ext_e_pert = args.sd_nu_ext_e_pert
params.update_JplusAB()
params.set_dependent_vars()

W, popsizeE, popsizeI = fcn_make_network_cluster(params)     
params.popsizeE = popsizeE
params.popsizeI = popsizeI

seed = np.random.choice(64654321)
params.set_external_inputs(seed) 

print('setup done')


#%% MFT

# non-selective fixed pointx
m_params.nu_bar_vec = m_params.nu_bar_vec_nonselective.copy()
sol, nu_bar_e, nu_bar_i = fcn_MFT_spatialVariance_simplified.fcn_master_MFT(params, m_params)

saddlePoint = np.append(nu_bar_e, nu_bar_i)
saddlePoint_inFocus_e = nu_bar_e[m_params.inFocus_pops_e]

print(nu_bar_e, nu_bar_i)


#%%

# selective fixed point
m_params.nu_bar_vec = m_params.nu_bar_vec_selective.copy()
sol, nu_bar_e, nu_bar_i = fcn_MFT_spatialVariance_simplified.fcn_master_MFT(params, m_params)


selective_fixedPoint = np.append(nu_bar_e, nu_bar_i)
selective_fixedPoint_inFocus_e = nu_bar_e[m_params.inFocus_pops_e]

print(nu_bar_e, nu_bar_i)


#%% verify the answer for the selective fixed point is the same with the dynamical equations

m_params.nu_bar_vec = selective_fixedPoint + np.array([0.1,0,0,0])
nu_bar_e_dyn, nu_bar_i_dyn  = fcn_MFT_spatialVariance_simplified.fcn_master_MFT_DynEqs(params, m_params)
print(nu_bar_e_dyn, nu_bar_i_dyn)


#%%


#%% check effective theory with dynamical equations

m_params.nu_bar_vec = saddlePoint.copy()
nu_bar_vec_in = m_params.nu_bar_vec.copy()


inFocus_nu_bar_e_out, inFocus_nu_bar_i_out, outFocus_nu_bar_e, outFocus_nu_bar_i = \
    fcn_effectiveMFT_spatialVariance_simplified.fcn_master_MFT_DynEqs(params, m_params)

print('rate out - rate in')
print(inFocus_nu_bar_e_out[0] - nu_bar_vec_in[0], inFocus_nu_bar_e_out[1] - nu_bar_vec_in[1])


m_params.nu_bar_vec = saddlePoint + np.array([0.1,0,0,0])
nu_bar_vec_in = m_params.nu_bar_vec.copy()

inFocus_nu_bar_e2, inFocus_nu_bar_i2, outFocus_nu_bar_e2, outFocus_nu_bar_i2 = \
    fcn_effectiveMFT_spatialVariance_simplified.fcn_master_MFT_DynEqs(params, m_params)

print('rate out - rate in')
print(inFocus_nu_bar_e2[0] - nu_bar_vec_in[0], inFocus_nu_bar_e2[1] - nu_bar_vec_in[1])




#%% check effective theory with root solver

m_params.nu_bar_vec = saddlePoint.copy()
nu_bar_vec_in = m_params.nu_bar_vec.copy()

inFocus_nu_bar_e_out, inFocus_nu_bar_i_out, outFocus_nu_bar_e, outFocus_nu_bar_i = \
    fcn_effectiveMFT_spatialVariance_simplified.fcn_master_MFT(params, m_params)
   
print('rate out - rate in')
print(inFocus_nu_bar_e_out[0] - nu_bar_vec_in[0], inFocus_nu_bar_e_out[1] - nu_bar_vec_in[1])


m_params.nu_bar_vec = saddlePoint + np.array([0.1,0,0,0])
nu_bar_vec_in = m_params.nu_bar_vec.copy()

inFocus_nu_bar_e2, inFocus_nu_bar_i2, outFocus_nu_bar_e2, outFocus_nu_bar_i2 = \
    fcn_effectiveMFT_spatialVariance_simplified.fcn_master_MFT(params, m_params)

print('rate out - rate in')
print(inFocus_nu_bar_e2[0] - nu_bar_vec_in[0], inFocus_nu_bar_e2[1] - nu_bar_vec_in[1])



#%% EFFECTIVE THEORY [WRITE FUNCTION FOR THIS]

# min and maximum rates to consider for in-focuse populations
rate_min = 0.
rate_max = np.round(np.max(selective_fixedPoint_inFocus_e) + np.min(selective_fixedPoint_inFocus_e))

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
m_params.nu_bar_vec = np.zeros(4)
m_params.nu_bar_vec[0] = pop1_rates[nu1_ind]
m_params.nu_bar_vec[1] = pop2_rates[nu2_ind]
m_params.nu_bar_vec[2] = selective_fixedPoint[2]
m_params.nu_bar_vec[3] = selective_fixedPoint[3]

# create a copy
nu_bar_vec_in = m_params.nu_bar_vec.copy()

# how to scan population 1 values (move left or right)
pop1Rule = 'left'


# start loop over nu1-nu2 grid
count = 0
while (count < np.size(nu1_grid)+10):
    
    print(count/np.size(nu1_grid))
        
    # counter variabile
    count = count + 1
    
    # solve for output in-focus population rates at this grid location
    inFocus_nu_bar_e, inFocus_nu_bar_i, outFocus_nu_bar_e, outFocus_nu_bar_i = \
        fcn_effectiveMFT_spatialVariance_simplified.fcn_master_MFT(params, m_params)
                
    
    # vector of sampled nu1 - nu2 locations
    nu1_vec = np.append(nu1_vec, nu1)
    nu2_vec = np.append(nu2_vec, nu2)
    

    # force vector at this location
    Fvec = np.array([inFocus_nu_bar_e[0] - nu_bar_vec_in[0], inFocus_nu_bar_e[1] - nu_bar_vec_in[1]])
    
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
    m_params.nu_bar_vec = [nu1, nu2, outFocus_nu_bar_e[0], outFocus_nu_bar_i[0]]
    
    
    # create a copy of the starting vector
    nu_bar_vec_in = m_params.nu_bar_vec.copy()
         

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
    

# mirror the path for going from saddle point to fixed_pointB
minEnergy_pathInds_SaddletoB = np.flip(minEnergy_pathInds_AtoSaddle, axis=1)
minEnergy_pathInds_SaddletoB = np.flip(minEnergy_pathInds_SaddletoB, axis=0)

minEnergy_pathCoords_SaddletoB = np.flip(minEnergy_pathCoords_AtoSaddle, axis=1)
minEnergy_pathCoords_SaddletoB = np.flip(minEnergy_pathCoords_SaddletoB, axis=0)


# GET FORCE VECTOR ALONG PATH
# go from A --> Saddle, then B --> Saddle 

force_minEnergy_path_AtoSaddle = np.zeros((np.shape(minEnergy_pathInds_AtoSaddle)[0], 2))
force_minEnergy_path_SaddletoB = np.zeros((np.shape(minEnergy_pathInds_AtoSaddle)[0], 2))

# A to saddle
for pathInd in range(0, np.shape(minEnergy_pathInds_AtoSaddle)[0]):
    
    xInd = minEnergy_pathInds_AtoSaddle[pathInd, 0].astype(int)
    yInd = minEnergy_pathInds_AtoSaddle[pathInd, 1].astype(int)
    
    force_minEnergy_path_AtoSaddle[pathInd, 0] = F_pop1[yInd, xInd]
    force_minEnergy_path_AtoSaddle[pathInd, 1] = F_pop2[yInd, xInd]    
    
    
# Saddle to B
for pathInd in range(0, np.shape(minEnergy_pathInds_SaddletoB)[0]):
    
    xInd = minEnergy_pathInds_SaddletoB[pathInd, 0].astype(int)
    yInd = minEnergy_pathInds_SaddletoB[pathInd, 1].astype(int)
    
    force_minEnergy_path_SaddletoB[pathInd, 0] = F_pop1[yInd, xInd]
    force_minEnergy_path_SaddletoB[pathInd, 1] = F_pop2[yInd, xInd]    
    



# COMPUTE LINE INTEGRAL OF F*dr
# FOR  (Saddle --> A) AND (Saddle --> B)

F_dot_dr_SaddletoA = np.zeros(np.shape(minEnergy_pathInds_AtoSaddle)[0]-1)
F_dot_dr_SaddletoB = np.zeros(np.shape(minEnergy_pathInds_SaddletoB)[0]-1)

dr_magnitude_SaddletoA = np.zeros(np.shape(minEnergy_pathInds_AtoSaddle)[0]-1)
dr_magnitude_SaddletoB = np.zeros(np.shape(minEnergy_pathInds_SaddletoB)[0]-1)

# step along path from Saddle to A and store F*dr at each point
count = 0
for rowInd in range(len(dr_magnitude_SaddletoA), 0, -1):

    # dr vector along step
    dr = minEnergy_pathCoords_AtoSaddle[rowInd - 1, :] - minEnergy_pathCoords_AtoSaddle[rowInd, :] 
        
    # F vector along step
    Fstep = force_minEnergy_path_AtoSaddle[rowInd,:]

    # F dot dr
    F_dot_dr_SaddletoA[count] = np.dot(Fstep, dr)
    
    # dr magnitude
    dr_magnitude_SaddletoA[count] = np.sqrt(np.sum(dr**2))
    
    # update step count
    count+=1
        
   
# step along path from Saddle toB and store F*dr at each point
for rowInd in range(0, len(dr_magnitude_SaddletoB), 1):
    
    # dr vector along step
    dr = minEnergy_pathCoords_SaddletoB[rowInd+1, :] - minEnergy_pathCoords_SaddletoB[rowInd, :] 
    
    # F vector along step
    Fstep = force_minEnergy_path_SaddletoB[rowInd,:]

    # F dot dr
    F_dot_dr_SaddletoB[rowInd] = np.dot(Fstep, dr)
    
    # dr magnitude
    dr_magnitude_SaddletoB[rowInd] = np.sqrt(np.sum(dr**2))
    
    
# COMPUTE POTENTIAL ENERGY AS A FUNCTION OF DISTANCE AWAY FROM SADDLE POINT

# compute for Saddle to A
U_of_r_SaddletoA = np.cumsum(-F_dot_dr_SaddletoA)
position_SaddletoA = np.cumsum(np.append(0,dr_magnitude_SaddletoA[:-1]))

# compute for Saddle to B
U_of_r_SaddletoB = np.cumsum(-F_dot_dr_SaddletoB)
position_SaddletoB = -np.cumsum(np.append(0,dr_magnitude_SaddletoB[:-1]))

# grand total
U_of_r = np.append(np.flip(U_of_r_SaddletoB), U_of_r_SaddletoA[1:])
F_of_r = np.append(np.flip(F_dot_dr_SaddletoB), -(F_dot_dr_SaddletoA[1:]))
path_distance = np.append(np.flip(position_SaddletoB), position_SaddletoA[1:])

# shift path distance so its not centered around 0
path_distance = np.abs(np.min(path_distance)) + path_distance

# shift U of r so that 0 = minima
U_of_r = U_of_r + np.abs(np.min(U_of_r))

    

'''
# COMPUTE LINE INTEGRAL OF F*dr
# FOR  (A --> Saddle) AND (Saddle --> B)

F_dot_dr_AtoSaddle = np.zeros(np.shape(minEnergy_pathInds_AtoSaddle)[0]-1)
F_dot_dr_SaddletoB = np.zeros(np.shape(minEnergy_pathInds_SaddletoB)[0]-1)

dr_magnitude_AtoSaddle = np.zeros(np.shape(minEnergy_pathInds_AtoSaddle)[0]-1)
dr_magnitude_SaddletoB = np.zeros(np.shape(minEnergy_pathInds_SaddletoB)[0]-1)

pathDist_AtoSaddle = np.zeros(np.shape(minEnergy_pathInds_AtoSaddle)[0]-1)
pathDist_SaddletoB = np.zeros(np.shape(minEnergy_pathInds_SaddletoB)[0]-1)
last_midPoint = minEnergy_pathCoords_AtoSaddle[rowInd, :]

# step along path from A to saddle and store F*dr at each point

for rowInd in range(0, len(dr_magnitude_AtoSaddle), 1):

    # dr vector along step
    dr = minEnergy_pathCoords_AtoSaddle[rowInd+1, :] - minEnergy_pathCoords_AtoSaddle[rowInd, :] 
        
    # F vector along step
    Fstep = force_minEnergy_path_AtoSaddle[rowInd,:]

    # F dot dr
    F_dot_dr_AtoSaddle[rowInd] = np.dot(Fstep, dr)
    
    # dr magnitude
    dr_magnitude_AtoSaddle[rowInd] = np.sqrt(np.sum(dr**2))
    
    # mid point dr
    midPoint = (minEnergy_pathCoords_AtoSaddle[rowInd+1, :] + minEnergy_pathCoords_AtoSaddle[rowInd, :])/2
    drprime = midPoint - last_midPoint
    pathDist_AtoSaddle[rowInd] = np.sqrt(np.sum(drprime**2))    
    last_midPoint = midPoint.copy()    

        
   
# step along path from Saddle toB and store F*dr at each point
for rowInd in range(0, len(dr_magnitude_SaddletoB), 1):
    
    # dr vector along step
    dr = minEnergy_pathCoords_SaddletoB[rowInd+1, :] - minEnergy_pathCoords_SaddletoB[rowInd, :] 
    
    # F vector along step
    Fstep = force_minEnergy_path_SaddletoB[rowInd,:]

    # F dot dr
    F_dot_dr_SaddletoB[rowInd] = np.dot(Fstep, dr)
    
    # dr magnitude
    dr_magnitude_SaddletoB[rowInd] = np.sqrt(np.sum(dr**2))
    
    # mid point dr
    midPoint = (minEnergy_pathCoords_SaddletoB[rowInd+1, :] + minEnergy_pathCoords_SaddletoB[rowInd, :])/2
    drprime = midPoint - last_midPoint
    pathDist_SaddletoB[rowInd] = np.sqrt(np.sum(drprime**2))  
    last_midPoint = midPoint.copy()   
    
    
# COMPUTE POTENTIAL ENERGY AS A FUNCTION OF DISTANCE AWAY FROM SADDLE POINT

# grand total
F_of_r = np.append(F_dot_dr_AtoSaddle, F_dot_dr_SaddletoB)
U_of_r = np.cumsum(-F_of_r)

path_distance = np.cumsum(np.append(pathDist_AtoSaddle, pathDist_SaddletoB))

# shift U of r so that 0 = minima
U_of_r = U_of_r + np.abs(np.min(U_of_r))
'''


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
plt.plot(path_distance, F_of_r, '-o', color='k')
plt.ylabel('force F(x)')
plt.xlabel('position x')
plt.tight_layout()
plt.savefig(fig_outpath + fName_begin + '_force_vs_path_%s.pdf' % (sweep_param_str_val))



        
#%% PLOT POTENTIAL ENERGY VS POSITION

plt.figure()
plt.plot(path_distance, U_of_r, '-o', color='k')
plt.ylabel('potential U(x)')
plt.xlabel('position x')
plt.tight_layout()
plt.savefig(fig_outpath + fName_begin + '_potentialEnergy_vs_path_%s.pdf' % (sweep_param_str_val))


#%% SAVE DATA


settings_dictionary = {'fName_begin':               fName_begin, \
                       'sweep_param_name':          sweep_param_name, \
                       'swept_param_vals':          param_vals, \
                       'func_path1':                func_path1, \
                       'func_path2':                func_path2, \
                       'func_path3':                func_path3, \
                       'func_path4':                func_path4, \
                       'fig_outpath':               fig_outpath, \
                       'data_outpath':              data_outpath, \
                       'maxCores':                  maxCores, \
                       'cores_per_job':             cores_per_job}        
    
    
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
                      'F_of_r':                         F_of_r, \
                      'U_of_r':                         U_of_r}



fName_end = ('_effectiveMFT_%s.mat' % (sweep_param_str_val))
save_filename = ( data_outpath +  fName_begin + fName_end)   
savemat(save_filename, results_dictionary)




