
#%% basic imports
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
from scipy.io import savemat
from scipy.io import loadmat
import glob

#%% parameters
from mftParams_JeePlus_sweep_2cluster_fullMFT import mft_params

#%% instatiate parameters for running mft
m_params = mft_params()

#%% baseline simulation parameters
simParams_path = m_params.simParams_path
sys.path.append(simParams_path)
from simParams_arousalSweep_noDisorder import sim_params

#%% unpack mft_params
func_path1 = m_params.func_path1
func_path2 = m_params.func_path2
fig_outpath = m_params.fig_outpath
data_outpath = m_params.data_outpath
pathName_paramData = m_params.pathName_paramData
fName_begin_paramData = m_params.fName_begin_paramData
fName_begin = m_params.fName_begin
n_Epops = m_params.n_Epops
n_Ipops = m_params.n_Ipops
JplusEE_array = m_params.JplusEE_array
nu_vec_selective = m_params.nu_vec_selective
nu_vec_nonselective = m_params.nu_vec_nonselective

#%% load custom functions
sys.path.append(func_path1)    
sys.path.append(func_path2)     
from fcn_compute_firing_stats import Dict2Class
import fcn_MFT_fixedInDeg_clusNet
from fcn_make_network_2cluster import fcn_make_network_cluster


#%% load simulation parameters 

if m_params.newRun == True:
      
    params = sim_params()
    params.set_dependent_vars()
    W, popsizeE, popsizeI = fcn_make_network_cluster(params)    
    params.popsizeE = popsizeE
    params.popsizeI = popsizeI
    print('setup done')
    
else:

    files =  sorted(glob.glob(pathName_paramData + fName_begin_paramData + '*'))
    data = loadmat(files[0], simplify_cells=True)
    sim_params = data['sim_params']
    print(files[0])
    params = Dict2Class(sim_params)


#%% setup for sweep over JeePlus

JplusEE_backwards = np.flip(JplusEE_array)

nu_e_backwards = np.zeros((n_Epops,len(JplusEE_backwards)))
nu_i_backwards = np.zeros((n_Ipops,len(JplusEE_backwards)))
MaxReEig_backwards = np.zeros((len(JplusEE_backwards)))

JplusEE_forwards = JplusEE_array.copy()

nu_e_forwards = np.zeros((n_Epops,len(JplusEE_forwards)))
nu_i_forwards = np.zeros((n_Ipops,len(JplusEE_forwards)))
MaxReEig_forwards = np.zeros((len(JplusEE_forwards)))


#%% RUN THE MFT

# backwards sweep

# initial rates
m_params.nu_vec = m_params.nu_vec_selective.copy()

# sweep over Jplus
for Jind in range(0,len(JplusEE_backwards),1):
    
    # update value of Jplus
    params.JplusEE = JplusEE_backwards[Jind]
    
    # find fixed point 
    mft_results = fcn_MFT_fixedInDeg_clusNet.fcn_master_MFT(params, m_params)   
        
    # save results
    nu_e_backwards[:,Jind] = mft_results['nu_e']
    nu_i_backwards[:,Jind] = mft_results['nu_i']
    MaxReEig_backwards[Jind] = np.max(mft_results['realPart_eigvals_S'])
               
    # update with new fixed point
    m_params.nu_vec = np.hstack((nu_e_backwards[:,Jind], nu_i_backwards[:,Jind]))
    
    # if no solution found
    if np.isnan(m_params.nu_vec[0]) == True:
        
        # try non selective fixed point
        m_params.nu_vec = m_params.nu_vec_nonselective.copy()
        
        # solve mft
        mft_results = fcn_MFT_fixedInDeg_clusNet.fcn_master_MFT(params, m_params)
            
        # save results
        nu_e_backwards[:,Jind] = mft_results['nu_e']
        nu_i_backwards[:,Jind] = mft_results['nu_i']
        MaxReEig_backwards[Jind] = np.max(mft_results['realPart_eigvals_S'])
        
        # update with new fixed point
        m_params.nu_vec = np.hstack((nu_e_backwards[:,Jind], nu_i_backwards[:,Jind]))
        
    
    print(Jind)


 
# forwards sweep

# initial rates
m_params.nu_vec = m_params.nu_vec_nonselective.copy()

for Jind in range(0,len(JplusEE_forwards),1):
    
    # update value of Jplus
    params.JplusEE = JplusEE_forwards[Jind]
        
    # solve mft
    mft_results = fcn_MFT_fixedInDeg_clusNet.fcn_master_MFT(params, m_params)  

    # save results
    nu_e_forwards[:,Jind] = mft_results['nu_e']
    nu_i_forwards[:,Jind] = mft_results['nu_i']
    MaxReEig_forwards[Jind] = np.max(mft_results['realPart_eigvals_S'])
               
    # update with new fixed point
    m_params.nu_vec = np.hstack((nu_e_forwards[:,Jind], nu_i_forwards[:,Jind]))
            
    # if no solution found
    if np.isnan(m_params.nu_vec[0]) == True:
        
        # try non selective fixed point
        m_params.nu_vec = m_params.nu_vec_nonselective.copy()
        
        # solve mft
        mft_results = fcn_MFT_fixedInDeg_clusNet.fcn_master_MFT(params, m_params)  

        # save results
        nu_e_forwards[:,Jind] = mft_results['nu_e']
        nu_i_forwards[:,Jind] = mft_results['nu_i']
        MaxReEig_forwards[Jind] = np.max(mft_results['realPart_eigvals_S'])
        
        # update with new fixed point
        m_params.nu_vec = np.hstack((nu_e_forwards[:,Jind], nu_i_forwards[:,Jind]))
    
    print(Jind)


#%% SAVE RESULTS


settings_dictionary = {'fName_begin':               fName_begin, \
                       'JplusEE_values':            JplusEE_array, \
                       'func_path1':                func_path1, \
                       'func_path2':                func_path2, \
                       'fig_outpath':               fig_outpath, \
                       'data_outpath':              data_outpath, 
                       'pathName_paramData':        pathName_paramData, \
                       'fName_begin_paramData':     fName_begin_paramData}        
    
    
results_dictionary = {'sim_params':                     params, \
                      'mft_params':                     m_params, \
                      'settings_dictionary':            settings_dictionary, \
                      'JplusEE_backwards':              JplusEE_backwards, \
                      'nu_e_backwards':                 nu_e_backwards, \
                      'nu_i_backwards':                 nu_i_backwards, \
                      'MaxReEig_backwards':             MaxReEig_backwards, \
                      'JplusEE_forwards':               JplusEE_forwards, \
                      'nu_e_forwards':                  nu_e_forwards, \
                      'nu_i_forwards':                  nu_i_forwards, \
                      'MaxReEig_forwards':              MaxReEig_forwards}


fName_end = ('_JeePlus_sweep_2cluster_fullMFT.mat')
save_filename = ( data_outpath +  fName_begin + fName_end)   
savemat(save_filename, results_dictionary)



#%% PLOTTING

plt.figure()
plt.plot( JplusEE_backwards, nu_e_backwards[0,:], '-', color='lightseagreen', linewidth=3, label='active E' )
plt.plot( JplusEE_backwards, nu_e_backwards[1,:], '-', color='slateblue', linewidth=3, label='inactive E' )
plt.plot( JplusEE_backwards, nu_e_backwards[0,:], '--', color='lightseagreen', linewidth=1 )
plt.plot( JplusEE_backwards, nu_e_backwards[1,:], '--', color='slateblue', linewidth=1)
plt.xlabel('Jee+')
plt.ylabel('firing rate [spks/sec]')    
plt.legend()
plt.tight_layout()
plt.savefig(fig_outpath + fName_begin + '_JeePlus_sweep_2cluster_fullMFT.pdf', transparent=True)



