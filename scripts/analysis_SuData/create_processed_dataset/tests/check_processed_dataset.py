'''
compare cell selection using original method and after processing
'''
#%%
import h5py
import numpy as np
import sys
from scipy.io import loadmat


# session information
all_sessions_to_run = [#'LA3_session3', \
                       'LA3_session3', \
                       'LA8_session1', 'LA8_session2', \
                       'LA9_session1', \
                       'LA9_session3', 'LA9_session4', 'LA9_session5', \
                       'LA11_session1', 'LA11_session2', 'LA11_session3', 'LA11_session4', \
                       'LA12_session1', 'LA12_session2', 'LA12_session3', 'LA12_session4'
                      ]

# add paths
sys.path.append('/home/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/scripts/functions/')    
sys.path.append('/home/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/scripts/analysis_SuData/parameters/')

# preprocessing
from fcn_preprocess_SuData import fcn_run_preprocessing
import preprocessing_params as pre_params

# path to raw data from su
path0 = '/mnt/ion-nas2/Brain_Initiative/Neuropixels/Su_NP/ToLiaLuca/'
  
# path to original cell selection data
path1 = '/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/analysis_SuData/amplitudeBased_cell_selection/'

# path to processed data
path2 = '/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/analysis_SuData/processed_data_LP/'


#%%

for indSession, session_name in enumerate(all_sessions_to_run):
    
    print(session_name)

    # load original data
    filename =  ( (path1 + '%s_amplitudeBased_cell_selection.mat') % (session_name) ) 
    original_data = loadmat(filename, simplify_cells=True)
    good_units_original = original_data['goodClusters_all']
    
    f = h5py.File(path0 + session_name + '.mat','r')
    
    # extract data
    if session_name == 'LA12_session1':
        time_stamp = f['time_stamp'][0]
        pupil_trace = f['pupil'][0]
        walk_trace = f['walk'][0]
        whisk_trace = f['whisk'][0]
        
    stim_on = f['stim_on_time'][0]
    stim_Hz = f['stimHz'][0]
    nCells = f['spk_Good_Aud'].shape[1]
    
    # cell spike times
    cell_spk_times = np.zeros(nCells,dtype='object')
    for i in range(0,nCells):
        spkTimes_ref = f['spk_Good_Aud'][0,i]
        cell_spk_times[i] = f[spkTimes_ref][0]
    
    
    # close file
    f.close()
    
    cell_spk_times = cell_spk_times[good_units_original]

    
    if session_name == 'LA12_session1':
        
        session_pre_params = pre_params.params_dict[session_name]

        time, pupil, run, whisk = fcn_run_preprocessing(time_stamp, \
                                                pupil_trace, walk_trace, whisk_trace, \
                                                session_pre_params)
    

    # load processed data
    filename = (('%s%s_processed_data.h5') % (path2, session_name))
    new_data = h5py.File(filename, 'r')
    
    cell_spk_times_new = new_data['cell_spikeTimes'][:]
    good_units_new = new_data['good_units'][:]
    
    if session_name == 'LA12_session1':
        pupil_new = new_data['behavioral_data']['pupil_trace'][:]
        run_new = new_data['behavioral_data']['run_trace'][:]
        time_new = new_data['behavioral_data']['time'][:]
        
    stim_on_new = new_data['stim_data']['stim_on_time'][:]
    stim_Hz_new = new_data['stim_data']['stim_Hz'][:]


    new_data.close()

    # compare
    are_equal = np.array_equal(good_units_original, good_units_new)
    if are_equal == False:
        sys.exit('cell selection not the same')
    else:
        print('PASSED: cell selection')
    
    # compare
    for indCell in range(0, len(good_units_new)):
        
        are_equal = np.array_equal(cell_spk_times[indCell], cell_spk_times_new[indCell])
        
        if are_equal == False:
            sys.exit('cell spk times not the same')
            
    print('PASSED: spike times')

            
    # compare
    are_equal = np.array_equal(stim_on, stim_on_new)
    if are_equal == False:
        sys.exit('stim onset times not the same')
    else:
        print('PASSED: stim onset times')
        
     # compare
    are_equal = np.array_equal(stim_Hz, stim_Hz_new)
    if are_equal == False:
        sys.exit('stim freq not the same')
    else:
        print('PASSED: stim freq')      
        
    
    # compare
    if session_name == 'LA12_session1':
        
        are_equal = np.array_equal(pupil[~np.isnan(pupil)], pupil_new[~np.isnan(pupil_new)])
        if are_equal == False:
            sys.exit('pupil trace not the same')
        else:
            print('PASSED: pupil')    
            
        are_equal = np.array_equal(run[~np.isnan(run)], run_new[~np.isnan(run_new)])
        if are_equal == False:
            sys.exit('run trace not the same')
        else:
            print('rPASSED: run')  
            
        are_equal = np.array_equal(time, time_new)
        if are_equal == False:
            sys.exit('time pts not the same')
        else:
            print('PASSED: time pts')  