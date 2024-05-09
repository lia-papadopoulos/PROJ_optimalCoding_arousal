
"""
master script for creating intermediate pre-processed dataset
"""
 
#%%

# import preprocessing functions
from fcn_spkTemplate_cellSelection import fcn_spkTemplate_cellSelection
from fcn_behavioral_preprocessing import fcn_run_behavioral_preprocessing
from fcn_get_stimulus_data import fcn_get_stimulus_data
from fcn_save_processed_data import fcn_save_processed_data

# import parameters
import cell_selection_info
import behavioral_preprocessing_info


# session information
all_sessions_to_run = [
                       'LA3_session3', \
                       'LA8_session1', 'LA8_session2', \
                       'LA9_session1', \
                       'LA9_session3', 'LA9_session4', 'LA9_session5', \
                       'LA11_session1', 'LA11_session2', 'LA11_session3', 'LA11_session4', \
                       'LA12_session1', 'LA12_session2', 'LA12_session3', 'LA12_session4'
                      ]

    
# path to data that Su processed
processed_data_path = '/mnt/ion-nas2/Brain_Initiative/Neuropixels/Su_NP/ToLiaLuca/'

# where to save the data
outpath = '/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/analysis_SuData/processed_data_LP/'


#%% loop over sessions and process data

for indSession, session_name in enumerate(all_sessions_to_run):
    
    
    ##### cell selection #####################################################
    
    # cell selection parameters and data path
    cellSelection_inputParams = cell_selection_info.parameters
    cellSelection_rawData_path = cell_selection_info.data_path[session_name]
    
    # run cell selection
    cellSelection_params, cellSelection_results = \
        fcn_spkTemplate_cellSelection(session_name, cellSelection_inputParams, cellSelection_rawData_path, processed_data_path)    


    ##### behavioral preprocessing ############################################

    # behavioral preprocessing parameters
    behavioralPreprocessing_inputParams =  behavioral_preprocessing_info.params_dict[session_name]
    
    # run behavioral preprocessing
    behavioralPreprocessing_params, behavioralData = \
        fcn_run_behavioral_preprocessing(session_name, behavioralPreprocessing_inputParams, processed_data_path) 
        
    
    ##### extract stimulus data ##############################################
    
    stimulusInfo = fcn_get_stimulus_data(session_name, processed_data_path)
    
    
    ##### save the data ######################################################
    fcn_save_processed_data(session_name, outpath, \
                            cellSelection_params, cellSelection_results, \
                            behavioralPreprocessing_params, behavioralData, \
                            stimulusInfo)


    print('processing done. data saved.')
    print(session_name)