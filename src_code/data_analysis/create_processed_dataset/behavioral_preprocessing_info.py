#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parameters for preprocessing behavioral data
"""

import numpy as np


#%% common parameters for all sessions
delta_t_compare = 0.5e-3
artifact_thresh_pupil = 0.08
artifact_window = [-0.25, 0.5]
smoothing_window_length = 1/30.
smoothing_window_step = 1e-3

#%% make dictionary to store preprocessing parameters for each session
params_dict = {}


#%% LA3_SESSION 3

remove_windows = np.array([])
params_dict['LA3_session3']={}
params_dict['LA3_session3']['remove_windows'] = remove_windows
params_dict['LA3_session3']['delta_t_compare'] = delta_t_compare
params_dict['LA3_session3']['artifact_thresh_pupil'] = artifact_thresh_pupil
params_dict['LA3_session3']['artifact_window'] = artifact_window
params_dict['LA3_session3']['smoothing_window_length'] = smoothing_window_length
params_dict['LA3_session3']['smoothing_window_step'] = smoothing_window_step

del remove_windows


#%% LA6_SESSION 1

remove_windows = np.array([])
params_dict['LA6_session1']={}
params_dict['LA6_session1']['remove_windows'] = remove_windows
params_dict['LA6_session1']['delta_t_compare'] = delta_t_compare
params_dict['LA6_session1']['artifact_thresh_pupil'] = artifact_thresh_pupil
params_dict['LA6_session1']['artifact_window'] = artifact_window
params_dict['LA6_session1']['smoothing_window_length'] = smoothing_window_length
params_dict['LA6_session1']['smoothing_window_step'] = smoothing_window_step

del remove_windows


#%% LA7_SESSION 1

remove_windows = np.array([])
params_dict['LA7_session1']={}
params_dict['LA7_session1']['remove_windows'] = remove_windows
params_dict['LA7_session1']['delta_t_compare'] = delta_t_compare
params_dict['LA7_session1']['artifact_thresh_pupil'] = artifact_thresh_pupil
params_dict['LA7_session1']['artifact_window'] = artifact_window
params_dict['LA7_session1']['smoothing_window_length'] = smoothing_window_length
params_dict['LA7_session1']['smoothing_window_step'] = smoothing_window_step

del remove_windows


#%% LA6_SESSION 2

remove_windows = np.array([])
params_dict['LA6_session2']={}
params_dict['LA6_session2']['remove_windows'] = remove_windows
params_dict['LA6_session2']['delta_t_compare'] = delta_t_compare
params_dict['LA6_session2']['artifact_thresh_pupil'] = artifact_thresh_pupil
params_dict['LA6_session2']['artifact_window'] = artifact_window
params_dict['LA6_session2']['smoothing_window_length'] = smoothing_window_length
params_dict['LA6_session2']['smoothing_window_step'] = smoothing_window_step

del remove_windows


#%% LA8_SESSION1

remove_windows = np.array([])
params_dict['LA8_session1']={}
params_dict['LA8_session1']['remove_windows'] = remove_windows
params_dict['LA8_session1']['delta_t_compare'] = delta_t_compare
params_dict['LA8_session1']['artifact_thresh_pupil'] = artifact_thresh_pupil
params_dict['LA8_session1']['artifact_window'] = artifact_window
params_dict['LA8_session1']['smoothing_window_length'] = smoothing_window_length
params_dict['LA8_session1']['smoothing_window_step'] = smoothing_window_step

del remove_windows


#%% LA8_SESSION2

remove_windows = np.array([])
params_dict['LA8_session2']={}
params_dict['LA8_session2']['remove_windows'] = remove_windows
params_dict['LA8_session2']['delta_t_compare'] = delta_t_compare
params_dict['LA8_session2']['artifact_thresh_pupil'] = artifact_thresh_pupil
params_dict['LA8_session2']['artifact_window'] = artifact_window
params_dict['LA8_session2']['smoothing_window_length'] = smoothing_window_length
params_dict['LA8_session2']['smoothing_window_step'] = smoothing_window_step

del remove_windows


#%% LA9_SESSION1

remove_windows = np.array([])
params_dict['LA9_session1']={}
params_dict['LA9_session1']['remove_windows'] = remove_windows
params_dict['LA9_session1']['delta_t_compare'] = delta_t_compare
params_dict['LA9_session1']['artifact_thresh_pupil'] = artifact_thresh_pupil
params_dict['LA9_session1']['artifact_window'] = artifact_window
params_dict['LA9_session1']['smoothing_window_length'] = smoothing_window_length
params_dict['LA9_session1']['smoothing_window_step'] = smoothing_window_step

del remove_windows


#%% LA9_SESSION 3

remove_windows = np.array( [ \
                                [225, 235], \
                                [360, 372], \
                                [402, 419], \
                                [424.3, 431.5], \
                                [493, 504], \
                                [610, 630], \
                                [811, 815], \
                                [910.1, 920], \
                                [922.2, 931.1], \
                                [1080,1098], \
                                [1595.5, 1604], \
                                [1697,1706], \
                                [1714,1729], \
                                [2318,2326], \
                                [2435,2471], \
                                [2360,2373], \
                                [3361,3364], \
                                [3576,3591], \
                                [3680,3770], \
                                [4263,4280], \
                                [4328,4334], \
                                [4346,4357], \
                                [5008,5017], \
                                [5025,5097], \
                                [5113,5128], \
                                [5153,5163], \
                                [5199, 5205], \
                                [5276,5283], \
                                [5288,5305], \
                                [5321, 5327], \
                                [5745,5751], \
                                [5757,5761], \
                                [6236,6242], \
                                [6263,6290], \
                                [6344, 6349], \
                                [6413,6426], \
                                [6482,6503], \
                                [6523,6560], \
                                [6577, 6607], \
                                [6663, 6696], \
                                [6756, 6996], \
                                [7010,7096], \
                                [7111, 7198], \
                                [7248, 7272], \
                                [7278, 7289], \
                                [7305, 7345], 
                                [7376, 7430], \
                                [7441, 7494], \
                                [7564, 7573], \
                                [7611, 7630] ])

params_dict['LA9_session3']={}
params_dict['LA9_session3']['remove_windows'] = remove_windows
params_dict['LA9_session3']['delta_t_compare'] = delta_t_compare
params_dict['LA9_session3']['artifact_thresh_pupil'] = artifact_thresh_pupil
params_dict['LA9_session3']['artifact_window'] = artifact_window
params_dict['LA9_session3']['smoothing_window_length'] = smoothing_window_length
params_dict['LA9_session3']['smoothing_window_step'] = smoothing_window_step

del remove_windows


#%% LA9_SESSION4

remove_windows = np.array([])
params_dict['LA9_session4']={}
params_dict['LA9_session4']['remove_windows'] = remove_windows
params_dict['LA9_session4']['delta_t_compare'] = delta_t_compare
params_dict['LA9_session4']['artifact_thresh_pupil'] = artifact_thresh_pupil
params_dict['LA9_session4']['artifact_window'] = artifact_window
params_dict['LA9_session4']['smoothing_window_length'] = smoothing_window_length
params_dict['LA9_session4']['smoothing_window_step'] = smoothing_window_step

del remove_windows


#%% LA9_SESSION 5


remove_windows = np.array([[5360, 5440], [7602,7679]])
params_dict['LA9_session5']={}
params_dict['LA9_session5']['remove_windows'] = remove_windows
params_dict['LA9_session5']['delta_t_compare'] = delta_t_compare
params_dict['LA9_session5']['artifact_thresh_pupil'] = artifact_thresh_pupil
params_dict['LA9_session5']['artifact_window'] = artifact_window
params_dict['LA9_session5']['smoothing_window_length'] = smoothing_window_length
params_dict['LA9_session5']['smoothing_window_step'] = smoothing_window_step

del remove_windows


#%% LA10_SESSION 2

remove_windows = np.array([])
params_dict['LA10_session2']={}
params_dict['LA10_session2']['remove_windows'] = remove_windows
params_dict['LA10_session2']['delta_t_compare'] = delta_t_compare
params_dict['LA10_session2']['artifact_thresh_pupil'] = artifact_thresh_pupil
params_dict['LA10_session2']['artifact_window'] = artifact_window
params_dict['LA10_session2']['smoothing_window_length'] = smoothing_window_length
params_dict['LA10_session2']['smoothing_window_step'] = smoothing_window_step

del remove_windows

#%% LA10_SESSION 3

remove_windows = np.array([])
params_dict['LA10_session3']={}
params_dict['LA10_session3']['remove_windows'] = remove_windows
params_dict['LA10_session3']['delta_t_compare'] = delta_t_compare
params_dict['LA10_session3']['artifact_thresh_pupil'] = artifact_thresh_pupil
params_dict['LA10_session3']['artifact_window'] = artifact_window
params_dict['LA10_session3']['smoothing_window_length'] = smoothing_window_length
params_dict['LA10_session3']['smoothing_window_step'] = smoothing_window_step

del remove_windows

#%% LA11_SESSION 1

remove_windows = np.array([])
params_dict['LA11_session1']={}
params_dict['LA11_session1']['remove_windows'] = remove_windows
params_dict['LA11_session1']['delta_t_compare'] = delta_t_compare
params_dict['LA11_session1']['artifact_thresh_pupil'] = artifact_thresh_pupil
params_dict['LA11_session1']['artifact_window'] = artifact_window
params_dict['LA11_session1']['smoothing_window_length'] = smoothing_window_length
params_dict['LA11_session1']['smoothing_window_step'] = smoothing_window_step

del remove_windows


#%% LA11_SESSION 2

remove_windows = np.array([])
params_dict['LA11_session2']={}
params_dict['LA11_session2']['remove_windows'] = remove_windows
params_dict['LA11_session2']['delta_t_compare'] = delta_t_compare
params_dict['LA11_session2']['artifact_thresh_pupil'] = artifact_thresh_pupil
params_dict['LA11_session2']['artifact_window'] = artifact_window
params_dict['LA11_session2']['smoothing_window_length'] = smoothing_window_length
params_dict['LA11_session2']['smoothing_window_step'] = smoothing_window_step

del remove_windows


#%% LA11_SESSION 3

remove_windows = np.array([])
params_dict['LA11_session3']={}
params_dict['LA11_session3']['remove_windows'] = remove_windows
params_dict['LA11_session3']['delta_t_compare'] = delta_t_compare
params_dict['LA11_session3']['artifact_thresh_pupil'] = artifact_thresh_pupil
params_dict['LA11_session3']['artifact_window'] = artifact_window
params_dict['LA11_session3']['smoothing_window_length'] = smoothing_window_length
params_dict['LA11_session3']['smoothing_window_step'] = smoothing_window_step

del remove_windows


#%% LA11_SESSION 4

remove_windows = np.array([])
params_dict['LA11_session4']={}
params_dict['LA11_session4']['remove_windows'] = remove_windows
params_dict['LA11_session4']['delta_t_compare'] = delta_t_compare
params_dict['LA11_session4']['artifact_thresh_pupil'] = artifact_thresh_pupil
params_dict['LA11_session4']['artifact_window'] = artifact_window
params_dict['LA11_session4']['smoothing_window_length'] = smoothing_window_length
params_dict['LA11_session4']['smoothing_window_step'] = smoothing_window_step

del remove_windows


#%% LA12_SESSION 1

remove_windows = np.array([[4770,4771.5],[5296, 5299.5], [5322, 5325.5], [5964, 5971.5], [6023, 6027], [6376.5, 6379], [7354, 7356.4]])
params_dict['LA12_session1']={}
params_dict['LA12_session1']['remove_windows'] = remove_windows
params_dict['LA12_session1']['delta_t_compare'] = delta_t_compare
params_dict['LA12_session1']['artifact_thresh_pupil'] = artifact_thresh_pupil
params_dict['LA12_session1']['artifact_window'] = artifact_window
params_dict['LA12_session1']['smoothing_window_length'] = smoothing_window_length
params_dict['LA12_session1']['smoothing_window_step'] = smoothing_window_step

del remove_windows


#%% LA12_SESSION 2

remove_windows = np.array([[931,933.5],[1096,1097], [2048.5, 2055], [2232.5, 2234.5], [2775.5, 2777.5], \
                           [2784, 2786], [2808, 2814.5], [5065, 5069], [5085, 5092.5], [5224, 5227.5], \
                           [6107, 6112], [6788.5, 6792], [7093, 7095], [7382, 7404]])
params_dict['LA12_session2']={}
params_dict['LA12_session2']['remove_windows'] = remove_windows
params_dict['LA12_session2']['delta_t_compare'] = delta_t_compare
params_dict['LA12_session2']['artifact_thresh_pupil'] = artifact_thresh_pupil
params_dict['LA12_session2']['artifact_window'] = artifact_window
params_dict['LA12_session2']['smoothing_window_length'] = smoothing_window_length
params_dict['LA12_session2']['smoothing_window_step'] = smoothing_window_step

del remove_windows


#%% LA12_SESSION 3

remove_windows = np.array([[1447,1448.5],[1535,1537], [1550, 1554], [1590.5, 1592], [1596.5, 1598], \
                           [1710, 1715], [1996.5, 1998], [2124, 2128], [2652.5, 2653.7], [2797, 2798], \
                           [2905, 2906.5], [3392.5, 3393.5], [3588.5, 3591], [3593.5, 3595.5], \
                           [3671, 3672.5], [5298, 5300], [5418, 5419.5], [5878, 5881], [5967.5, 5971], \
                           [5975, 5978], [6026.5, 6029], [6050, 6056.5], [6139, 6161], \
                           [6174, 6185], [6188.5, 6190], [6193.5, 6195], [6232, 6238], \
                           [6243, 6246], [6416, 6420], [6436.5, 6440], [6451.5, 6453.5], [6501, 6503], \
                           [6622.5, 6628.5], [6702.5, 6705.5], [6763.5, 6765.5], [6860, 6864.5], \
                           [7037, 7039.5], [7073, 7074], [7097, 7099], [7103.5, 7104.5], \
                           [7226.5, 7229.5], [7446, 7449], [7464.5, 7469.5]])
    
params_dict['LA12_session3']={}
params_dict['LA12_session3']['remove_windows'] = remove_windows
params_dict['LA12_session3']['delta_t_compare'] = delta_t_compare
params_dict['LA12_session3']['artifact_thresh_pupil'] = artifact_thresh_pupil
params_dict['LA12_session3']['artifact_window'] = artifact_window
params_dict['LA12_session3']['smoothing_window_length'] = smoothing_window_length
params_dict['LA12_session3']['smoothing_window_step'] = smoothing_window_step


del remove_windows


#%% LA12_SESSION 4

remove_windows = np.array([[2402, 2403.5], [2990.5, 2991.5], [3125, 3124.5], [3860, 3860.5], \
                           [3941.3, 3941.8], [3960, 3961.5], [4164.5, 4167.5], [4745.5, 4747.5], \
                           [5182.2, 5182.7], [5204, 5206.5], [5228.5, 5231.5], [5320, 5321.5], \
                           [5361, 5363], [5517.5, 5521], [5545, 5548], [5992, 5997], [6000, 6015.5], \
                           [6027.5, 6036], [6089, 6091.5], [6093.3, 6093.8], [6094.7, 6103.7], \
                           [6152, 6158], [6212, 6220], [6272.5, 6274.5], [6303.5, 6312], \
                           [6348.5, 6353.5], [6379, 6391], [6542, 6546], [6585.5, 6588.5], \
                           [6650.5, 6653], [6662, 6673]])
    
params_dict['LA12_session4']={}
params_dict['LA12_session4']['remove_windows'] = remove_windows
params_dict['LA12_session4']['delta_t_compare'] = delta_t_compare
params_dict['LA12_session4']['artifact_thresh_pupil'] = artifact_thresh_pupil
params_dict['LA12_session4']['artifact_window'] = artifact_window
params_dict['LA12_session4']['smoothing_window_length'] = smoothing_window_length
params_dict['LA12_session4']['smoothing_window_step'] = smoothing_window_step


#%% CLEAR ALL PARAMETERS

del remove_windows, delta_t_compare, artifact_thresh_pupil, artifact_window, smoothing_window_length, smoothing_window_step



