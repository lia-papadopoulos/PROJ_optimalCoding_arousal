"""
parameters for running spike template amplitude based cell selection
"""

# PARAMETERS
parameters = {}

parameters['windowSize'] = 5*60.
parameters['fracData_affected'] = 0.1
parameters['ratioCut'] = 10.
parameters['distanceCut'] = 40.
parameters['noiseFloor_cut'] = 15.      # tradeoff between saying all data is below noise floor (cut too high) and not including data as noise floor (cut too low)
parameters['peakLocation_cut'] = 25.
parameters['rateThresh'] = 0.25
parameters['cut_type'] = 'version1'     # version1 or version2

# PATHS TO DATA
data_path = {}
data_path['LA3_session3'] = '/mnt/ion-nas2/Brain_Initiative/Neuropixels/Su_NP/LA3/2021-12-17_13-29-25_Sort/Record Node 104/experiment1/recording1/continuous/Neuropix-PXI-102.0/'
data_path['LA8_session1'] = '/mnt/ion-nas2/Brain_Initiative/Neuropixels/Su_NP/LA8/2022-03-08_17-55-37_Sort/Record Node 104/experiment1/recording1/continuous/Neuropix-PXI-102.0/'
data_path['LA8_session2'] = '/mnt/ion-nas2/Brain_Initiative/Neuropixels/Su_NP/LA8/2022-03-09_17-13-32_Sort/Record Node 104/experiment1/recording1/continuous/Neuropix-PXI-102.0/'
data_path['LA9_session1'] = '/mnt/ion-nas2/Brain_Initiative/Neuropixels/Su_NP/LA9/2022-03-17_15-56-52_Sort/Record Node 104/experiment1/recording1/continuous/Neuropix-PXI-102.0/'
data_path['LA9_session3'] = '/mnt/ion-nas2/Brain_Initiative/Neuropixels/Su_NP/LA9/2022-03-22_14-48-18_Sort/Record Node 104/experiment1/recording1/continuous/Neuropix-PXI-102.0/'
data_path['LA9_session4'] = '/mnt/ion-nas2/Brain_Initiative/Neuropixels/Su_NP/LA9/2022-03-23_14-06-13_Sort/Record Node 104/experiment1/recording1/continuous/Neuropix-PXI-102.0/'
data_path['LA9_session5'] = '/mnt/ion-nas2/Brain_Initiative/Neuropixels/Su_NP/LA9/2022-03-24_14-07-22_Sort/Record Node 104/experiment1/recording1/continuous/Neuropix-PXI-102.0/'
data_path['LA11_session1'] = '/mnt/ion-nas2/Brain_Initiative/Neuropixels/Su_NP/LA11/2022-05-12_13-36-28_Sort/Record Node 102/experiment1/recording1/continuous/Neuropix-PXI-100.0/'
data_path['LA11_session2'] = '/mnt/ion-nas2/Brain_Initiative/Neuropixels/Su_NP/LA11/2022-05-13_14-00-39_Sort/Record Node 102/experiment1/recording1/continuous/Neuropix-PXI-100.0/'
data_path['LA11_session3'] = '/mnt/ion-nas2/Brain_Initiative/Neuropixels/Su_NP/LA11/2022-05-14_15-04-29_Sort/Record Node 102/experiment1/recording1/continuous/Neuropix-PXI-100.0/'
data_path['LA11_session4'] = '/mnt/ion-nas2/Brain_Initiative/Neuropixels/Su_NP/LA11/2022-05-15_13-27-58_Sort/Record Node 102/experiment1/recording1/continuous/Neuropix-PXI-100.0/'
data_path['LA12_session1'] = '/mnt/ion-nas2/Brain_Initiative/Neuropixels/Su_NP/LA12/2022-05-24_13-04-14_Sort/Record Node 102/experiment1/recording1/continuous/Neuropix-PXI-100.0/'
data_path['LA12_session2'] = '/mnt/ion-nas2/Brain_Initiative/Neuropixels/Su_NP/LA12/2022-05-25_13-32-39_Sort/Record Node 102/experiment1/recording1/continuous/Neuropix-PXI-100.0/'
data_path['LA12_session3'] = '/mnt/ion-nas2/Brain_Initiative/Neuropixels/Su_NP/LA12/2022-05-26_13-11-10_Sort/Record Node 102/experiment1/recording1/continuous/Neuropix-PXI-100.0/'
data_path['LA12_session4'] = '/mnt/ion-nas2/Brain_Initiative/Neuropixels/Su_NP/LA12/2022-05-27_13-18-06_Sort/Record Node 102/experiment1/recording1/continuous/Neuropix-PXI-100.0/'