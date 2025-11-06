"""
Metadata for each ephys session used in Papadopoulos et al (2025).
"Modulation of metastable ensemble dynamics explains the inverted-U
relationship between tone discriminability and arousal in auditory cortex"

Loads version of data that was used for all analyses in the main text.
"""

# imports
from datetime import datetime
from dateutil import tz

# path to preprocessed data on liap@picard (saved as .h5 files for each session)
# generated using run_data_processing.py
path_to_data = '/mnt/data0/liap/PostdocWork_Oregon/My_Projects/PROJ_VariabilityGainMod/data_files/analysis_SuData/processed_data_LP/'

# Common values across sessions
experimenter = 'Jo, Suhyun'
lab = 'McCormick'
institution = 'University of Oregon'
session_description = 'Recordings from mouse auditory cortex during passive exposure to sounds'
experiment_description = session_description
subject_species = 'Mus musculus'
subject_strain = 'C57/BJ6'
related_publications = 'doi:10.1101/2024.04.04.588209'
keywords = ['auditory cortex', 'arousal', 'pupil', 'neuropixels', 'pure tones']

pupil_description = 'Pre-processed pupil diameter trace. Measurements of pupil diameter were obtained via video analysis. The pupil diameter trace was then subjected to three pre-processing steps: (1) artifact removal, (2) smoothing, and (3) max-normalization. This is the pre-processed pupil diameter trace, represented as a fraction of the maximum value across the session.'
running_description = 'Pre-processed running speed trace. Running speed was measured by a rotary encoder on the wheel. Periods of time corresponding to artifacts in the pupil diameter were then removed from the running trace, and the trace was smoothed using the same process as for the pupil diameter. This is the pre-processed running trace, in units of cm/s.'

# NOTES about formatting subjects' sex and age
# subject_sex: M, F or U (for unknown)
# subject_age: P90D (for 90 days old in ISO 8601 Duration format) or U (for unknown)

# NOTES about session_start_time
# the date and time for each session are taken from Su's original data folders located in:
# /mnt/ion-nas2/Brain_Initiative/Neuropixels/Su_NP/subject_name/ (e.g. subject_name = LA3_session3)


# Dictionary for metadata
metadata = {}

# LA3_SESSION3
metadata['LA3_session3'] = {}
metadata['LA3_session3']['identifier'] = 'LA3_session3'
metadata['LA3_session3']['session_id'] = '3'
metadata['LA3_session3']['filename'] = 'LA3_session3_processed_data.h5'
metadata['LA3_session3']['path_to_data'] = path_to_data
metadata['LA3_session3']['session_description'] = session_description
metadata['LA3_session3']['session_start_time'] = datetime(2021, 12, 17, 13, 29, 25, tzinfo=tz.gettz('US/Pacific')).isoformat()
metadata['LA3_session3']['subject_id'] = 'LA3'
metadata['LA3_session3']['Mouse_ID'] = '4863'
metadata['LA3_session3']['subject_age'] = 'P129D'
metadata['LA3_session3']['subject_description'] = 'mouse 1'
metadata['LA3_session3']['subject_species'] = subject_species
metadata['LA3_session3']['subject_strain'] = subject_strain
metadata['LA3_session3']['subject_sex'] = 'M'
metadata['LA3_session3']['experiment_description'] = experiment_description
metadata['LA3_session3']['keywords'] = keywords
metadata['LA3_session3']['experimenter'] = experimenter
metadata['LA3_session3']['lab'] = lab
metadata['LA3_session3']['institution'] = institution              
metadata['LA3_session3']['related_publications'] = related_publications


# LA8_SESSION1
metadata['LA8_session1'] = {}
metadata['LA8_session1']['identifier'] = 'LA8_session1'
metadata['LA8_session1']['session_id'] = '1'
metadata['LA8_session1']['filename'] = 'LA8_session1_processed_data.h5'
metadata['LA8_session1']['path_to_data'] = path_to_data
metadata['LA8_session1']['session_description'] = session_description
metadata['LA8_session1']['session_start_time'] = datetime(2022, 3, 8, 17, 55, 37, tzinfo=tz.gettz('US/Pacific')).isoformat()
metadata['LA8_session1']['subject_id'] = 'LA8'
metadata['LA8_session1']['Mouse_ID'] = '4885'
metadata['LA8_session1']['subject_age'] = 'P138D'
metadata['LA8_session1']['subject_description'] = 'mouse 2'
metadata['LA8_session1']['subject_species'] = subject_species
metadata['LA8_session1']['subject_strain'] = subject_strain
metadata['LA8_session1']['subject_sex'] = 'F'
metadata['LA8_session1']['experiment_description'] = experiment_description
metadata['LA8_session1']['keywords'] = keywords
metadata['LA8_session1']['experimenter'] = experimenter
metadata['LA8_session1']['lab'] = lab
metadata['LA8_session1']['institution'] = institution              
metadata['LA8_session1']['related_publications'] = related_publications


# LA8_SESSION2
metadata['LA8_session2'] = {}
metadata['LA8_session2']['identifier'] = 'LA8_session2'
metadata['LA8_session2']['session_id'] = '2'
metadata['LA8_session2']['filename'] = 'LA8_session2_processed_data.h5'
metadata['LA8_session2']['path_to_data'] = path_to_data
metadata['LA8_session2']['session_description'] = session_description
metadata['LA8_session2']['session_start_time'] = datetime(2022, 3, 9, 17, 13, 32, tzinfo=tz.gettz('US/Pacific')).isoformat()
metadata['LA8_session2']['subject_id'] = 'LA8'
metadata['LA8_session2']['Mouse_ID'] = '4885'
metadata['LA8_session2']['subject_age'] = 'P139D'
metadata['LA8_session2']['subject_description'] = 'mouse 2'
metadata['LA8_session2']['subject_species'] = subject_species
metadata['LA8_session2']['subject_strain'] = subject_strain
metadata['LA8_session2']['subject_sex'] = 'F'
metadata['LA8_session2']['experiment_description'] = experiment_description
metadata['LA8_session2']['keywords'] = keywords
metadata['LA8_session2']['experimenter'] = experimenter
metadata['LA8_session2']['lab'] = lab
metadata['LA8_session2']['institution'] = institution              
metadata['LA8_session2']['related_publications'] = related_publications



# LA9_SESSION1
metadata['LA9_session1'] = {}
metadata['LA9_session1']['identifier'] = 'LA9_session1'
metadata['LA9_session1']['session_id'] = '1'
metadata['LA9_session1']['filename'] = 'LA9_session1_processed_data.h5'
metadata['LA9_session1']['path_to_data'] = path_to_data
metadata['LA9_session1']['session_description'] = session_description
metadata['LA9_session1']['session_start_time'] = datetime(2022, 3, 17, 15, 56, 52, tzinfo=tz.gettz('US/Pacific')).isoformat()
metadata['LA9_session1']['subject_id'] = 'LA9'
metadata['LA9_session1']['Mouse_ID'] = '5285'
metadata['LA9_session1']['subject_age'] = 'P100D'
metadata['LA9_session1']['subject_description'] = 'mouse 3'
metadata['LA9_session1']['subject_species'] = subject_species
metadata['LA9_session1']['subject_strain'] = subject_strain
metadata['LA9_session1']['subject_sex'] = 'M'
metadata['LA9_session1']['experiment_description'] = experiment_description
metadata['LA9_session1']['keywords'] = keywords
metadata['LA9_session1']['experimenter'] = experimenter
metadata['LA9_session1']['lab'] = lab
metadata['LA9_session1']['institution'] = institution              
metadata['LA9_session1']['related_publications'] = related_publications


# LA9_SESSION3
metadata['LA9_session3'] = {}
metadata['LA9_session3']['identifier'] = 'LA9_session3'
metadata['LA9_session3']['session_id'] = '3'
metadata['LA9_session3']['filename'] = 'LA9_session3_processed_data.h5'
metadata['LA9_session3']['path_to_data'] = path_to_data
metadata['LA9_session3']['session_description'] = session_description
metadata['LA9_session3']['session_start_time'] = datetime(2022, 3, 22, 14, 48, 18, tzinfo=tz.gettz('US/Pacific')).isoformat()
metadata['LA9_session3']['subject_id'] = 'LA9'
metadata['LA9_session3']['Mouse_ID'] = '5285'
metadata['LA9_session3']['subject_age'] = 'P105D'
metadata['LA9_session3']['subject_description'] = 'mouse 3'
metadata['LA9_session3']['subject_species'] = subject_species
metadata['LA9_session3']['subject_strain'] = subject_strain
metadata['LA9_session3']['subject_sex'] = 'M'
metadata['LA9_session3']['experiment_description'] = experiment_description
metadata['LA9_session3']['keywords'] = keywords
metadata['LA9_session3']['experimenter'] = experimenter
metadata['LA9_session3']['lab'] = lab
metadata['LA9_session3']['institution'] = institution              
metadata['LA9_session3']['related_publications'] = related_publications



# LA9_SESSION4
metadata['LA9_session4'] = {}
metadata['LA9_session4']['identifier'] = 'LA9_session4'
metadata['LA9_session4']['session_id'] = '4'
metadata['LA9_session4']['filename'] = 'LA9_session4_processed_data.h5'
metadata['LA9_session4']['path_to_data'] = path_to_data
metadata['LA9_session4']['session_description'] = session_description
metadata['LA9_session4']['session_start_time'] = datetime(2022, 3, 23, 14, 6, 13, tzinfo=tz.gettz('US/Pacific')).isoformat()
metadata['LA9_session4']['subject_id'] = 'LA9'
metadata['LA9_session4']['Mouse_ID'] = '5285'
metadata['LA9_session4']['subject_age'] = 'P106D'
metadata['LA9_session4']['subject_description'] = 'mouse 3'
metadata['LA9_session4']['subject_species'] = subject_species
metadata['LA9_session4']['subject_strain'] = subject_strain
metadata['LA9_session4']['subject_sex'] = 'M'
metadata['LA9_session4']['experiment_description'] = experiment_description
metadata['LA9_session4']['keywords'] = keywords
metadata['LA9_session4']['experimenter'] = experimenter
metadata['LA9_session4']['lab'] = lab
metadata['LA9_session4']['institution'] = institution              
metadata['LA9_session4']['related_publications'] = related_publications


# LA9_SESSION5
metadata['LA9_session5'] = {}
metadata['LA9_session5']['identifier'] = 'LA9_session5'
metadata['LA9_session5']['session_id'] = '5'
metadata['LA9_session5']['filename'] = 'LA9_session5_processed_data.h5'
metadata['LA9_session5']['path_to_data'] = path_to_data
metadata['LA9_session5']['session_description'] = session_description
metadata['LA9_session5']['session_start_time'] = datetime(2022, 3, 24, 14, 7, 22, tzinfo=tz.gettz('US/Pacific')).isoformat()
metadata['LA9_session5']['subject_id'] = 'LA9'
metadata['LA9_session5']['Mouse_ID'] = '5285'
metadata['LA9_session5']['subject_age'] = 'P107D'
metadata['LA9_session5']['subject_description'] = 'mouse 3'
metadata['LA9_session5']['subject_species'] = subject_species
metadata['LA9_session5']['subject_strain'] = subject_strain
metadata['LA9_session5']['subject_sex'] = 'M'
metadata['LA9_session5']['experiment_description'] = experiment_description
metadata['LA9_session5']['keywords'] = keywords
metadata['LA9_session5']['experimenter'] = experimenter
metadata['LA9_session5']['lab'] = lab
metadata['LA9_session5']['institution'] = institution              
metadata['LA9_session5']['related_publications'] = related_publications

# LA11_SESSION1
metadata['LA11_session1'] = {}
metadata['LA11_session1']['identifier'] = 'LA11_session1'
metadata['LA11_session1']['session_id'] = '1'
metadata['LA11_session1']['filename'] = 'LA11_session1_processed_data.h5'
metadata['LA11_session1']['path_to_data'] = path_to_data
metadata['LA11_session1']['session_description'] = session_description
metadata['LA11_session1']['session_start_time'] = datetime(2022, 5, 12, 13, 36, 28, tzinfo=tz.gettz('US/Pacific')).isoformat()
metadata['LA11_session1']['subject_id'] = 'LA11'
metadata['LA11_session1']['Mouse_ID'] = '5543'
metadata['LA11_session1']['subject_age'] = 'P79D'
metadata['LA11_session1']['subject_description'] = 'mouse 4'
metadata['LA11_session1']['subject_species'] = subject_species
metadata['LA11_session1']['subject_strain'] = subject_strain
metadata['LA11_session1']['subject_sex'] = 'M'
metadata['LA11_session1']['experiment_description'] = experiment_description
metadata['LA11_session1']['keywords'] = keywords
metadata['LA11_session1']['experimenter'] = experimenter
metadata['LA11_session1']['lab'] = lab
metadata['LA11_session1']['institution'] = institution              
metadata['LA11_session1']['related_publications'] = related_publications



# LA11_SESSION2
metadata['LA11_session2'] = {}
metadata['LA11_session2']['identifier'] = 'LA11_session2'
metadata['LA11_session2']['session_id'] = '2'
metadata['LA11_session2']['filename'] = 'LA11_session2_processed_data.h5'
metadata['LA11_session2']['path_to_data'] = path_to_data
metadata['LA11_session2']['session_description'] = session_description
metadata['LA11_session2']['session_start_time'] = datetime(2022, 5, 13, 14, 00, 39, tzinfo=tz.gettz('US/Pacific')).isoformat()
metadata['LA11_session2']['subject_id'] = 'LA11'
metadata['LA11_session2']['Mouse_ID'] = '5543'
metadata['LA11_session2']['subject_age'] = 'P80D'
metadata['LA11_session2']['subject_description'] = 'mouse 4'
metadata['LA11_session2']['subject_species'] = subject_species
metadata['LA11_session2']['subject_strain'] = subject_strain
metadata['LA11_session2']['subject_sex'] = 'M'
metadata['LA11_session2']['experiment_description'] = experiment_description
metadata['LA11_session2']['keywords'] = keywords
metadata['LA11_session2']['experimenter'] = experimenter
metadata['LA11_session2']['lab'] = lab
metadata['LA11_session2']['institution'] = institution              
metadata['LA11_session2']['related_publications'] = related_publications



# LA11_SESSION3
metadata['LA11_session3'] = {}
metadata['LA11_session3']['identifier'] = 'LA11_session3'
metadata['LA11_session3']['session_id'] = '3'
metadata['LA11_session3']['filename'] = 'LA11_session3_processed_data.h5'
metadata['LA11_session3']['path_to_data'] = path_to_data
metadata['LA11_session3']['session_description'] = session_description
metadata['LA11_session3']['session_start_time'] = datetime(2022, 5, 14, 15, 4, 29, tzinfo=tz.gettz('US/Pacific')).isoformat()
metadata['LA11_session3']['subject_id'] = 'LA11'
metadata['LA11_session3']['Mouse_ID'] = '5543'
metadata['LA11_session3']['subject_age'] = 'P81D'
metadata['LA11_session3']['subject_description'] = 'mouse 4'
metadata['LA11_session3']['subject_species'] = subject_species
metadata['LA11_session3']['subject_strain'] = subject_strain
metadata['LA11_session3']['subject_sex'] = 'M'
metadata['LA11_session3']['experiment_description'] = experiment_description
metadata['LA11_session3']['keywords'] = keywords
metadata['LA11_session3']['experimenter'] = experimenter
metadata['LA11_session3']['lab'] = lab
metadata['LA11_session3']['institution'] = institution              
metadata['LA11_session3']['related_publications'] = related_publications



# LA11_SESSION4
metadata['LA11_session4'] = {}
metadata['LA11_session4']['identifier'] = 'LA11_session4'
metadata['LA11_session4']['session_id'] = '4'
metadata['LA11_session4']['filename'] = 'LA11_session4_processed_data.h5'
metadata['LA11_session4']['path_to_data'] = path_to_data
metadata['LA11_session4']['session_description'] = session_description
metadata['LA11_session4']['session_start_time'] = datetime(2022, 5, 15, 13, 27, 58, tzinfo=tz.gettz('US/Pacific')).isoformat()
metadata['LA11_session4']['subject_id'] = 'LA11'
metadata['LA11_session4']['Mouse_ID'] = '5543'
metadata['LA11_session4']['subject_age'] = 'P82D'
metadata['LA11_session4']['subject_description'] = 'mouse 4'
metadata['LA11_session4']['subject_species'] = subject_species
metadata['LA11_session4']['subject_strain'] = subject_strain
metadata['LA11_session4']['subject_sex'] = 'M'
metadata['LA11_session4']['experiment_description'] = experiment_description
metadata['LA11_session4']['keywords'] = keywords
metadata['LA11_session4']['experimenter'] = experimenter
metadata['LA11_session4']['lab'] = lab
metadata['LA11_session4']['institution'] = institution              
metadata['LA11_session4']['related_publications'] = related_publications


# LA12_SESSION1
metadata['LA12_session1'] = {}
metadata['LA12_session1']['identifier'] = 'LA12_session1'
metadata['LA12_session1']['session_id'] = '1'
metadata['LA12_session1']['filename'] = 'LA12_session1_processed_data.h5'
metadata['LA12_session1']['path_to_data'] = path_to_data
metadata['LA12_session1']['session_description'] = session_description
metadata['LA12_session1']['session_start_time'] = datetime(2022, 5, 24, 13, 4, 14, tzinfo=tz.gettz('US/Pacific')).isoformat()
metadata['LA12_session1']['subject_id'] = 'LA12'
metadata['LA12_session1']['Mouse_ID'] = '5547'
metadata['LA12_session1']['subject_age'] = 'P91D'
metadata['LA12_session1']['subject_description'] = 'mouse 5'
metadata['LA12_session1']['subject_species'] = subject_species
metadata['LA12_session1']['subject_strain'] = subject_strain
metadata['LA12_session1']['subject_sex'] = 'F'
metadata['LA12_session1']['experiment_description'] = experiment_description
metadata['LA12_session1']['keywords'] = keywords
metadata['LA12_session1']['experimenter'] = experimenter
metadata['LA12_session1']['lab'] = lab
metadata['LA12_session1']['institution'] = institution              
metadata['LA12_session1']['related_publications'] = related_publications


# LA12_SESSION2
metadata['LA12_session2'] = {}
metadata['LA12_session2']['identifier'] = 'LA12_session2'
metadata['LA12_session2']['session_id'] = '2'
metadata['LA12_session2']['filename'] = 'LA12_session2_processed_data.h5'
metadata['LA12_session2']['path_to_data'] = path_to_data
metadata['LA12_session2']['session_description'] = session_description
metadata['LA12_session2']['session_start_time'] = datetime(2022, 5, 25, 13, 32, 39, tzinfo=tz.gettz('US/Pacific')).isoformat()
metadata['LA12_session2']['subject_id'] = 'LA12'
metadata['LA12_session2']['Mouse_ID'] = '5547'
metadata['LA12_session2']['subject_age'] = 'P92D'
metadata['LA12_session2']['subject_description'] = 'mouse 5'
metadata['LA12_session2']['subject_species'] = subject_species
metadata['LA12_session2']['subject_strain'] = subject_strain
metadata['LA12_session2']['subject_sex'] = 'F'
metadata['LA12_session2']['experiment_description'] = experiment_description
metadata['LA12_session2']['keywords'] = keywords
metadata['LA12_session2']['experimenter'] = experimenter
metadata['LA12_session2']['lab'] = lab
metadata['LA12_session2']['institution'] = institution              
metadata['LA12_session2']['related_publications'] = related_publications



# LA12_SESSION3
metadata['LA12_session3'] = {}
metadata['LA12_session3']['identifier'] = 'LA12_session3'
metadata['LA12_session3']['session_id'] = '3'
metadata['LA12_session3']['filename'] = 'LA12_session3_processed_data.h5'
metadata['LA12_session3']['path_to_data'] = path_to_data
metadata['LA12_session3']['session_description'] = session_description
metadata['LA12_session3']['session_start_time'] = datetime(2022, 5, 26, 13, 11, 10, tzinfo=tz.gettz('US/Pacific')).isoformat()
metadata['LA12_session3']['subject_id'] = 'LA12'
metadata['LA12_session3']['Mouse_ID'] = '5547'
metadata['LA12_session3']['subject_age'] = 'P93D'
metadata['LA12_session3']['subject_description'] = 'mouse 5'
metadata['LA12_session3']['subject_species'] = subject_species
metadata['LA12_session3']['subject_strain'] = subject_strain
metadata['LA12_session3']['subject_sex'] = 'F'
metadata['LA12_session3']['experiment_description'] = experiment_description
metadata['LA12_session3']['keywords'] = keywords
metadata['LA12_session3']['experimenter'] = experimenter
metadata['LA12_session3']['lab'] = lab
metadata['LA12_session3']['institution'] = institution              
metadata['LA12_session3']['related_publications'] = related_publications



# LA12_SESSION4
metadata['LA12_session4'] = {}
metadata['LA12_session4']['identifier'] = 'LA12_session4'
metadata['LA12_session4']['session_id'] = '4'
metadata['LA12_session4']['filename'] = 'LA12_session4_processed_data.h5'
metadata['LA12_session4']['path_to_data'] = path_to_data
metadata['LA12_session4']['session_description'] = session_description
metadata['LA12_session4']['session_start_time'] = datetime(2022, 5, 27, 13, 18, 6, tzinfo=tz.gettz('US/Pacific')).isoformat()
metadata['LA12_session4']['subject_id'] = 'LA12'
metadata['LA12_session4']['Mouse_ID'] = '5547'
metadata['LA12_session4']['subject_age'] = 'P94D'
metadata['LA12_session4']['subject_description'] = 'mouse 5'
metadata['LA12_session4']['subject_species'] = subject_species
metadata['LA12_session4']['subject_strain'] = subject_strain
metadata['LA12_session4']['subject_sex'] = 'F'
metadata['LA12_session4']['experiment_description'] = experiment_description
metadata['LA12_session4']['keywords'] = keywords
metadata['LA12_session4']['experimenter'] = experimenter
metadata['LA12_session4']['lab'] = lab
metadata['LA12_session4']['institution'] = institution              
metadata['LA12_session4']['related_publications'] = related_publications
