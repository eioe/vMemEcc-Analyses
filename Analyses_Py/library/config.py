"""
===========
Config file
===========
Configuration parameters for the study. This should be in a folder called
``library/`` inside the ``processing/`` directory.

Code inspired by: 
https://github.com/mne-tools/mne-biomag-group-demo/blob/master/scripts/processing/library/config.py

"""

import os  
import os.path as op
import numpy as np
from pathlib import Path
import __main__ as main

study_path = '../../'

# Paths:

paths = dict()

if hasattr(main, '__file__'): 
    # running from shell
    path_study = Path(os.path.abspath(__file__)).parents[3]
    paths['study'] = path_study
else: 
    # running interactively:
    path_study = Path(os.getcwd()).parents[1]
    paths['study'] = path_study
    



path_data = os.path.join(path_study, 'Data2022')
paths['00_raw'] = os.path.join(path_data, 'DataMNE', 'EEG', '00_raw')
paths['01_prepared'] = os.path.join(path_data, 'DataMNE', 'EEG', '01_prepared')
paths['01_prepared-events'] = os.path.join(path_data, 'DataMNE', 'EEG', '01_prepared-events')
paths['02_epochs'] = os.path.join(path_data, 'DataMNE', 'EEG', '02_epochs')
paths['03_preproc'] = op.join(path_data, 'DataMNE', 'EEG', '03_preproc')
paths['03_preproc-ica'] = os.path.join(path_data, 'DataMNE', 'EEG', '03_preproc', 'ica')
paths['03_preproc-ica-ar'] = os.path.join(path_data, 'DataMNE', 'EEG', '03_preproc', 'ica', 'ar')
paths['03_preproc-ica-eog'] = os.path.join(path_data, 'DataMNE', 'EEG', '03_preproc', 'ica', 'eog')
paths['03_preproc-ar'] = os.path.join(path_data, 'DataMNE', 'EEG', '03_preproc', 'ar')
paths['03_preproc-rejectET'] = op.join(paths['03_preproc'], 'reject-ET')
paths['03_preproc-rejectET-CSVs'] = op.join(paths['03_preproc'], 'reject-ET', 'CSV_rejEpos_ET')
paths['03_preproc-pooled'] = op.join(paths['03_preproc'], 'pooled')
paths['04_evokeds'] = op.join(path_data, 'DataMNE', 'EEG', '04_evokeds')
paths['04_evokeds-pooled'] = op.join(paths['04_evokeds'], 'pooled')
paths['04_evokeds-CDA'] = op.join(paths['04_evokeds'], 'CDA')
paths['04_evokeds-PNP'] = op.join(paths['04_evokeds'], 'PNP')
paths['05_tfrs'] = op.join(path_data, 'DataMNE', 'EEG', '05_tfrs')
paths['05_tfrs-summaries'] = op.join(paths['05_tfrs'], 'summaries')

paths['plots'] = op.join(path_study, 'Plots2022')
paths['extracted_vars_dir'] = op.join(path_study, 'Writing', 'Other', 'ExtractedVariables2022')

for p in paths:
    if not op.exists(paths[p]):
        os.makedirs(paths[p])
        print('creating dir: ' + paths[p]) 

# Add paths to files:
paths['extracted_vars_file'] = op.join(paths['extracted_vars_dir'],
                                       'VME_extracted_vars.json')
        
        
# path_postICA = op.join(path_data, 'DataMNE', 'EEG', '05.3_rejICA')
# path_rejepo = op.join(path_data, 'DataMNE', 'EEG', '05.1_rejepo')
# path_reject_epos_extern = op.join(path_rejepo, 'CSV_rejEpos_ET')
# path_rejepo_summaries = op.join(path_rejepo, 'summaries')
# path_autoreject_logs = op.join(path_data, 'DataMNE', 'EEG', '05.4_autorej', 'logs')
# path_autoreject = op.join(path_data, 'DataMNE', 'EEG', '05.4_autorej')
# path_evokeds = op.join(path_data, 'DataMNE', 'EEG', '07_evokeds')
# path_evokeds_cue = op.join(path_data, 'DataMNE', 'EEG', '07_evokeds', 'cue')
# path_evokeds_summaries = op.join(path_evokeds, 'summaries')
# path_tfrs = op.join(path_data, 'DataMNE', 'EEG', '08_tfr')
# path_tfrs_summaries = op.join(path_tfrs, 'summaries')
# path_epos_sorted = op.join(path_data, 'DataMNE', 'EEG', '07_epos_sorted')
# path_epos_sorted_cue = op.join(path_data, 'DataMNE', 'EEG', '07_epos_sorted', 'cue')
# path_decod_temp = op.join(path_data, 'DataMNE', 'EEG', '09_temporal_decoding')
# path_decod_tfr = op.join(path_data, 'DataMNE', 'EEG', '10_tfr_decoding')
# path_plots = op.join(path_study, 'Plots')



# conditions:
factor_levels = [load + ecc for load in ['LoadLow', 'LoadHigh',''] 
                 for ecc in ['EccS', 'EccM', 'EccL','']][:-1] 

factor_dict = {name: factor_levels.index(name) for name in factor_levels}

# ROIs: 
chans_CDA_dict = {'Left': ['P3', 'P5', 'PO3', 'PO7', 'O1'], 
                  'Right': ['P4', 'P6', 'PO4', 'PO8', 'O2'], 
                  'Contra': ['P3', 'P5', 'PO3', 'PO7', 'O1'], 
                  'Ipsi': ['P4', 'P6', 'PO4', 'PO8', 'O2']}
chans_CDA_all = [ch for v in list(chans_CDA_dict.values())[0:2] for ch in v]

# Freqs: 
alpha_freqs = [8, 13]

# times:
times_dict = dict(CDA_start = 0.450, 
                  CDA_end = 1.450, 
                  blink_dur = 0.8,
                  fix_dur = 0.8, 
                  cue_dur = 0.8, 
                  stim_dur = 0.2, 
                  retention_dur = 2.0, 
                  bl_dur_erp = 0.2, 
                  bl_dur_tfr = 0.2)

# parallelization: 
n_jobs = 50 # adapt this to the number of cores in your machine. If in doubt, 6-8 is probably a good choice.

# subjects: 
n_subjects_total = 27
ids_missing_subjects = [11, 14, 19]
ids_excluded_subjects =  [12, 13, 22]   # [7, 12, 22]<<< with old preprocessing

# font sizes:
plt_label_size = 12

# colors: 
#"#66C2A5" "#3288BD"

colors = dict()
colors['LoadHigh'] = "#F1942E"
colors['LoadLow'] = "#32628A"
colors['Load High'] = "#F1942E"
colors['Load Low'] = "#32628A"
colors['4'] = "#F1942E"
colors['2'] = "#32628A"
colors['Ipsi'] = 'purple'
colors['Contra'] = 'pink'

colors['4°'] = "#00A878"
colors['9°'] = "#FCEC52"
colors['14°'] = "#FE5E41"
colors['EccS'] = "#00A878"
colors['EccM'] = "#FCEC52"
colors['EccL'] = "#FE5E41"
colors['Chance'] = "#B5B4B3"
colors['Random'] = "#B5B4B3"
colors['Load'] = "#72DDED"
colors['Diff'] = "black"
colors['Ipsi'] = '#FAC748'
colors['Contra'] = '#8390FA'

# labels
labels = dict()
labels['EccS'] = '4°'
labels['EccM'] = '9°'
labels['EccL'] = '14°'
labels['LoadLow'] = '2'
labels['LoadHigh'] = '4'
labels['Ecc'] = 'Eccentricity'
labels['Load'] = 'Size Memory Array'
labels['Chance'] = 'Random'
labels['Random'] = 'Random'

event_dict = {'CueL': ['Stimulus/S150',
                       'Stimulus/S152',
                       'Stimulus/S154',
                       'Stimulus/S156',
                       'Stimulus/S158',
                       'Stimulus/S160',
                       'Stimulus/S162',
                       'Stimulus/S164',
                       'Stimulus/S166',
                       'Stimulus/S168',
                       'Stimulus/S170',
                       'Stimulus/S172'],
              'CueR': ['Stimulus/S151',
                       'Stimulus/S153',
                       'Stimulus/S155',
                       'Stimulus/S157',
                       'Stimulus/S159',
                       'Stimulus/S161',
                       'Stimulus/S163',
                       'Stimulus/S165',
                       'Stimulus/S167',
                       'Stimulus/S169',
                       'Stimulus/S171',
                       'Stimulus/S173'],
                'LoadLow': ['Stimulus/S151',
                            'Stimulus/S153',
                            'Stimulus/S159',
                            'Stimulus/S161',
                            'Stimulus/S167',
                            'Stimulus/S169',
                            'Stimulus/S150',
                            'Stimulus/S152',
                            'Stimulus/S158',
                            'Stimulus/S160',
                            'Stimulus/S166',
                            'Stimulus/S168'],
                'LoadHigh': ['Stimulus/S155',
                             'Stimulus/S157',
                             'Stimulus/S163',
                             'Stimulus/S165',
                             'Stimulus/S171',
                             'Stimulus/S173',
                             'Stimulus/S154',
                             'Stimulus/S156',
                             'Stimulus/S162',
                             'Stimulus/S164',
                             'Stimulus/S170',
                             'Stimulus/S172'],
                'EccS': ['Stimulus/S151',
                         'Stimulus/S153',
                         'Stimulus/S155',
                         'Stimulus/S157',
                         'Stimulus/S150',
                         'Stimulus/S152',
                         'Stimulus/S154',
                         'Stimulus/S156'],
                'EccM': ['Stimulus/S159',
                         'Stimulus/S161',
                         'Stimulus/S163',
                         'Stimulus/S165',
                         'Stimulus/S158',
                         'Stimulus/S160',
                         'Stimulus/S162',
                         'Stimulus/S164'],
                'EccL': ['Stimulus/S167',
                         'Stimulus/S169',
                         'Stimulus/S171',
                         'Stimulus/S173',
                         'Stimulus/S166',
                         'Stimulus/S168',
                         'Stimulus/S170',
                         'Stimulus/S172']}