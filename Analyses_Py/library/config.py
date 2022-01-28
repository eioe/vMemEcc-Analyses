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
    
path_extracted_vars = op.join(path_study, 'Writing', 'Other',
                              'VME_extracted_vars.json')
paths['extracted_vars'] = path_extracted_vars

path_data = os.path.join(path_study, 'Data2022')
paths['00_raw'] = os.path.join(path_data, 'DataMNE', 'EEG', '00_raw')
paths['01_prepared'] = os.path.join(path_data, 'DataMNE', 'EEG', '01_prepared')
paths['01_prepared-events'] = os.path.join(path_data, 'DataMNE', 'EEG', '01_prepared-events')

for p in paths:
    if not op.exists(paths[p]):
        os.makedirs(paths[p])
        print('creating dir: ' + paths[p]) 

path_postICA = op.join(path_data, 'DataMNE', 'EEG', '05.3_rejICA')
path_rejepo = op.join(path_data, 'DataMNE', 'EEG', '05.1_rejepo')
path_reject_epos_extern = op.join(path_rejepo, 'CSV_rejEpos_ET')
path_rejepo_summaries = op.join(path_rejepo, 'summaries')
path_autoreject_logs = op.join(path_data, 'DataMNE', 'EEG', '05.4_autorej', 'logs')
path_autoreject = op.join(path_data, 'DataMNE', 'EEG', '05.4_autorej')
path_evokeds = op.join(path_data, 'DataMNE', 'EEG', '07_evokeds')
path_evokeds_cue = op.join(path_data, 'DataMNE', 'EEG', '07_evokeds', 'cue')
path_evokeds_summaries = op.join(path_evokeds, 'summaries')
path_tfrs = op.join(path_data, 'DataMNE', 'EEG', '08_tfr')
path_tfrs_summaries = op.join(path_tfrs, 'summaries')
path_epos_sorted = op.join(path_data, 'DataMNE', 'EEG', '07_epos_sorted')
path_epos_sorted_cue = op.join(path_data, 'DataMNE', 'EEG', '07_epos_sorted', 'cue')
path_decod_temp = op.join(path_data, 'DataMNE', 'EEG', '09_temporal_decoding')
path_decod_tfr = op.join(path_data, 'DataMNE', 'EEG', '10_tfr_decoding')
path_plots = op.join(path_study, 'Plots')

#TODO: make more elegant (dict?)
for pp in [path_postICA,
           path_rejepo,
           path_rejepo_summaries,
           path_evokeds,
           path_evokeds_cue,
           path_evokeds_summaries,
           path_epos_sorted,
           path_tfrs_summaries,
           path_epos_sorted_cue,
           path_autoreject_logs,
           path_autoreject,
           path_decod_temp,
           path_plots]:
    if not op.exists(pp):
        os.makedirs(pp)
        print('creating dir: ' + pp) 

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
                  bl_dur_erp = 0.4, 
                  bl_dur_tfr = 0.2)

# parallelization: 
n_jobs = 4 # let's leave the CPU some air to breath

# subjects: 
n_subjects_total = 27
ids_missing_subjects = [11, 14, 19]
ids_excluded_subjects = [7, 12, 22]

# font sizes:
plt_label_size = 18

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