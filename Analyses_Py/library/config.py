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
if hasattr(main, '__file__'): 
    # running from shell
    path_study = Path(os.path.abspath(__file__)).parents[3]
else: 
    # running interactively:
    path_study = Path(os.getcwd()).parents[1]
#path_study = os.path.join(path_study, 'Experiments', 'vMemEcc')
path_data = os.path.join(path_study, 'Data')
path_postICA = op.join(path_data, 'DataMNE', 'EEG', '05.3_rejICA')
path_autoreject_logs = op.join(path_data, 'DataMNE', 'EEG', '05.4_autorej', 'logs')
path_autoreject = op.join(path_data, 'DataMNE', 'EEG', '05.4_autorej')
path_evokeds = op.join(path_data, 'DataMNE', 'EEG', '07_evokeds')
path_evokeds_cue = op.join(path_data, 'DataMNE', 'EEG', '07_evokeds', 'cue')
path_evokeds_summaries = op.join(path_evokeds, 'summaries')
path_tfrs = op.join(path_data, 'DataMNE', 'EEG', '08_tfr')
path_tfrs_summaries = op.join(path_tfrs, 'summaries')
path_epos_sorted = op.join(path_data, 'DataMNE', 'EEG', '07_epos_sorted')
path_epos_sorted_cue = op.join(path_data, 'DataMNE', 'EEG', '07_epos_sorted', 'cue')

#TODO: make more elegant (dict?)
for pp in [path_postICA, 
           path_evokeds, 
           path_evokeds_cue,
           path_evokeds_summaries, 
           path_epos_sorted, 
           path_tfrs_summaries, 
           path_epos_sorted_cue, 
           path_autoreject_logs, 
           path_autoreject]:
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
alpha_freqs = [9, 12]

# times:
times_dict = dict(CDA_start = 0.450, 
                  CDA_end = 1.450, 
                  blink_dur = 0.8,
                  fix_dur = 0.8, 
                  cue_dur = 0.8, 
                  stim_dur = 0.2, 
                  retention_dur = 2.0)

# parallelization: 
n_jobs = 4 # let's leave the CPU some air to breath

# subjects: 
n_subjects_total = 27
ids_missing_subjects = [11, 14, 19]
