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

study_path = '../../'

# Paths:
path_study = Path(os.getcwd()).parents[1]
path_data = os.path.join(path_study, 'Data')
path_postICA = op.join(path_data, 'DataMNE', 'EEG', '05.3_rejICA')
path_autoreject_logs = op.join(path_data, 'DataMNE', 'EEG', '05.4_autorej')
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
           path_autoreject_logs]:
    if not op.exists(pp):
        os.makedirs(pp)
        print('creating dir: ' + pp) 

# conditions:
factor_levels = [load + ecc for load in ['LoadLow', 'LoadHigh',''] 
                 for ecc in ['EccS', 'EccM', 'EccL','']][:-1] 

factor_dict = {name: factor_levels.index(name) for name in factor_levels}

# times:
times_dict = dict(CDA_start = 0.450, 
                  CDA_end = 1.450, 
                  blink_dur = 0.8,
                  fix_dur = 0.8, 
                  cue_dur = 0.8, 
                  stim_dur = 0.2, 
                  retention_dur = 2.0)

# parallelization: 
n_jobs = -2 # let's leave the CPU some air to breath

# subjects: 
n_subjects_total = 27
ids_missing_subjects = [11, 14, 19]