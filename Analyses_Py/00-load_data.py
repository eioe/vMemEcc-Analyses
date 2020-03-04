"""
=============================
00. Load data from disk
=============================

Loads vMemEcc raw EEG data (BrainVision format) from disk. 
"""

import os
import os.path as op
from pathlib import Path

import mne

subsub = 'VME_S05'
get_data_from_sdrive = False

# set paths:

path_study = Path(os.getcwd()).parents[1] #str(Path(__file__).parents[2])
# note: returns Path object >> cast for string

path_data = os.path.join(path_study, 'Data')

if (get_data_from_sdrive): 
    path_sdrive = os.path.join('S:\\', 'Meine Bibliotheken', 'Experiments', 'vMemEcc')
    path_data_in = os.path.join(path_sdrive, 'Data')
else:
    path_data_in = path_data

path_eegdata = os.path.join(path_data_in, 'SubjectData', '%s', 'EEG')
path_outp = op.join(path_data, 'DataMNE', 'EEG', '00_raw')
if not op.exists(path_outp):
    os.makedirs(path_outp)

def load_data_raw(subID):
    # Get data:
    path_sub = path_eegdata % subID
    raw_files = [op.join(path_sub, f) for f in os.listdir(path_sub) if f.endswith('.vhdr')]
    raw = mne.io.concatenate_raws([mne.io.read_raw_brainvision(f, preload=False) for f in raw_files])
    return raw

def save_data(data, filename, path, append=''):
    ff = op.join(path, filename + append + '.fif')
    #print("Saving %s ..." % ff)
    data.save(fname=ff, overwrite=True)

######################################################################################################

data_raw = load_data_raw(subsub)
save_data(data_raw, subsub, path_outp, '-raw')
