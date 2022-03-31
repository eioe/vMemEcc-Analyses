"""
=============================
01.1 Epoch data
=============================

Extracts 4 different kinds of epochs:

name                timelocked to            intervall           filter    
--------------------------------------------------------------------------
1. "stimon"         stimulus onset           [-0.6;2.3]          [0.1;40]Hz
2. "ica"            stimulus onset           [-0.6;2.3]          [1;40]Hz
3. "fulllength"     cue onset                [-0.6;3.3]          [0.1;40]Hz
3. "cue"            cue onset                [-0.6;1.0]          [0.1;40]Hz



"""

import os
import os.path as op
import pickle
import sys
import numpy as np
import mne
from pathlib import Path
from library import helpers, config


def load_events(subID, epo_part):
    fname_eve = op.join(config.paths['01_prepared-events'], '-'.join([subID, epo_part,'eve.fif']))
    events_ = mne.read_events(fname_eve)
    fname_eve_id = op.join(config.paths['01_prepared-events'], '-'.join([subID, 'event_id.pkl']))
    with open(fname_eve_id, 'rb') as f:
        event_id_ = pickle.load(f)
    fname_bad_epos = op.join(config.paths['01_prepared-events'], '-'.join([subID, 'bad_epos_recording.pkl']))
    with open(fname_bad_epos, 'rb') as f:
        bad_epos_=pickle.load(f)    
    return(events_, event_id_, bad_epos_)


def extract_epochs(raw_data, events, event_id_, tmin, tmax, l_freq, h_freq, baseline=None, bad_epos=None, n_jobs=1):
    # filter the data:
    filtered = raw_data.load_data().filter(l_freq=l_freq,
                                           h_freq=h_freq,
                                           n_jobs=n_jobs,
                                           verbose=False)
    epos_ = mne.Epochs(filtered, 
                        events, 
                        event_id=event_id_, 
                        tmin=tmin, 
                        tmax=tmax, 
                        baseline=baseline,
                        preload=False)
    if bad_epos is not None:
        epos_.drop(bad_epos, 'BADRECORDING')
    return epos_




# parse args:
if len(sys.argv) > 1:
    helpers.print_msg('Running Job Nr. ' + sys.argv[1])
    helpers.print_msg('Study path set to ' + str(path_study))
    job_nr = int(float(sys.argv[1]))
else:
    job_nr = None



## Full procedure:
sub_list = np.setdiff1d(np.arange(1,config.n_subjects_total+1), config.ids_missing_subjects)
sub_list_str = ['VME_S%02d' % sub for sub in sub_list]

if job_nr is not None:
    sub_list_str = [sub_list_str[job_nr]]

## to run a single subject, modify and uncomment one of the following lines:
# sub_list_str = ['VME_S01']


for idx, subID in enumerate(sub_list_str):
    helpers.print_msg('Processing subject ' + subID + '.')
    
    # Get data:
    raw = helpers.load_data(subID + '-prepared',
                            config.paths['01_prepared'],
                            append='-raw',
                            verbose=False)
    events_cue, event_id, bad_epos = load_events(subID, 'cue')
    events_stimon,_,_ = load_events(subID, 'stimon')
    
    event_id_cue    = {key: event_id[key] for key in event_id if event_id[key] in events_cue[:,2]}
    event_id_stimon = {key: event_id[key] for key in event_id if event_id[key] in events_stimon[:,2]}

    epos = dict()
    epos["ica"] = extract_epochs(raw.copy(), 
                              events_stimon, 
                              event_id_stimon,
                              tmin=-0.6,
                              tmax=2.3,
                              l_freq=1,
                              h_freq=40,
                              baseline=None, #baseline corr is bad for ICA
                              bad_epos=None,
                              n_jobs = config.n_jobs)
    
    for l_freq in [0.01, 0.05, 0.1, 0.5]:
    
        epos["stimon"] = extract_epochs(raw.copy(), 
                                  events_stimon, 
                                  event_id_stimon,
                                  tmin=-0.6,
                                  tmax=2.3,
                                  l_freq=l_freq,
                                  h_freq=40,
                                  baseline=None, #(-config.times_dict['bl_dur_erp'], 0),
                                  bad_epos=bad_epos.get('stimon',[]),
                                  n_jobs = config.n_jobs)

        epos["cue"] = extract_epochs(raw.copy(), 
                                  events_cue, 
                                  event_id_cue,
                                  tmin=-0.6,
                                  tmax=1.0,
                                  l_freq=l_freq,
                                  h_freq=40,
                                  baseline=None, #(-config.times_dict['bl_dur_erp'], 0),
                                  bad_epos=bad_epos.get('cue',[]),
                                  n_jobs = config.n_jobs)

        epos["fulllength"] = extract_epochs(raw.copy(), 
                                  events_cue, 
                                  event_id_cue,
                                  tmin=-0.6,
                                  tmax=3.3,
                                  l_freq=l_freq,
                                  h_freq=40,
                                  baseline=None, #(-config.times_dict['bl_dur_erp'], 0),
                                  bad_epos=np.unique([v for k in bad_epos.keys() for v in bad_epos.get(k, [])]),
                                  n_jobs = config.n_jobs)

        for part in ["ica", "stimon", "cue", "fulllength"]:
            helpers.save_data(epos[part],
                              subID + '-' + part,
                              op.join(config.paths["02_epochs"], str(l_freq), part), 
                              '-epo')


    
