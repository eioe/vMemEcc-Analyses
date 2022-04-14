# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: mne
#     language: python
#     name: mne
# ---

import os
import os.path as op
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import mne
from library import helpers, config

path_study = Path(os.getcwd())
# parse args:
helpers.print_msg('Running Job Nr. ' + sys.argv[1])
helpers.print_msg('Study path set to ' + str(path_study))
job_nr = int(float(sys.argv[1]))


# +
def get_tfrs_list(sub, part_epo, pwr_style, picks='eeg'):
    subID = 'VME_S%02d' % sub
    fpath = op.join(config.paths['03_preproc-pooled'], part_epo, "collapsed")
    epos_ = helpers.load_data(f"{subID}-{part_epo}-collapsed", fpath, '-epo')
    
    if (part_epo in ['cue', 'fulllength']):
        # Shift time, so that 0 == Stimulus Onset:
        epos_ = epos_.shift_time(-config.times_dict['cue_dur'])
        
    if pwr_style == 'induced':
        epos_ = epos_.subtract_evoked()

    #  picks = config.chans_CDA_all
    #tfrs_ = get_tfr(epos_, picks=picks, average=False)

    event_dict = helpers.get_event_dict(epos_.event_id)

    sub_tfrs = list()

    for load in ['LoadLow', 'LoadHigh']:
        avgtfrs_load = get_tfr(epos_[event_dict[load]], picks=picks, 
                               average=False)
        avgtfrs_load.comment = load
        save_tfr(subID, avgtfrs_load, load, pwr_style, part_epo)
        sub_tfrs.append(avgtfrs_load)
        # Save averaged version:
        save_tfr(subID, avgtfrs_load.average(), load, pwr_style, part_epo, averaged=True)
        for ecc in ['EccS', 'EccM', 'EccL']:
            if load == 'LoadLow':  # we don't want to do this twice
                avgtfrs_ecc = get_tfr(epos_[event_dict[ecc]], picks=picks, 
                                      average=False)#tfrs_[event_dict[ecc]].copy().average()
                avgtfrs_ecc.comment = ecc
                save_tfr(subID, avgtfrs_ecc, ecc, pwr_style, part_epo)
                sub_tfrs.append(avgtfrs_ecc)
                save_tfr(subID, avgtfrs_ecc.average(), ecc, pwr_style, part_epo, averaged=True)
            # Interaction:
            avgtfrs_interac = get_tfr(epos_[event_dict[load]][event_dict[ecc]],
                                      picks=picks, average=False)
            avgtfrs_interac.comment = load+ecc
            save_tfr(subID, avgtfrs_interac, load+ecc, pwr_style, part_epo)
            sub_tfrs.append(avgtfrs_interac)
            save_tfr(subID, avgtfrs_interac.average(), load+ecc, pwr_style, part_epo, averaged=True)
    avgtfrs_all = get_tfr(epos_, picks=picks, average=False)
    avgtfrs_all.comment = 'all'
    save_tfr(subID, avgtfrs_all, 'all', pwr_style, part_epo)
    sub_tfrs.append(avgtfrs_all)
    save_tfr(subID, avgtfrs_all.average(), 'all', pwr_style, part_epo, averaged=True)

    fpath = op.join(config.paths['05_tfrs'], pwr_style, 'tfr_lists', part_epo)
    helpers.chkmk_dir(fpath)
    fname = op.join(fpath, subID + '-collapsed-singletrialTFRs-tfr.h5')
    mne.time_frequency.write_tfrs(fname, sub_tfrs, overwrite=True)
    fname = op.join(fpath, subID + '-collapsed-avgTFRs-tfr.h5')
    mne.time_frequency.write_tfrs(fname, [t.average() for t in sub_tfrs], overwrite=True)
    return(sub_tfrs)


def save_tfr(subID, sub_tfrs, condition, pwr_style='induced', part_epo='fulllength', averaged=False):
    fpath = op.join(config.paths['05_tfrs'], pwr_style, 'tfr_lists', part_epo, condition)
    helpers.chkmk_dir(fpath)
    if averaged:
        fname = op.join(fpath, subID + '-collapsed-avgTFRs-tfr.h5')
    else:
        fname = op.join(fpath, subID + '-collapsed-singletrialTFRs-tfr.h5')  
    
    mne.time_frequency.write_tfrs(fname, sub_tfrs, overwrite=True)


def get_tfr(epos, picks='all', average=True, freqs=None):
    if freqs is None:
        freqs = np.concatenate([np.arange(6, 26, 1)])  # , np.arange(16,30,2)])
    n_cycles = freqs / 2.  # different number of cycle per frequency
    power = mne.time_frequency.tfr_morlet(epos,
                                          picks=picks,
                                          freqs=freqs,
                                          n_cycles=n_cycles,
                                          use_fft=True,
                                          return_itc=False,
                                          average=average,
                                          decim=1,
                                          n_jobs=-2)
    return power

# +
    # fpath = op.join(config.path_tfrs, pwr_style, 'tfr_lists', part_epo)
    # helpers.chkmk_dir(fpath)
    # fname = op.join(fpath, subID + '-collapsed-avgTFRs-tfr.h5')
    # mne.time_frequency.write_tfrs(fname, sub_tfrs, overwrite=True)


sub_list = np.setdiff1d(np.arange(1, 28), config.ids_missing_subjects +
                        config.ids_excluded_subjects)

part_epo = 'fulllength'
pwr_style = 'induced'  

job_nr = 0
#sub = sub_list[job_nr]
for sub in sub_list:
  _ = get_tfrs_list(sub, part_epo, pwr_style)
# -


