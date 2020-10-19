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


def get_tfrs_list(sub, part_epo, pwr_style, picks='eeg'):
    subID = 'VME_S%02d' % sub
    epos_ = helpers.load_data(subID, config.path_epos_sorted + '/' +
                              part_epo + '/collapsed', '-epo')
    
    # Shift time, so that 0 == Stimulus Onset:
    epos_ = epos_.shift_time(-config.times_dict['cue_dur'])
    
    if pwr_style == 'induced':
        epos_ = epos_.subtract_evoked()

    #  picks = config.chans_CDA_all
    tfrs_ = get_tfr(epos_, picks=picks, average=False)

    event_dict = helpers.get_event_dict(epos_.event_id)

    sub_tfrs = list()

    for load in ['LoadLow', 'LoadHigh']:
        avgtfrs_load = get_tfr(epos_[event_dict[load]], picks=picks, 
                               average=True)
        avgtfrs_load.comment = load
        sub_tfrs.append(avgtfrs_load)
        for ecc in ['EccS', 'EccM', 'EccL']:
            if load == 'LoadLow':  # we don't want to do this twice
                avgtfrs_ecc = get_tfr(epos_[event_dict[ecc]], picks=picks, 
                                      average=True)#tfrs_[event_dict[ecc]].copy().average()
                avgtfrs_ecc.comment = ecc
                sub_tfrs.append(avgtfrs_ecc)
            # Interaction:
            avgtfrs_interac = get_tfr(epos_[event_dict[load]][event_dict[ecc]],
                                      picks=picks, average=True)
            avgtfrs_interac.comment = load+ecc
            sub_tfrs.append(avgtfrs_interac)
    avgtfrs_all = get_tfr(epos_, picks=picks, average=True)
    avgtfrs_all.comment = 'all'
    sub_tfrs.append(avgtfrs_all)

    fpath = op.join(config.path_tfrs, pwr_style, 'tfr_lists', part_epo)
    helpers.chkmk_dir(fpath)
    fname = op.join(fpath, subID + '-collapsed-avgTFRs-tfr.h5')
    mne.time_frequency.write_tfrs(fname, sub_tfrs, overwrite=True)
    return(sub_tfrs)



def get_tfr(epos, picks='all', average=True, freqs=None):
    if freqs is None:
        freqs = np.concatenate([np.arange(6, 26, 1)])  # , np.arange(16,30,2)])
    n_cycles = freqs / 2.  # different number of cycle per frequency
    power = mne.time_frequency.tfr_morlet(epos, picks=picks, freqs=freqs,
                                          n_cycles=n_cycles, use_fft=True,
                                          return_itc=False, average=average,
                                          decim=1, n_jobs=-2)
    return power



    fpath = op.join(config.path_tfrs, pwr_style, 'tfr_lists', part_epo)
    helpers.chkmk_dir(fpath)
    fname = op.join(fpath, subID + '-collapsed-avgTFRs-tfr.h5')
    mne.time_frequency.write_tfrs(fname, sub_tfrs, overwrite=True)


# TODO: Fix path
sub_list = np.setdiff1d(np.arange(1, 28), config.ids_missing_subjects +
                        config.ids_excluded_subjects)


# Mirror structure in "03.3-evaluate_CDA"

part_epo = 'fulllength'
pwr_style = 'induced'  # 'evoked' # 

parallel, run_func, _ = parallel_func(get_tfrs_list,
                                      n_jobs=config.10)
parallel(run_func(subject) for subject in sub_list)

_ = get_tfrs_list(sub, part_epo, pwr_style)
