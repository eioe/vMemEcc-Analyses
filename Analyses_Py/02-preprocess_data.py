"""
=============================
02. Preprocess data
=============================

TODO: Write doc 
"""


import os
import os.path as op
import numpy as np
import mne
from pathlib import Path
from library import helpers
from datetime import datetime


# define dummy subject:
subsub = 'VME_S22'

# set paths:
path_study = Path(os.getcwd()).parents[1] #str(Path(__file__).parents[2])
# note: returns Path object >> cast for string

#TODO: give reasonable names
path_data = os.path.join(path_study, 'Data')
path_inp = os.path.join(path_data, 'DataMNE', 'EEG', '00_raw')
path_outp_ev = op.join(path_data, 'DataMNE', 'EEG', '01_events')
path_prep_epo = op.join(path_data, 'DataMNE', 'EEG', '04_epo')
path_outp_filt = op.join(path_data, 'DataMNE', 'EEG', '03_filt')
path_outp_rejepo = op.join(path_data, 'DataMNE', 'EEG', '05.1_rejepo')
path_outp_rejepo_summaries = op.join(path_data, 'DataMNE', 'EEG', '05.1_rejepo', 'summaries')
path_outp_ICA = op.join(path_data, 'DataMNE', 'EEG', '05.2_ICA')
path_outp_rejICA = op.join(path_data, 'DataMNE', 'EEG', '05.3_rejICA')



for pp in [path_outp_rejepo, path_outp_rejepo_summaries, path_outp_ICA, path_outp_rejICA]:
    if not op.exists(pp):
        os.makedirs(pp)
        print('creating dir: ' + pp)


def mark_bad_epos_for_ica(subID):
    data_epo = mne.read_epochs(fname=op.join(path_prep_epo, subID + '-forica-epo.fif'))
    data_epo.plot(scalings=dict(eeg=20e-5), 
                    n_epochs=6, 
                    n_channels=64, 
                    block=True)
    return data_epo

def write_bads_to_file(subsub_, data_, out_file):
    with open(out_file, 'a') as ff:
        bad_epos = ",".join([str(idx) for idx in np.where(data_.drop_log)[0]])
        bad_chans = ",".join(data_.info['bads'])
        n_epos_remain = str(len(data_))
        timestamp = str(datetime.now())
        ff.write(subsub_ + ";" + bad_epos + ";" + bad_chans + ";" + n_epos_remain + ";" + timestamp + "\n")


def get_ica_weights(subID, ica_from_disc = False, data_epo = None):
    ### Load ICA data (after comp rejection)?
    if ica_from_disc:
        ica = mne.preprocessing.read_ica(fname=op.join(path_outp_ICA, subID + '-ica.fif.'))
    else:
        # make copy for ICA (note: you should better make a copy from raw and filter cont data and re-epoch)
        data_1hz = data_epo.copy()
        #FIXME: You should filter cont data
        data_1hz.load_data().filter(l_freq=1., h_freq=None)
        ica = mne.preprocessing.ICA(method='infomax', fit_params=dict(extended=True))
        ica.fit(data_1hz)
        ica.save(fname=op.join(path_outp_ICA, subID + '-ica.fif.'))
    return ica


## Reject components:

# Via correlation w/ EOG channels:
def rej_ica_eog(_data_ica, _data_rejepo):
    EOGexclude = []
    for ch in ('VEOG', 'HEOG'):
        eog_indices, eog_scores = _data_ica.find_bads_eog(_data_rejepo, ch_name=ch, threshold=2)
        EOGexclude.extend(eog_indices)
        _data_ica.plot_scores(eog_scores)

    # Plot marked components:
    _data_ica.plot_components(inst=_data_rejepo, picks=EOGexclude)
    # Ask user which of the suggested components shall stay in data:
    _data_ica.exclude = EOGexclude
    # and kick out components:
    data_rejcomp = _data_rejepo.copy()
    _data_ica.apply(data_rejcomp)
    return data_rejcomp

def vis_compare_ica(data_before, data_after, show_data_before=False):
    # visual comparison:
    # old = data_before.plot(scalings=dict(eeg=20e-5), 
    #                 n_epochs=10, 
    #                 n_channels=64), 
    new = data_after.plot(scalings=dict(eeg=20e-5), 
                    n_epochs=10, 
                    n_channels=64, 
                    block=True)


######################################################################################################

data_rejepo = mark_bad_epos_for_ica(subsub)

# write drop log to file:
summary_file = op.join(path_outp_rejepo_summaries, 'rej_preica.csv')
write_bads_to_file(subsub, data_rejepo, summary_file)


"""

# read drop log from file:
with open(summary_file) as ifile:
    csv_reader = csv.reader(ifile, delimiter=';')
    for row in csv_reader:
        if row[0] == subsub:
            badd_epos = row[1]
            badd_chans = row[2]
        else:
            continue
    


data_rejepo.apply_baseline((-.3,-0.0))
helpers.save_data(data_rejepo, subsub + '-rejepo', path_outp_rejepo, '-epo')

data_rejepo = helpers.load_data(subsub + '-rejepo', path_outp_rejepo, '-epo')

ica = get_ica_weights(subsub, ica_from_disc=False, data_epo=data_rejepo)


data_rejcomp = rej_ica_eog(ica, data_rejepo)
vis_compare_ica(data_rejepo, data_rejcomp)
helpers.save_data(data_rejcomp, subsub + '-rejcomp', path_outp_rejICA, append='-epo')

 """