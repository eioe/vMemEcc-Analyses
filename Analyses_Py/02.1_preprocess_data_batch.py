"""
=============================
02. Preprocess data
=============================

TODO: Write doc 
"""

import os
import os.path as op
import numpy as np
import csv
import mne
from pathlib import Path
from library import helpers
from datetime import datetime

# define subject:
subsub_list = ['VME_S10', 'VME_S13', 'VME_S16', 'VME_S22']

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


def mark_bad_epos_for_ica(data_):
    data_.plot(scalings=dict(eeg=20e-5), 
                    n_epochs=6, 
                    n_channels=64, 
                    block=True)
    return data_

def write_bads_to_file(subsub_, data_, out_file):
    with open(out_file, 'a') as ff:
        bad_epos = ",".join([str(idx) for idx in np.where(data_.drop_log)[0]])
        bad_chans = ",".join(data_.info['bads'])
        n_epos_remain = str(len(data_))
        timestamp = str(datetime.now())
        ff.write(subsub_ + ";" + bad_epos + ";" + bad_chans + ";" + n_epos_remain + ";" + timestamp + "\n")

def read_bads_from_file(ID, file_):
    with open(file_) as ifile:
        csv_reader = csv.reader(ifile, delimiter=';')
        for row in csv_reader:
            if row[0] == ID:
                bad_epos_ = [int(i) for i in row[1].split(',') if not i == '']
                bad_chans_ = [s for s in row[2].split(',') if not s == '']
                return bad_epos_, bad_chans_
            else:
                continue
    print(f'Subject {ID} not found in {file_}')
        


def get_ica_weights(subID, data_, n_interp_chans_, ica_from_disc = False):
    ### Load ICA data (after comp rejection)?
    if ica_from_disc:
        ica = mne.preprocessing.read_ica(fname=op.join(path_outp_ICA, subID + '-ica.fif.'))
    else:
        ica = mne.preprocessing.ICA(method='infomax', 
                                    fit_params=dict(extended=True), 
                                    max_pca_components = len(data_.info['ch_names']) - 3 - len(data_.info['bads']) - n_interp_chans)
        ica.fit(data_)
        ica.save(fname=op.join(path_outp_ICA, subID + '-ica.fif.'))
    return ica


## Reject components:

# Via correlation w/ EOG channels:
def rej_ica_eog(data_ica_, data_forica_, data_to_clean_):
    """
    Find EOG components, remove them, and apply ICA weights to full data.
    """
    EOGexclude = []
    for ch in ('VEOG', 'HEOG'):
        eog_indices, eog_scores = data_ica_.find_bads_eog(data_forica_, ch_name=ch) #, threshold=2)
        EOGexclude.extend(np.argsort(eog_scores)[-2:])

        data_ica_.plot_scores(eog_scores)

    # Plot marked components:
    data_ica_.plot_components(inst=data_forica_, picks=EOGexclude)
    # Ask user which of the suggested components shall stay in data:
    data_ica_.exclude = EOGexclude
    # and kick out components:
    # data_rejcomp = data_to_clean_.copy()
    data_ica_.apply(data_to_clean_)
    return data_to_clean_

def vis_compare_ica(data_before, data_after, show_data_before=False):
    # visual comparison:
    # old = data_before.plot(scalings=dict(eeg=20e-5), 
    #                 n_epochs=10, 
    #                 n_channels=64), 
    new = data_after.plot(scalings=dict(eeg=20e-5), 
                    n_epochs=10, 
                    n_channels=64, 
                    block=True)


# TODO: make this more secure to avoid overwriting, multiple lines with same ID, ...
def reject_bads(ID, data_, mode, write_results_to_file = True):
    """ reject bad epochs and mark bad channels. 

    Keyword arguments: 
    ID     --  subject ID
    data_  --  epoched data
    mode   --  'manual' (mark in plot) or 'fromfile' (read from file)
    """

    summary_file = op.join(path_outp_rejepo_summaries, 'rej_preica.csv')

    if mode == 'manual':
        data_rejepo_ = mark_bad_epos_for_ica(data_)

        if write_results_to_file:
            # write drop log to file:
            write_bads_to_file(ID, data_rejepo_, summary_file)
    
    elif mode == 'fromfile':
        # read drop log from file:
        bad_epos, bad_chans = read_bads_from_file(ID, summary_file)
        data_rejepo_ = data_.drop(bad_epos)
        data_rejepo_.info['bads'].extend(bad_chans)
        
    else: 
        raise ValueError('Not a valid mode parameter. Use "manual" or "fromfile".')
    
    return data_rejepo_

def interpolate_bad_chans(data_):
    # We don't want to interpolate LO1/2 and IO1/2 as long as we don't have their coordinates. 
    # For now we just drop them.
    data_.drop_channels([ch for ch in ['LO1', 'LO2', 'IO1', 'IO2'] if ch in data_.info['ch_names']])
    n_bads_ = len(data_.info['bads'])
    data_.interpolate_bads()
    return data_, n_bads_


######################################################################################################

for subsub in subsub_list:

    # get BP [1; 40Hz] filtered data to train ICA:
    data_forica = mne.read_epochs(fname=op.join(path_prep_epo, subsub + '-forica-epo.fif'))

    # get BP [0.01; 40Hz] filtered data to apply ICA weights:
    #data_forcda = mne.read_epochs(fname=op.join(path_prep_epo, subsub + '-stimon-epo.fif'))

    data_forica = reject_bads(subsub, data_forica, 'fromfile', write_results_to_file=False)
    ## Skip this for now:
    ###data_forica, n_interp_chans = interpolate_bad_chans(data_forica)
    n_interp_chans = 0

    #data_forcda = reject_bads(subsub, data_forcda, mode='fromfile')

    ## Skip this for now:
    ###data_forcda, _ = interpolate_bad_chans(data_forcda)

    data_ica = get_ica_weights(subsub, data_forica, n_interp_chans, ica_from_disc=False)
    #data_forcda = rej_ica_eog(data_ica, data_forica, data_forcda)

    #vis_compare_ica(data_forcda, data_forcda)
    #helpers.save_data(data_forcda, subsub + '-forcda-postica', path_outp_rejICA, append='-epo')