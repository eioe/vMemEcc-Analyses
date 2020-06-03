"""
=============================
02. Preprocess data
=============================

TODO: Write doc 
"""

import os
import os.path as op
import sys
import numpy as np
import csv
import mne
import autoreject
from pathlib import Path
from library import helpers, config
from datetime import datetime

# set paths:
path_study = Path(os.getcwd()) #str(Path(__file__).parents[2])
# path_study = os.path.join(path_study, 'Experiments', 'vMemEcc')
# note: returns Path object >> cast for string

#TODO: get from config
path_data = os.path.join(path_study, 'Data')
path_inp = os.path.join(path_data, 'DataMNE', 'EEG', '00_raw')
path_outp_ev = op.join(path_data, 'DataMNE', 'EEG', '01_events')
path_prep_epo = op.join(path_data, 'DataMNE', 'EEG', '04_epo')
path_outp_filt = op.join(path_data, 'DataMNE', 'EEG', '03_filt')
path_outp_rejepo = op.join(path_data, 'DataMNE', 'EEG', '05.1_rejepo')
path_outp_rejepo_summaries = op.join(path_data, 'DataMNE', 'EEG', '05.1_rejepo', 'summaries')
path_outp_ICA = op.join(path_data, 'DataMNE', 'EEG', '05.2_ICA')
path_outp_rejICA = op.join(path_data, 'DataMNE', 'EEG', '05.3_rejICA')

# parse args:
helpers.print_msg('Running Job Nr. ' + sys.argv[1])
helpers.print_msg('Study path set to ' + str(path_study))
job_nr = int(float(sys.argv[1]))


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
        

def get_rejthresh_for_ica(subID, data_, out_file=None): 
    thresh = autoreject.get_rejection_threshold(data_, ch_types='eeg')
    data_clean = data_.copy().drop_bad(reject=thresh)
    n_epos_rejected = len(data_) - len(data_clean)
    if not out_file is None:
        with open(out_file, 'a') as ff:
            ff.write(subID + ';' + str(thresh) + ';' + str(n_epos_rejected) + '\n')
        
    return thresh

def clean_with_ar_local(data_):
    picks = mne.pick_types(data_.info, meg=False, eeg=True, stim=False,
                       eog=False)
    ar = autoreject.AutoReject(n_interpolate=np.array([2,8,16]), 
                               consensus= np.linspace(0.3, 1.0, 8),
                               picks=picks, 
                               n_jobs=config.n_jobs,
                               random_state = 42,
                               verbose='tqdm')
    epo_clean, reject_log = ar.fit_transform(data_, return_log=True)
    return epo_clean, ar, reject_log

def get_ica_weights(subID, data_, ica_from_disc = False, reject=None):
    ### Load ICA data (after comp rejection)?
    if ica_from_disc:
        ica = mne.preprocessing.read_ica(fname=op.join(path_outp_ICA, subID + '-ica.fif.'))
    else:
        data_.drop_bad(reject=reject)
        ica = mne.preprocessing.ICA(method='infomax', 
                                    fit_params=dict(extended=True))
        ica.fit(data_)
        ica.save(fname=op.join(path_outp_ICA, subID + '-ica.fif.'))
    return ica


## Reject components:

# Via correlation w/ EOG channels:
def rej_ica_eog(subID, data_ica_, data_forica_, data_to_clean_):
    """
    Find EOG components, remove them, and apply ICA weights to full data.
    """
    EOGexclude = []
    
    eog_indices, eog_scores = data_ica_.find_bads_eog(data_forica_)  #, threshold=2)
        #EOGexclude.extend(np.argsort(eog_scores)[-2:])

    #    data_ica_.plot_scores(eog_scores)

    ## Plot marked components:
    # data_ica_.plot_components(inst=data_forica_, picks=EOGexclude)
    
    data_ica_.exclude = eog_indices
    # overwrite on disk with updated version:
    data_ica_.save(fname=op.join(path_outp_ICA, subID + '-ica.fif.'))
    # and kick out components:
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

def save_rejlog(rejlog, fname):
    np.savetxt(fname, rejlog.labels, header=','.join(rejlog_stimon.ch_names), 
                                     delimiter=',',
                                     comments='',
                                     fmt='%1.0f')
                                    
    


######################################################################################################


# define subject:
#sub_list = [10, 13]# , 'VME_S13', 'VME_S16', 'VME_S22']
sub_list = np.setdiff1d(np.arange(1,28), config.ids_missing_subjects)
if job_nr > len(sub_list)-1: 
    helpers.print_msg('All jobs taken.')
    exit()

sub_list = np.array([sub_list[job_nr]])

for subsub in sub_list:
    subsub = 'VME_S%02d' % subsub
    helpers.print_msg('Starting with subject: ' + subsub)

    # get BP [1; 40Hz] filtered data to train ICA:
    data_forica = mne.read_epochs(fname=op.join(path_prep_epo, subsub + '-forica-epo.fif'))
    data_stimon = mne.read_epochs(fname=op.join(path_prep_epo, subsub + '-stimon-epo.fif'))
    data_cue = mne.read_epochs(fname=op.join(path_prep_epo, subsub + '-cue-epo.fif'))
    data_fulllength = mne.read_epochs(fname=op.join(path_prep_epo, subsub + '-fulllength-epo.fif'))
    
    # clean it with autoreject local:
    data_forica_c, _, _ = clean_with_ar_local(data_forica)
    
    # fit ICA to cleaned data:
    data_ica = get_ica_weights(subsub, data_forica_c, ica_from_disc=False)

    # Apply baseline:
    # data_forica.crop(tmin=-0.6, tmax=2.3)#.apply_baseline((-0.4,0))
    data_stimon.apply_baseline((-0.2,0))
    data_cue.apply_baseline((-0.2,0))
    data_fulllength.apply_baseline((-0.2,0))

    # remove eog components and project to actual data:
    data_stimon = rej_ica_eog(subsub, data_ica, data_forica_c, data_stimon)
    data_cue = rej_ica_eog(subsub, data_ica, data_forica_c, data_cue)
    data_fulllength = rej_ica_eog(subsub, data_ica, data_forica_c, data_fulllength)

    # clean actual data with autoreject local:
    data_stimon_c, ar_stimon, rejlog_stimon = clean_with_ar_local(data_stimon)
    data_cue_c, ar_cue, rejlog_cue = clean_with_ar_local(data_cue)
    data_fulllength_c, ar_fulllength, rejlog_fulllength = clean_with_ar_local(data_fulllength)

    # write n of rejected epos to file: 
    rej_epos = []
    for ds in [data_stimon_c, data_cue_c, data_fulllength_c]: 
        n_rej_epo = np.sum([1 for e in ds.drop_log if e == ['AUTOREJECT']])
        rej_epos.append(str(n_rej_epo))
    fname_ar_rejsummary = op.join(config.path_autoreject_logs, 'ar_reject_summary.csv')
    with open(fname_ar_rejsummary, 'a+') as f:
        f.write(subsub + ';' + ';'.join(rej_epos))


    # Save results: 
    helpers.save_data(data_forica_c, 
                      subsub + '-forica-postar', 
                      path_outp_rejepo, 
                      append = '-epo')

    helpers.save_data(data_stimon_c, 
                      subsub + '-stimon-postica', 
                      config.path_postICA, 
                      append = '-epo')
    helpers.save_data(ar_stimon, 
                      subsub + '-stimon-arlocal', 
                      config.path_autoreject)
    helpers.save_data(data_cue_c, 
                      subsub + '-cue-postica', 
                      config.path_postICA, 
                      append = '-epo')
    helpers.save_data(ar_cue, 
                      subsub + '-cue-arlocal', 
                      config.path_autoreject)
    helpers.save_data(data_fulllength_c, 
                      subsub + '-fulllength-postica', 
                      config.path_postICA, 
                      append = '-epo')
    helpers.save_data(ar_fulllength, 
                      subsub + '-fulllength-arlocal', 
                      config.path_autoreject)
    # save autoreject logs: 
    fname = os.path.join(config.path_autoreject_logs, subsub + 'stimon-rejlog.csv')
    save_rejlog(rejlog_stimon, fname)
    fname = os.path.join(config.path_autoreject_logs, subsub + 'cue-rejlog.csv')
    save_rejlog(rejlog_cue, fname)
    fname = os.path.join(config.path_autoreject_logs, subsub + 'fulllength-rejlog.csv')
    save_rejlog(rejlog_fulllength, fname)
    

    # get BP [0.01; 40Hz] filtered data to apply ICA weights:
    #data_forcda = mne.read_epochs(fname=op.join(path_prep_epo, subsub + '-stimon-epo.fif'))

    #data_forica = reject_bads(subsub, data_forica, 'fromfile', write_results_to_file=False)


    #data_forcda = reject_bads(subsub, data_forcda, mode='fromfile')

    ## Skip this for now:
    ###data_forcda, _ = interpolate_bad_chans(data_forcda)

    #

    #vis_compare_ica(data_forcda, data_forcda)
    #helpers.save_data(data_forcda, subsub + '-forcda-postica', path_outp_rejICA, append='-epo')

