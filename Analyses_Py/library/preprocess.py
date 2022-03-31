from os import path as op
import sys
import pandas as pd
import numpy as np
import mne
from mne.preprocessing import create_eog_epochs
import autoreject

from library import config, helpers


def clean_with_ar_local(subID,
                        data_,
                        epo_part='',
                        n_jobs=config.n_jobs,
                        ar_from_disc=False,
                        save_to_disc=True,
                        ar_path=None,
                        rand_state=42):
    
    if ar_path is None:
        req_path = [p for p in [ar_from_disc, save_to_disc] if p]
        if len(req_path) > 0:
            message = 'If you want to read from or write to disk, you need to provide the according path.' + \
                      '(Argument "ar_path")'
            raise ValueError(message) 
    else:
        if config.paths['03_preproc-ica-ar'] in ar_path:
            append = '-preICA-ar'
        elif config.paths['03_preproc-ar'] in ar_path:
            append = f'-{epo_part}-postICA-ar'
        else:
            message = f'I only can write or read AR files in these folders:\n' + \
                      config.paths['03_preproc-ica-ar'] + '\n' + \
                      config.paths['03_preproc-ar']
            raise ValueError(message)
    
    if ar_from_disc:
        fname_ar = op.join(ar_path, f'{subID}{append}.fif')
        ar = autoreject.read_auto_reject(fname_ar)
        epo_clean, reject_log = ar.transform(data_, return_log=True)
        
    else:
        picks = mne.pick_types(data_.info, meg=False, eeg=True, stim=False,
                               eog=False)
        ar = autoreject.AutoReject(n_interpolate=np.array([2,4,8,16]), 
                                   consensus= np.linspace(0.1, 1.0, 11),
                                   picks=picks, 
                                   n_jobs=n_jobs,
                                   random_state = rand_state,
                                   verbose='tqdm')
        epo_clean, reject_log = ar.fit_transform(data_, return_log=True)
    
    if save_to_disc:
        # Save results of AR to disk:
        fpath_ar = op.join(ar_path, epo_part)
        fname_ar = op.join(fpath_ar, f'{subID}{append}.fif')
        helpers.chkmk_dir(fpath_ar)
        ar.save(fname_ar, overwrite=True)
        # externally readable version of reject log
        rejlog_df = pd.DataFrame(reject_log.labels,
                                 columns=reject_log.ch_names,
                                 dtype=int)
        rejlog_df['badEpoch'] = reject_log.bad_epochs
        fpath_rejlog = op.join(ar_path, epo_part, 'rejectlogs')
        fname_rejlog = op.join(fpath_rejlog, f'{subID}{append}-rejectlog.csv')
        helpers.chkmk_dir(fpath_rejlog)
        rejlog_df.to_csv(fname_rejlog, float_format="%i")
        if 'postICA' in append:
            fpath_cleaneddata = op.join(ar_path, epo_part, 'cleaneddata')
            helpers.save_data(epo_clean,
                              f'{subID}-{epo_part}-postAR',
                              fpath_cleaneddata,
                              append='-epo')


    return epo_clean, ar, reject_log


def get_ica_weights(subID,
                    data_=None,
                    picks=None,
                    reject=None,
                    method='picard',
                    fit_params=None,
                    ica_from_disc=False,
                    save_to_disc=True,
                    ica_path=None):
    ### Load ICA data
    if ica_from_disc:
        ica = mne.preprocessing.read_ica(fname=op.join(ica_path, subID + '-ica.fif'))
    else:
        data_.drop_bad(reject=reject)
        ica = mne.preprocessing.ICA(method=method, 
                                    fit_params=fit_params)
        ica.fit(data_,
                picks=picks)
        
        if save_to_disc:
            ica.save(fname=op.join(ica_path, subID + '-ica.fif'),
                     overwrite=True)
    return ica
