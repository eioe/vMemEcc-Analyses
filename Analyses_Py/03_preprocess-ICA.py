
from os import path as op
import sys
import pandas as pd
import numpy as np
import mne
from mne.preprocessing import create_eog_epochs
import autoreject

from library import config, helpers

def clean_with_ar_local(data_, n_jobs=config.n_jobs):
    picks = mne.pick_types(data_.info, meg=False, eeg=True, stim=False,
                       eog=False)
    ar = autoreject.AutoReject(n_interpolate=np.array([2,8,16]), 
                               consensus= np.linspace(0.3, 1.0, 8),
                               picks=picks, 
                               n_jobs=n_jobs,
                               random_state = 42,
                               verbose='tqdm')
    epo_clean, reject_log = ar.fit_transform(data_, return_log=True)
    return epo_clean, ar, reject_log


def get_ica_weights(subID, data_,
                    ica_from_disc=False,
                    reject=None,
                    method='picard',
                    fit_params=None,
                    picks=None,
                    save_to_disc=True):
    ### Load ICA data (after comp rejection)?
    if ica_from_disc:
        ica = mne.preprocessing.read_ica(fname=op.join(config.paths['03_preproc-ica'], subID + '-ica.fif'))
    else:
        data_.drop_bad(reject=reject)
        ica = mne.preprocessing.ICA(method=method, 
                                    fit_params=fit_params)
        ica.fit(data_,
                picks=picks)
        
        if save_to_disc:
            ica.save(fname=op.join(config.paths['03_preproc-ica'], subID + '-ica.fif'),
                     overwrite=True)
    return ica


## Reject components:
# Via correlation w/ EOG channels:
def rej_ica_eog(subID, data_ica_, data_raw_, data_forica_, data_to_clean_):
    """
    Find EOG components, remove them, and apply ICA weights to full data.
    """
    EOGexclude = []
    
    epochs_eog    = mne.preprocessing.create_eog_epochs(raw = data_raw_, ch_name = 'VEOG')
    eog_indices, eog_scores = data_ica_.find_bads_eog(data_forica_)  #, threshold=2)
        #EOGexclude.extend(np.argsort(eog_scores)[-2:])

    #    data_ica_.plot_scores(eog_scores)

    ## Plot marked components:
    # data_ica_.plot_components(inst=data_forica_, picks=EOGexclude)
    
    data_ica_.exclude = eog_indices[:2]
    # overwrite on disk with updated version:
    data_ica_.save(fname=op.join(path_outp_ICA, subID + '-ica.fif.'))
    # and kick out components:
    data_clean_ = data_ica_.apply(data_to_clean_.copy())
    return data_clean_

def vis_compare_ica(data_before, data_after, show_data_before=False):
    # visual comparison:
    picks=['eeg','eog']
    order = pd.unique(['VEOG', 'HEOG', 'Fp1', 'F4', 'Cz', 'PO7', 'PO8', 'O1', 'O2'] + data_before.ch_names)
    if show_data_before:
        old = data_before.copy().\
        apply_baseline((None,None)).\
        reorder_channels(order).\
        plot(scalings=dict(eeg=50e-6),
                               n_epochs=10,
                               n_channels=15,
                               picks=picks) 
    new = data_after.copy().\
    apply_baseline((None,None)).\
    reorder_channels(order).\
    plot(scalings=dict(eeg=50e-6),
                          n_epochs=10,
                          n_channels=15,
                          picks=picks,
                          block=False)


# %matplotlib qt

# parse args:
if (len(sys.argv) > 1):
    helpers.print_msg('Running Job Nr. ' + sys.argv[1])
    job_nr = int(sys.argv[1])
else:
    job_nr = None

## Full procedure:
sub_list = np.setdiff1d(np.arange(1,config.n_subjects_total+1), config.ids_missing_subjects)
sub_list_str = ['VME_S%02d' % sub for sub in sub_list]

if job_nr is not None:
    sub_list_str = [sub_list_str[job_nr]]

## to run a single subject, modify and uncomment one of the following lines:
# sub_list_str = ['VME_S20']


for subID in sub_list_str:
    # Load data
    data_raw = helpers.load_data(subID + '-prepared',
                                        config.paths['01_prepared'],
                                        append='-raw').load_data()
    data_forICA = helpers.load_data(subID + '-ica',
                                        config.paths['02_epochs'],
                                        append='-epo')
    
    # clean it with autoreject local:
    data_forAR = data_forICA.copy().apply_baseline((-0.4,0)) # AR does not perform well on non-baseline corrected data
    _, ar, reject_log = clean_with_ar_local(data_forAR)
    

    # Save results of AR to disk:
    rejlog_df = pd.DataFrame(reject_log.labels,
                             columns=reject_log.ch_names,
                             dtype=int)
    rejlog_df['badEpoch'] = reject_log.bad_epochs
    fname_rejlog = op.join(config.paths['03_preproc-ica-ar'], subID + '-rejectlog-preICA.csv')
    rejlog_df.to_csv(fname_rejlog, float_format="%i")
    fname_ar = op.join(config.paths['03_preproc-ica-ar'], subID + '-ar-preICA.csv')
    ar.save(fname_ar, overwrite=True)
    
    # Calculate ICA after removing bad epochs
    picks = ['eeg', 'eog']
    data_ica = get_ica_weights(subID,
                               data_forICA[~reject_log.bad_epochs],
                               picks=picks,
                               ica_from_disc=False,
                               save_to_disc=True)