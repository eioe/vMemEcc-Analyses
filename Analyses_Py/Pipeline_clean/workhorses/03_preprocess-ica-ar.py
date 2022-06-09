"""
===========
03_preprocess-ica-ar
===========
Batch file to precalculate the ICA weights for the notebook `03_preprocess-ICA``. 
You can call it from a batch\slurm file with job arrays.

02/2022 Felix Klotzsche
"""



from os import path as op
import sys
import numpy as np
import autoreject
import mne
from library import config, helpers, preprocess

# parse args:
if (len(sys.argv) > 1):
    helpers.print_msg('Running Job Nr. ' + sys.argv[1])
    job_nr = int(sys.argv[1])
else:
    job_nr = None

## Full procedure:
#sub_list = np.setdiff1d(np.arange(1,config.n_subjects_total+1), config.ids_missing_subjects)
sub_list = np.setdiff1d(np.arange(1,18+1), config.ids_missing_subjects)
sub_list_str = ['VME_S%02d' % sub for sub in sub_list]

if job_nr is not None:
    sub_list_str = [sub_list_str[job_nr]]

for subID in sub_list_str:    
    data_raw = helpers.load_data(subID + '-prepared',
                                        config.paths['01_prepared'],
                                        append='-raw').load_data()
    data_forICA = helpers.load_data(subID + '-ica',
                                        op.join(config.paths['02_epochs'], '0.1', 'ica')
                                        append='-epo')

    # clean it with autoreject local to remove bad epochs for better ICA fit:
    data_forAR = data_forICA.copy().apply_baseline((-config.times_dict['dur_bl_erp'], 0)) 
    # AR does not perform well on non-baseline corrected data

    _, ar, reject_log = preprocess.clean_with_ar_local(subID,
                                                       data_forAR,
                                                       n_jobs=config.n_jobs, #adapt this to your setup
                                                       ar_from_disc=False,
                                                       save_to_disc=True,
                                                       ar_path=config.paths['03_preproc-ica-ar'],
                                                       rand_state=42)
        # Get ICA weights
    ica = preprocess.get_ica_weights(subID,
                                      data_forICA[~reject_log.bad_epochs],
                                      picks=None,
                                      reject=None,
                                      method='picard',
                                      fit_params=None,
                                      ica_from_disc=False,
                                      save_to_disc=True
                                      ica_path=config.paths['03_preproc-ica'])
