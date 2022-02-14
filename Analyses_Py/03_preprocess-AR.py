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
sub_list = np.setdiff1d(np.arange(1,config.n_subjects_total+1), config.ids_missing_subjects)

#sub_list = [7, 21, 22, 23, 24, 25, 26, 27] 

sub_list_str = ['VME_S%02d' % sub for sub in sub_list]

if job_nr is not None:
    sub_list_str = [sub_list_str[job_nr]]
for epo_part in ['stimon', 'cue', 'fulllength']: # []: # 
    for subID in sub_list_str:
        fname = '-'.join([subID, epo_part, 'postICA'])
        try:
            path_in = op.join(config.paths['03_preproc-ica'], 'cleaneddata', '0.1', epo_part)
            data_pre = helpers.load_data(fname, path_in, '-epo')
        except FileNotFoundError:
            print(f'No data for {subID}.')
            continue
            
        data_bl = data_pre.copy().apply_baseline((-config.times_dict['bl_dur_erp'], 0)) 
        
        path_save = op.join(config.paths['03_preproc-ar'], '0.1')
        data_post, ar, reject_log = preprocess.clean_with_ar_local(subID,
                                                                   data_bl,
                                                                   epo_part=epo_part,
                                                                   n_jobs=70,
                                                                   save_to_disc=True,
                                                                   ar_path=path_save,
                                                                   rand_state=42)
