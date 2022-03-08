from os import path as op
import sys
import numpy as np
import autoreject
import mne
from library import config, helpers, preprocess


def norm_vec(x):
    return x / np.sqrt(np.sum(x**2))

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
        
        ars = []
        reject_logs = []
        rand_ints = [30,7,19,88,307,198,8,3,0,71988]
        for rs in rand_ints:
            data_post, ar, reject_log = preprocess.clean_with_ar_local(subID,
                                                                       data_bl,
                                                                       epo_part=epo_part,
                                                                       n_jobs=70,
                                                                       save_to_disc=False,
                                                                       rand_state=rs)
            ars.append(ar)
            reject_logs.append(reject_log)
        
        all_badepos = np.stack([rl.bad_epochs for rl in reject_logs])
        avg_badepos = all_badepos.mean(axis=0)

        # sims = [np.dot(avg_rl.flatten(), rl.flatten()) for rl in all_rls]
        sims = [np.dot(norm_vec(avg_badepos), norm_vec(be)) for be in all_badepos]

        idx_max = np.argmax(sims)
        
        path_save = op.join(config.paths['03_preproc-ar'], '0.1', 'robust')
        helpers.chkmk_dir(path_save)
        data_post, ar, reject_log = preprocess.clean_with_ar_local(subID,
                                                                   data_bl,
                                                                   epo_part=epo_part,
                                                                   n_jobs=70,
                                                                   save_to_disc=True,
                                                                   ar_path=path_save,
                                                                   rand_state=rand_ints[idx_max])

        file_diag = op.join(path_save, epo_part, 'info.txt')
        n_bad_epos = [sum(rl.bad_epochs) for rl in reject_logs]
        n_epos_min = np.min(n_bad_epos)
        n_epos_max = np.max(n_bad_epos)
        with open(file_diag, 'a+') as f:
            f.write(f'{subID};{n_epos_min};{n_epos_max};{n_bad_epos};{n_bad_epos[idx_max]}\n')
        
        
        

        
