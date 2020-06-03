import os
from os import path as op
import numpy as np
import pandas as pd

import mne

from library import helpers, config


sub_list = np.setdiff1d(np.arange(1,27), config.ids_missing_subjects)
#sub_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27]

# Initialize summary DF: 
remaining_epos = pd.DataFrame(columns = ['subID', 'stimon', 'cue', 'fulllength'])

for sub_nr in sub_list:
    subID = 'VME_S%02d' % sub_nr
    # subID = 'VME_S02'

    # read in epochs which shall be rejected: 
    fname = op.join(config.path_reject_epos_extern, subID + '-rejTrials-ET.csv')
    df_rej_epos_ext = pd.read_csv(fname)
    rej_epos_ext = np.array(df_rej_epos_ext).flatten()
    # subtract 1 to account for 0-indexing in Python: 
    rej_epos_ext -= 1

    # Init dict to collect ns of remaining epos: 
    rem_epos_dict = dict(subID = subID)

    for epo_part in ['cue', 'stimon', 'fulllength']:

        # Load data:
        data = helpers.load_data(subID + '-' + epo_part + '-postica', config.path_postICA, '-epo')

        # kick epos:
        rej_idx = np.isin(data.selection, rej_epos_ext)
        data.drop(rej_idx)

        rem_epos_dict[epo_part] = len(data)

        helpers.save_data(data, subID + '-' + epo_part + '-postica-rejepo', config.path_rejepo)
    
    remaining_epos = remaining_epos.append(rem_epos_dict, verify_integrity=True, ignore_index=True)

# Save overview DF:
fname = op.join(config.path_rejepo_summaries, 'remaining_epos_per_sub.csv')
remaining_epos.to_csv(fname)
