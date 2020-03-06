

import os.path as op
import numpy as np
import mne
from library import helpers, config


# Combine subjects:
condition = 'LoadHigh'
cond_str = '-'+condition
sub_list = [1,2,3,5, 6, 7, 8, 9, 10, 13, 16, 17, 18, 20, 22, 23, 24, 26, 27]
all_tfrs = []
for idx, sub in enumerate(sub_list):
    subID = 'VME_S%02d' % sub
    tfr = mne.time_frequency.read_tfrs(op.join(config.path_tfrs, subID + '-PowDiff' + cond_str + '-tfr.fif'))
    all_tfrs.append(tfr[0])  # Insert to the container

glob_tfr = all_tfrs[0].copy()
glob_tfr._data = np.stack([all_tfrs[i]._data for i in range(len(sub_list))]).mean(0)
img = glob_tfr.plot(baseline=None, 
                    picks='all', 
                    mode='mean',
                    vmax = 3e-10, 
                    vmin = -3e-10,
                    tmin=-1.3, tmax=2.7, 
                    title='contra-ipsi' + '  (' + condition + ')', 
                    cmap='RdBu')
ff = 'tfr_' + 'avg_' + cond_str + '.png'
img.savefig(op.join(config.path_tfrs, 'plots', ff))