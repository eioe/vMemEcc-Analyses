
import os
import os.path as op
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mne
from pathlib import Path
from library import helpers, config

subsub = 'VME_S01'

eposIpsi = helpers.load_data(subsub, config.path_epos_sorted + '/RoiIpsi' , '-epo')
eposContra = helpers.load_data(subsub, config.path_epos_sorted + '/RoiContra' , '-epo')


freqs = np.logspace(*np.log10([6, 35]), num=20)
n_cycles = freqs / 2.  # different number of cycle per frequency
powerI = mne.time_frequency.tfr_morlet(eposIpsi, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=False, decim=3, n_jobs=1)
powerC = mne.time_frequency.tfr_morlet(eposContra, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=False, decim=3, n_jobs=1)

powDiff = powerC.copy()
powDiff._data = powerC.reorder_channels(powerI.info['ch_names'])._data - powerI._data

# powSum = powerC.copy()
# powSum._data = powerC.reorder_channels(powerI.info['ch_names'])._data + powerI._data

# powNorm = powerC.copy()
# powNorm._data = powDiff._data / powSum._data


img = powDiff.plot(baseline=None, picks='all', mode='logratio', tmin=-1.3, tmax=2, title='contra-ipsi', cmap='RdBu')
ff = 'tfr_' + subsub + '.png'
# img.savefig(op.join(config.path_tfrs, 'Plots', ff))

# save to file:
helpers.save_data(powDiff, subsub + '-PowDiff', config.path_tfrs, append='-tfr')

# Combine subjects:

sub_list = [1,2] #3, 7, 22]
all_tfrs = []
for idx, sub in enumerate(sub_list):
    subID = 'VME_S%02d' % sub
    tfr = mne.time_frequency.read_tfrs(op.join(config.path_tfrs, subID + '-PowDiff-tfr.fif'))
    all_tfrs.append(tfr[0])  # Insert to the container

glob_tfr = all_tfrs[0].copy()
glob_tfr._data = np.stack(all_tfrs[i]._data for i in range(len(sub_list))).mean(0)
img = glob_tfr.plot(baseline=None, picks='all', mode='logratio', tmin=-1.3, tmax=2, title='contra-ipsi', cmap='RdBu')
ff = 'tfr_' + 'avg' + '.png'
#img.savefig(op.join(config.path_tfrs, 'Plots', ff))