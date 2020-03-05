
import os
import os.path as op
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mne
from pathlib import Path
from library import helpers, config

subsub = 'VME_S03'

def get_lateralized_tfr(ID, condition_=''):
    eposIpsi = helpers.load_data(ID, config.path_epos_sorted + '/RoiIpsi' + condition_, '-epo')
    eposContra = helpers.load_data(ID, config.path_epos_sorted + '/RoiContra' + condition_, '-epo')

    freqs = np.logspace(*np.log10([6, 35]), num=20)
    n_cycles = freqs / 2.  # different number of cycle per frequency
    powerI = mne.time_frequency.tfr_morlet(eposIpsi, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                            return_itc=False, decim=3, n_jobs=1)
    powerC = mne.time_frequency.tfr_morlet(eposContra, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                            return_itc=False, decim=3, n_jobs=1)

    powDiff = powerC.copy()
    powDiff._data = powerC.reorder_channels(powerI.info['ch_names'])._data - powerI._data
    return powDiff

# powSum = powerC.copy()
# powSum._data = powerC.reorder_channels(powerI.info['ch_names'])._data + powerI._data

def extract_alpha_pow(data_diff_):
    tms_idx = [(config.times_dict['CDA_start'] < data_diff_.times) & (data_diff_.times < config.times_dict['CDA_end'])]
    frqs_idx = [(8 < data_diff_.freqs) & (data_diff_.freqs < 12)]
    alpha_dat = data_diff_._data[:,frqs_idx[0]][:,:,tms_idx[0]]
    alpha_pow = np.mean(alpha_dat)
    return alpha_pow


# powNorm = powerC.copy()
# powNorm._data = powDiff._data / powSum._data

# store mean tfrs:
mean_tfrs_dict = dict()


def write_mean_alphapwr_to_file(ID):
    #conds = ['LoadHighEccS', 'LoadHighEccM', 'LoadHighEccL', 'LoadLowEccS', 'LoadLowEccM', 'LoadLowEccL']
    #data = [str(mean_amplitues_dict[key] * 1000) for key in conds]
    file_mean_alphapwr = op.join(config.path_tfrs_summaries, ID + '_mean_alphapwr.csv')
    with open(file_mean_alphapwr, 'w') as ffile:
        for load in ['LoadHigh', 'LoadLow']:
            for ecc in ['EccS', 'EccM', 'EccL']:
                dat_ = get_lateralized_tfr(ID, load+ecc)
                apwr = extract_alpha_pow(dat_)
                data_txt = ";".join([ID, load, ecc, str(apwr)])
                ffile.write(data_txt + "\n")

write_mean_alphapwr_to_file(subsub)

pD = get_lateralized_tfr(subsub, '')
img = pD.plot(baseline=None, picks='all', mode='logratio', tmin=-1.3, tmax=2, title='contra-ipsi', cmap='RdBu')
ff = 'tfr_' + subsub + '.png'
# img.savefig(op.join(config.path_tfrs, 'Plots', ff))

# save to file:
helpers.save_data(pD, subsub + '-PowDiff', config.path_tfrs, append='-tfr')

# Combine subjects:

sub_list = [1,2,3] #3, 7, 22]
all_tfrs = []
for idx, sub in enumerate(sub_list):
    subID = 'VME_S%02d' % sub
    tfr = mne.time_frequency.read_tfrs(op.join(config.path_tfrs, subID + '-PowDiff-tfr.fif'))
    all_tfrs.append(tfr[0])  # Insert to the container

glob_tfr = all_tfrs[0].copy()
glob_tfr._data = np.stack(all_tfrs[i]._data for i in range(len(sub_list))).mean(0)
img = glob_tfr.plot(baseline=None, picks='all', mode='logratio', tmin=-1.3, tmax=2, title='contra-ipsi', cmap='RdBu')
ff = 'tfr_' + 'avg' + '.png'
img.savefig(op.join(config.path_tfrs, 'plots', ff))