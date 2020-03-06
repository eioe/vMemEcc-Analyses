

import os
import os.path as op
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mne
from pathlib import Path
from library import helpers, config

subsub = 'VME_S01'

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
img = pD.plot(baseline=None, 
              picks='all', 
              mode='mean', 
              tmin=-1.3, 
              tmax=2, 
              title='contra-ipsi', 
              cmap='RdBu')
ff = 'tfr_' + subsub + '.png'
# img.savefig(op.join(config.path_tfrs, 'Plots', ff))

sub_list = [1,2,3,5, 6, 7, 8, 9, 10, 13, 16, 17, 18, 20, 22, 23, 24, 26, 27]

for sub in sub_list:
    subsub = 'VME_S%02d' % sub
    for load in ['LoadLow', 'LoadHigh']:
        pD = get_lateralized_tfr(subsub, load)
        # save to file:
        cond_str = '-'+load
        helpers.save_data(pD, subsub + '-PowDiff'+cond_str, config.path_tfrs, append='-tfr')

