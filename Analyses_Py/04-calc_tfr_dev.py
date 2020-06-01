

import os
import os.path as op
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import mne
from pathlib import Path
from library import helpers, config


def get_tfr(epos, picks='all', average=True):
    freqs = np.concatenate([np.arange(6, 26, 1)])#, np.arange(16,30,2)])
    n_cycles = freqs / 2.  # different number of cycle per frequency
    power = mne.time_frequency.tfr_morlet(epos, picks=picks, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                          return_itc=False, average=average, decim=1, n_jobs=-2)
    return power


def extract_alpha_pow(data_diff_):
    tms_idx = [(config.times_dict['CDA_start'] < data_diff_.times) & (data_diff_.times < config.times_dict['CDA_end'])]
    frqs_idx = [(8 < data_diff_.freqs) & (data_diff_.freqs < 12)]
    alpha_dat = data_diff_._data[:,frqs_idx[0]][:,:,tms_idx[0]]
    alpha_pow = np.mean(alpha_dat)
    return alpha_pow


def get_laterized_power_difference(pwr_, picks_contra, picks_ipsi): 
            if not len(picks_contra) == len(picks_ipsi): 
                raise ValueError('Picks must be of same length.')
            pwr_diff = pwr_.copy().pick_channels(picks_contra, ordered = True)
            d_contra = pwr_.copy().reorder_channels(picks_contra + picks_ipsi)._data[:,:len(picks_contra),:,:]
            d_ipsi = pwr_.copy().reorder_channels(picks_contra + picks_ipsi)._data[:,len(picks_contra):,:,:]
            pwr_diff._data = d_contra - d_ipsi
            return pwr_diff

# Write alpha time series to txt-file: 
def write_trialwise_alpha_ts_to_csv(pwr_obj, subID, part_epo, condition, calc_induc=False):
    # average over channels: 
    wr_out = pwr_obj.data.mean(axis=1)
    # pick out alpha freqs and average over these: 
    a_freqs = [list(pwr_obj.freqs).index(f) for f in pwr_obj.freqs if f>=config.alpha_freqs[0] and f<=config.alpha_freqs[1]]
    wr_out = wr_out[:,a_freqs,:].mean(axis=1)
    if calc_induc:
        fpath = op.join(config.path_tfrs_summaries, 'induc', 'timeseries', part_epo, condition)
    else: 
        fpath = op.join(config.path_tfrs_summaries, 'timeseries', part_epo, condition)
    
    helpers.chkmk_dir(fpath)
    fname = op.join(fpath, subID + '_alpha_latdiff_ts.csv')
    df = pd.DataFrame(wr_out.swapaxes(1,0), index=pwr_obj.times)
    df.to_csv(fname, sep=';', header=False, float_format='%.18f')
    #np.savetxt(fname, wr_out, fmt='%.18f', delimiter=';',header=';'.join(hdr))

def get_avg_timecourse(pwr_obj):
    wr_out = pwr_obj.data.mean(axis=1)
    # pick out alpha freqs and average over these: 
    a_freqs = [list(pwr_obj.freqs).index(f) for f in pwr_obj.freqs if f>=config.alpha_freqs[0] and f<=config.alpha_freqs[1]]
    wr_out = wr_out[:,a_freqs,:].mean(axis=1).mean(axis=0)
    return(wr_out)

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

#write_mean_alphapwr_to_file(subsub)

#TODO: Fix path
sub_list = np.setdiff1d(np.arange(1,28), config.ids_missing_subjects)
sub_list = [4]

for sub_nr in sub_list:
    subID = 'VME_S%02d' % sub_nr

    # calculate induced o rinduced+evoked power:
    calc_induc = False

    for part_epo in ['stimon', 'cue']:

        epos_ = helpers.load_data(subID, config.path_epos_sorted + '/' + part_epo + '/collapsed', '-epo')
        
        if calc_induc: 
            epos_.subtract_evoked()

        event_dict = helpers.get_event_dict(epos_.event_id)

        picks = config.chans_CDA_all

        pwr_ = get_tfr(epos_, picks=picks, average=False)
        #pwr_cue = get_tfr(epos_cue, average=False)

        pwr_diff = get_laterized_power_difference(pwr_, config.chans_CDA_dict['Contra'], config.chans_CDA_dict['Ipsi'])

        condition_averages_df = pd.DataFrame(pwr_diff.times, columns=['time'], index=None)

        ## Let's look at the different conditions: 

        # Main effect load: 
        for load in ['LoadLow', 'LoadHigh']:
            pD = pwr_diff[event_dict[load]]
            write_trialwise_alpha_ts_to_csv(pD, subID, part_epo, load, calc_induc)
            if calc_induc:
                helpers.save_data(pD.average(), subID + '-PowDiff-' + load, config.path_tfrs + '\\induc' + '\\'+ part_epo, append='-tfr')
            else: 
                helpers.save_data(pD.average(), subID + '-PowDiff-' + load, config.path_tfrs + '\\'+ part_epo, append='-tfr')
            # Add column with average timecourse to condition_averages_df:
            avg_tc = get_avg_timecourse(pD)
            condition_averages_df[load] = avg_tc

        # Main effect ecc: 
        for ecc in ['EccS', 'EccM', 'EccL']:
            pD = pwr_diff[event_dict[ecc]]
            write_trialwise_alpha_ts_to_csv(pD, subID, part_epo, ecc, calc_induc)
            if calc_induc:
                helpers.save_data(pD.average(), subID + '-PowDiff-' + ecc, config.path_tfrs + '\\induc' + '\\'+ part_epo, append='-tfr')
            else:
                helpers.save_data(pD.average(), subID + '-PowDiff-' + ecc, config.path_tfrs + '\\' + part_epo, append='-tfr')
            avg_tc = get_avg_timecourse(pD)
            condition_averages_df[ecc] = avg_tc
        # All:
        pD = pwr_diff[:]
        write_trialwise_alpha_ts_to_csv(pD, subID, part_epo, 'all', calc_induc)
        if calc_induc:
            helpers.save_data(pD.average(), subID + '-PowDiff-' + 'all', config.path_tfrs + '\\induc' + '\\'+ part_epo, append='-tfr')
        else:
            helpers.save_data(pD.average(), subID + '-PowDiff-' + 'all', config.path_tfrs + '\\' + part_epo, append='-tfr')
        avg_tc = get_avg_timecourse(pD)
        condition_averages_df['all'] = avg_tc

        # Write condition_averages_df to file: 
        if calc_induc:
            fpath = op.join(config.path_tfrs_summaries, 'induc', 'timeseries', part_epo, 'averages')
        else:
            fpath = op.join(config.path_tfrs_summaries, 'timeseries', part_epo, 'averages')
        helpers.chkmk_dir(fpath)
        fname = op.join(fpath, subID + '_condition_averages_ts.csv')
        condition_averages_df.to_csv(fname, sep=';', header=True, float_format='%.18f', encoding='UTF-8-sig')

# Interaction: 


# img.savefig(op.join(config.path_tfrs, 'Plots', ff))

