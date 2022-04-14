

import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import mne
from library import helpers, config

def get_tfrs_dict(part_epo, sub_list, pwr_style, picks='eeg'):
    avgtfrs = defaultdict(list)
    for sub in sub_list:
        subID = 'VME_S%02d' % sub
        epos_ = helpers.load_data(subID, config.path_epos_sorted + '/' +
                                  part_epo + '/collapsed', '-epo')
        
        # Shift time, so that 0 == Stimulus Onset:
        epos_ = epos_.shift_time(-config.times_dict['cue_dur'])
        
        if pwr_style == 'induced':
            epos_ = epos_.subtract_evoked()

        #picks = config.chans_CDA_all
        tfrs_ = get_tfr(epos_, picks=picks, average=False)


        event_dict = helpers.get_event_dict(epos_.event_id)

        sub_tfrs = list()

        for load in ['LoadLow', 'LoadHigh']:
            avgtfrs_load = get_tfr(epos_[event_dict[load]], picks=picks, average=True)
            avgtfrs[load].append(avgtfrs_load)
            sub_tfrs.append(avgtfrs_load)
            for ecc in ['EccS', 'EccM', 'EccL']:
                if load == 'LoadLow':  # we don't want to do this twice
                    avgtfrs_ecc = get_tfr(epos_[event_dict[ecc]], picks=picks, average=True)#tfrs_[event_dict[ecc]].copy().average()
                    avgtfrs[ecc].append(avgtfrs_ecc)
                    sub_tfrs.append(avgtfrs_ecc)
                # Interaction:
                avgtfrs_interac = get_tfr(epos_[event_dict[load]][event_dict[ecc]], picks=picks, average=True)
                avgtfrs[load+ecc].append(avgtfrs_interac)
                sub_tfrs.append(avgtfrs_interac)
        avgtfrs_all = get_tfr(epos_, picks=picks, average=True)
        avgtfrs['all'].append(avgtfrs_all)
        sub_tfrs.append(avgtfrs_all)

        # save list of tfrs: 
        fpath = op.join(config.path_tfrs, pwr_style, 'tfr_lists', part_epo)
        helpers.chkmk_dir(fpath)
        fname = op.join(fpath, subID + '-collapsed-avgTFRs-tfr.h5')
        mne.time_frequency.write_tfrs(fname, sub_tfrs, overwrite=True)


    return(avgtfrs)



def get_tfr(epos, picks='all', average=True):
    freqs = np.concatenate([np.arange(6, 26, 1)])  # , np.arange(16,30,2)])
    n_cycles = freqs / 2.  # different number of cycle per frequency
    power = mne.time_frequency.tfr_morlet(epos, picks=picks, freqs=freqs,
                                          n_cycles=n_cycles, use_fft=True,
                                          return_itc=False, average=average,
                                          decim=1, n_jobs=-2)
    return power


def extract_alpha_pow(data_diff_):
    tms_idx = [(config.times_dict['CDA_start'] < data_diff_.times) &
               (data_diff_.times < config.times_dict['CDA_end'])]
    frqs_idx = [(8 < data_diff_.freqs) & (data_diff_.freqs < 12)]
    alpha_dat = data_diff_._data[:, frqs_idx[0]][:, :, tms_idx[0]]
    alpha_pow = np.mean(alpha_dat)
    return alpha_pow


def get_laterized_power_difference(pwr_, picks_contra, picks_ipsi):
    if not len(picks_contra) == len(picks_ipsi):
        raise ValueError('Picks must be of same length.')
    pwr_diff = pwr_.copy().pick_channels(picks_contra, ordered=True)
    pwr_ordered_chans = pwr_.copy().reorder_channels(picks_contra + picks_ipsi)
    # keep flexible to use for data with 3 (AvgTFR) and 4 (EpoTFR) dimensions: 
    d_contra = pwr_ordered_chans._data[..., :len(picks_contra), :, :]
    d_ipsi = pwr_ordered_chans._data[..., len(picks_contra):, :, :]
    pwr_diff._data = d_contra - d_ipsi
    return pwr_diff


# Write alpha time series to txt-file:
def write_trialwise_alpha_ts_to_csv(pwr_obj, subID, part_epo, condition,
                                    pwr_style='evoked'):
    # average over channels:
    wr_out = pwr_obj.data.mean(axis=1)
    # pick out alpha freqs and average over these:
    a_freqs = [list(pwr_obj.freqs).index(f) for f in pwr_obj.freqs
               if f >= config.alpha_freqs[0] and f <= config.alpha_freqs[1]]
    wr_out = wr_out[:, a_freqs, :].mean(axis=1)
    fpath = op.join(config.path_tfrs, pwr_style, 'timeseries', part_epo,
                    condition)
    helpers.chkmk_dir(fpath)
    fname = op.join(fpath, subID + '_alpha_latdiff_ts.csv')
    df = pd.DataFrame(wr_out.swapaxes(1, 0), index=pwr_obj.times)
    df.to_csv(fname, sep=';', header=False, float_format='%.18f')


def get_avg_timecourse(pwr_obj):
    wr_out = pwr_obj.copy().data.mean(axis=1)
    # pick out alpha freqs and average over these:
    a_freqs = [list(pwr_obj.freqs).index(f) for f in pwr_obj.freqs
               if f >= config.alpha_freqs[0] and f <= config.alpha_freqs[1]]
    wr_out = wr_out[:, a_freqs, :].mean(axis=1).mean(axis=0)
    return(wr_out)

# powNorm = powerC.copy()
# powNorm._data = powDiff._data / powSum._data


# store mean tfrs:
def write_mean_alphapwr_to_file(ID):
    file_mean_alphapwr = op.join(config.path_tfrs_summaries,
                                 ID + '_mean_alphapwr.csv')
    with open(file_mean_alphapwr, 'w') as ffile:
        for load in ['LoadHigh', 'LoadLow']:
            for ecc in ['EccS', 'EccM', 'EccL']:
                dat_ = get_lateralized_tfr(ID, load+ecc)
                apwr = extract_alpha_pow(dat_)
                data_txt = ";".join([ID, load, ecc, str(apwr)])
                ffile.write(data_txt + "\n")

def plot_tf(ax, avg_tfrs, picks):
    res = avg_tfrs.plot(axes=ax, show=False,colorbar=True, picks=picks)
    ax.axvspan(0, 0.2, color='grey', alpha=0.3)
    ax.axvspan(2.2, 2.5, color='grey', alpha=0.3)
    ax.vlines(-0.8, ymin=-1000, ymax=10000)


def save_tfrs_dict(tfrs_dict, part_epo, pwr_style): 
    for k in tfrs_dict.keys(): 
        # save list of tfrs: 
        fpath = op.join(config.path_tfrs, pwr_style, 'avgtfr_dict', part_epo, k)
        helpers.chkmk_dir(fpath)
        fname = op.join(fpath, 'collapsed-avgTFRs-tfr.h5')
        mne.time_frequency.write_tfrs(fname, tfrs_dict[k], overwrite=True)



# TODO: Fix path
sub_list = np.setdiff1d(np.arange(1, 28), config.ids_missing_subjects +
                        config.ids_excluded_subjects)


# Mirror structure in "03.3-evaluate_CDA"

part_epo = 'fulllength'
pwr_style = 'induced'  # 'evoked' # 
avgtfrs_perc = get_tfrs_dict(part_epo, sub_list, pwr_style)

# Plot overall TF per hemisphere: 
grand_avgtfr_perc = mne.grand_average(avgtfrs_perc['all'])
def plot_tfr_side(ax, tfr, sides=['Contra', 'Ipsi'], cbar=True):
    if isinstance(sides, str):
        sides = [sides]
    for side, axis in zip(sides, ax):
        ha = tfr.copy().crop(-1, 2.3).plot(axes=axis[0], 
                                      show=False, 
                                      colorbar=cbar,
                                      picks=config.chans_CDA_dict[side], 
                                      combine='mean', 
                                      title='', 
                                      vmax=0.02, 
                                      vmin=-0.2)
        axis.axvspan(0, 0.2, color='grey', alpha=0.3)
        axis.axvspan(2.2, 2.5, color='grey', alpha=0.3)
        axis.vlines((-0.8, 0, 0.2, 2.2), ymin=-1000, ymax=10000, linestyles='dashed')
        axis.set_title(side)
        return ha

fig, ax = plt.subplots(1, 2, figsize=(6,4))
plt.subplots_adjust(hspace = 0.8)
plot_tfr_side(ax, grand_avgtfr_perc, 'Contra', False)

def plot_cbar(ax, vmin, vmax, cmap='RdBu', orient='horizontal'):
    cm = plt.get_cmap('RdBu')
    norm=mpl.colors.Normalize(vmin, vmax)
    cb = mpl.colorbar.ColorbarBase(ax, 
                                   orientation=orient, 
                                   cmap=cm, 
                                   norm=norm)

def plot_tfr_side(ax, tfr, cbar=True, tmin=None, tmax=None, 
                  vmin=None, vmax=None, title=''):
        ha = tfr.copy().crop(tmin, tmax).plot(axes=ax, 
                                      show=False, 
                                      colorbar=cbar,
                                      picks=config.chans_CDA_dict[side], 
                                      combine='mean', 
                                      title='', 
                                      vmax=vmax, 
                                      vmin=vmin)
        ax.axvspan(0, 0.2, color='grey', alpha=0.3)
        ax.axvspan(2.2, 2.5, color='grey', alpha=0.3)
        ax.vlines((-0.8, 0, 0.2, 2.2), ymin=-1000, ymax=10000, linestyles='dashed')
        ax.set_title(side)
        return ha



plt.show()

dd=list()
f,a = plt.subplots(4, figsize=(12,4))
plt.subplots_adjust(hspace = 0.8)
for i, ax, dB, tit, vmax in zip([grand_avgtfr_mean, 
                  grand_avgtfr_lograt, 
                  grand_avgtfr_perc], 
                 a[0:3], 
                 [False, False, False], 
                 ['mean subtracted', 'logratio', 'percent'], 
                 [None, 0.1, None]):
    m = i.crop(-1, 2.3).plot(axes=ax, show=False, 
                            colorbar=True, combine='mean', dB=dB, 
                            tmin=-1, tmax=2.3, vmax=vmax)
    dd.append(m)
    ax.axvspan(0, 0.2, color='grey', alpha=0.3)
    ax.axvspan(2.2, 2.5, color='grey', alpha=0.3)
    ax.vlines((-0.8, 0, 0.2, 2.2), ymin=-1000, ymax=10000, linestyles='dashed')
    ax.set_title(tit, loc='center')
    ax.set_xlabel("")



plt.show()

# Plot main tf diagram:
f,a = plt.subplots(3, figsize=(12,4))
#ax = plt.axes()
ax = a[0]

plot_tf(ax, avgtfrs['all'][1], picks=config.chans_CDA_dict['Contra'])
ax = a[1]
plot_tf(ax, avgtfrs['all'][1], picks=config.chans_CDA_dict['Ipsi'])

plt.show()

f,a = plt.subplots(4, figsize=(12,4))
for i, ax in zip([avgtfrs_mean, avgtfrs_lograt, avgtfrs_perc], a):
    avgtfrs_diff = [get_laterized_power_difference(aaa,
                                            config.chans_CDA_dict['Contra'],
                                            config.chans_CDA_dict['Ipsi']) for aaa in i['all']]
    gg = mne.grand_average(avgtfrs_diff)
    gg.plot(axes=ax, show=False, colorbar=True, combine='mean')

# Old version:


for sub_nr in sub_list:
    subID = 'VME_S%02d' % sub_nr

    # calculate induced or induced+evoked power:
    for pwr_style in ['evoked', 'induced']:

        for part_epo in ['fulllength']:  # ['stimon', 'cue']: #,

            epos_ = helpers.load_data(subID, config.path_epos_sorted + '/' +
                                      part_epo + '/collapsed', '-epo')
            
            if pwr_style == 'induced':
                epos_.subtract_evoked()

            event_dict = helpers.get_event_dict(epos_.event_id)

            picks = config.chans_CDA_all

            pwr_ = get_tfr(epos_, picks=picks, average=False)
            pwr_.apply_baseline((-config.times_dict['bl_dur_tfr'], 0))

            pwr_diff = get_laterized_power_difference(pwr_,
                                                      config.chans_CDA_dict['Contra'],
                                                      config.chans_CDA_dict['Ipsi'])

            # Get grand average across all conditions for both hemispheres:
            pwr_avg = pwr_.

            ## DF based approach:
            pwr_diff_df = pwr_diff.to_data_frame()
            
            # add column coding for conditions:
            # Load manipulation:
            fac_levels = ['LoadLow', 'LoadHigh']
            conds_load = [pwr_diff_df['condition'].isin(event_dict[fac]) for
                          fac in fac_levels]
            pwr_diff_df['load'] = np.select(conds_load, fac_levels, 'NA')

            # Ecc manipulation:
            fac_levels = ['EccS', 'EccM', 'EccL']
            conds_ecc = [pwr_diff_df['condition'].isin(event_dict[fac]) for
                         fac in fac_levels]
            pwr_diff_df['ecc'] = np.select(conds_ecc, fac_levels, 'NA')

            mask_alpha = ((pwr_diff_df['frequency'] >= config.alpha_freqs[0]) &
                          (pwr_diff_df['frequency'] <= config.alpha_freqs[1]))


            
            # filter out alpha freqs:
            alpha_df = pwr_diff_df.loc[mask_alpha, :]
            #alpha_df_sum = alpha_df.groupby(['time', 'load', 'ecc']).agg({'P3':np.mean}).reset_index()
            
            # get avg. power across ROI channels:
            chans = config.chans_CDA_dict['Left']
            alpha_df['roi_amp'] = alpha_df.loc[:, chans].mean(axis=1)

            # plot main effects:
            # ecc:
            alpha_avg = alpha_df.groupby(['time', 'ecc']).agg(np.mean)
            alpha_plt = alpha_avg.reset_index().pivot(columns='ecc', 
                                                      values='roi_amp',
                                                      index='time')
            alpha_plt.plot.line()

            # 

            asum_piv = alpha_df_sum.pivot(index='time', columns=['ecc', 'load'], values='P3')
            asum_piv = alpha_df_sum.pivot_table(index='time', columns=['load', 'ecc'], values='P3')

            fig, ax = plt.subplots()
            for key, grp in asum.groupby(['Color']):
                ax = grp.plot(ax=ax, kind='line', x='time', y='P3', c=key)

            condition_averages_df = pd.DataFrame(pwr_diff.times,
                                                 columns=['time'],
                                                 index=None)

            # Let's look at the different conditions: 

            list_tfr = []

            # Main effect load: 
            for load in ['LoadLow', 'LoadHigh']:
                # Main eff load:
                pD = pwr_diff[event_dict[load]]
                avg_tc = get_avg_timecourse(pD)
                condition_averages_df[load] = avg_tc
                # write average to list: 
                pD = pD.average()
                pD.comment = load
                list_tfr.append(pD) 
                for ecc in ['EccS', 'EccM', 'EccL']:
                    # Main eff ecc:
                    pD = pwr_diff[event_dict[ecc]]
                    avg_tc = get_avg_timecourse(pD)
                    condition_averages_df[ecc] = avg_tc
                    # write to list:
                    pD = pD.average()
                    pD.comment = ecc
                    list_tfr.append(pD) 

                    # Interaction:
                    pD = pwr_diff[event_dict[load]][event_dict[ecc]]
                    write_trialwise_alpha_ts_to_csv(pD, subID, part_epo, load+ecc, pwr_style)
                    # Add column with average timecourse to condition_averages_df:
                    avg_tc = get_avg_timecourse(pD)
                    condition_averages_df[load+ecc] = avg_tc
                    # write to list: 
                    pD = pD.average()
                    pD.comment = load+ecc
                    list_tfr.append(pD) 
            
            # All:
            pD = pwr_diff[:]
            avg_tc = get_avg_timecourse(pD)
            condition_averages_df['all'] = avg_tc
            # write to list: 
            pD = pD.average()
            pD.comment = 'all'
            list_tfr.append(pD) 
            
            # Write condition_averages_df to file: 
            fpath = op.join(config.path_tfrs, pwr_style, 'timeseries', part_epo, 'averages')
            helpers.chkmk_dir(fpath)
            fname = op.join(fpath, subID + '_condition_averages_ts.csv')
            condition_averages_df.to_csv(fname, sep=';', header=True, float_format='%.18f', encoding='UTF-8-sig')

            # save list of tfrs: 
            fpath = op.join(config.path_tfrs, pwr_style, 'tfr_lists', part_epo)
            helpers.chkmk_dir(fpath)
            fname = op.join(fpath, subID + '-PowDiff-avgTFRs-tfr.h5')
            mne.time_frequency.write_tfrs(fname, list_tfr, overwrite=True)

                # # save average tfr:
                # if calc_induc:
                #     helpers.save_data(pD.average(), subID + '-PowDiff-' + load + ecc, config.path_tfrs + '\\induc' + '\\'+ part_epo, append='-tfr')
                # else: 
                #     helpers.save_data(pD.average(), subID + '-PowDiff-' + load + ecc, config.path_tfrs + '\\'+ part_epo, append='-tfr')


        # # Main effect ecc: 
        # for ecc in ['EccS', 'EccM', 'EccL']:
        #     pD = pwr_diff[event_dict[ecc]]
        #     write_trialwise_alpha_ts_to_csv(pD, subID, part_epo, ecc, calc_induc)
        #     if calc_induc:
        #         helpers.save_data(pD.average(), subID + '-PowDiff-' + ecc, config.path_tfrs + '\\induc' + '\\'+ part_epo, append='-tfr')
        #     else:
        #         helpers.save_data(pD.average(), subID + '-PowDiff-' + ecc, config.path_tfrs + '\\' + part_epo, append='-tfr')
        #     avg_tc = get_avg_timecourse(pD)
        #     condition_averages_df[ecc] = avg_tc
        # # All:
        # pD = pwr_diff[:]
        # write_trialwise_alpha_ts_to_csv(pD, subID, part_epo, 'all', calc_induc)
        # if calc_induc:
        #     helpers.save_data(pD.average(), subID + '-PowDiff-' + 'all', config.path_tfrs + '\\induc' + '\\'+ part_epo, append='-tfr')
        # else:
        #     helpers.save_data(pD.average(), subID + '-PowDiff-' + 'all', config.path_tfrs + '\\' + part_epo, append='-tfr')
        # avg_tc = get_avg_timecourse(pD)
        # condition_averages_df['all'] = avg_tc

        # Write condition_averages_df to file: 
        # if calc_induc:
        #     fpath = op.join(config.path_tfrs_summaries, 'induc', 'timeseries', part_epo, 'averages')
        # else:
        #     fpath = op.join(config.path_tfrs_summaries, 'timeseries', part_epo, 'averages')
        # helpers.chkmk_dir(fpath)
        # fname = op.join(fpath, subID + '_condition_averages_ts.csv')
        # condition_averages_df.to_csv(fname, sep=';', header=True, float_format='%.18f', encoding='UTF-8-sig')

# Interaction: 


# img.savefig(op.join(config.path_tfrs, 'Plots', ff))

