#%%
import os
import os.path as op
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from scipy import stats
from scipy.ndimage import measurements

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder

import mne
from mne.stats import permutation_cluster_1samp_test, f_mway_rm, f_threshold_mway_rm
from mne.decoding import CSP
from library import helpers, config


# %%
def get_epos(subID, epo_part, signaltype, condition, event_dict):
    if signaltype == 'uncollapsed':
        fname = op.join(config.path_rejepo, subID + '-' + epo_part +
                        '-postica-rejepo' + '-epo.fif')
    elif signaltype in ['collapsed']:
        fname = op.join(config.path_epos_sorted, epo_part, signaltype,
                        subID + '-epo.fif')
    else:
        raise ValueError(f'Invalid value for "signaltype": {signaltype}')
    epos = mne.read_epochs(fname, verbose=False)
    epos = epos.pick_types(eeg=True)
    uppers = [letter.isupper() for letter in condition]
    if (np.sum(uppers) > 2):
        cond_1 = condition[:np.where(uppers)[0][2]]
        cond_2 = condition[np.where(uppers)[0][2]:]
        selection = epos[event_dict[cond_1]][event_dict[cond_2]]
    else:
        selection = epos[event_dict[condition]]
    return(selection)

def get_sensordata(subID, epo_part, signaltype, conditions, event_dict):
    epos_dict = defaultdict(dict)
    for cond in conditions:
        epos_dict[cond] = get_epos(subID,
                                   epo_part=epo_part,
                                   signaltype=signaltype,
                                   condition=cond,
                                   event_dict=event_dict)

    times = epos_dict[conditions[0]][0].copy().times

    # Setup data:
    X_epos = mne.concatenate_epochs([epos_dict[cond] for cond in conditions])
    n_ = {cond: len(epos_dict[cond]) for cond in conditions}

    times_n = times

    y = np.r_[np.zeros(n_[conditions[0]]),
              np.concatenate([(np.ones(n_[conditions[i]]) * i)
                              for i in np.arange(1, len(conditions))])]

    return X_epos, y, times_n


def load_avgtfr(subID, condition, pwr_style='induced', 
                part_epo='fulllength', baseline=None, mode=None): 
    fpath = op.join(config.path_tfrs, pwr_style, 'tfr_lists', part_epo)
    fname = op.join(fpath, subID + '-collapsed-avgTFRs-tfr.h5')
    tfr_ = mne.time_frequency.read_tfrs(fname, condition=condition)
    if baseline is not None:
        tfr_.apply_baseline(baseline=baseline, mode=mode)
    return tfr_


def plot_tfr_side(ax, tfr, picks, cbar=True, tmin=None, tmax=None, 
                  vmin=None, vmax=None, title='', cmap='RdBu_r'):
        ha = tfr.copy().crop(tmin, tmax).plot(axes=ax, 
                                      show=False, 
                                      colorbar=cbar,
                                      picks=picks, 
                                      combine='mean', 
                                      title=title, 
                                      vmax=vmax, 
                                      vmin=vmin, 
                                      cmap=cmap)
        ytick_range = ax.get_ylim()
        ytick_vals = np.arange(*np.round(ytick_range), 2)
        ax.yaxis.set_ticks(ytick_vals)
        ax.axvspan(0, 0.2, color='grey', alpha=0.3)
        ax.axvspan(2.2, 2.5, color='grey', alpha=0.3)
        ax.vlines((-0.8, 0, 0.2, 2.2), ymin=-1000, ymax=10000, linestyles='dashed')
        #ax.set_title('uV^2/Hz')
        return ha

def get_lateralized_power_difference(pwr_, picks_contra, picks_ipsi):
    if not len(picks_contra) == len(picks_ipsi):
        raise ValueError('Picks must be of same length.')
    pwr_diff = pwr_.copy().pick_channels(picks_contra, ordered=True)
    pwr_ordered_chans = pwr_.copy().reorder_channels(picks_contra + picks_ipsi)
    # keep flexible to use for data with 3 (AvgTFR) and 4 (EpoTFR) dimensions: 
    d_contra = pwr_ordered_chans._data[..., :len(picks_contra), :, :]
    d_ipsi = pwr_ordered_chans._data[..., len(picks_contra):, :, :]
    pwr_diff._data = d_contra - d_ipsi
    return pwr_diff


def run_cbp_test(data, p_initial = 0.05, p_thresh = 0.05):
    # number of permutations to run
    n_permutations = 1000 
    # set initial threshold
    p_initial = p_initial
    # set family-wise p-value
    p_thresh = p_thresh
    connectivity = None
    tail = 0.  # for two sided test

    # set cluster threshold
    n_samples = len(data)
    threshold = -stats.t.ppf(p_initial / (1 + (tail == 0)), n_samples - 1)
    if np.sign(tail) < 0:
        threshold = -threshold

    cluster_stats = permutation_cluster_1samp_test(
        data, threshold=threshold, n_jobs=config.n_jobs, verbose=True, tail=tail,
        step_down_p=0.05, connectivity=connectivity,
        n_permutations=n_permutations, seed=42)

    T_obs, clusters, cluster_p_values, _ = cluster_stats
    return(T_obs, clusters, cluster_p_values)


def plot_cbp_result_cda(ax, T_obs, clusters, cluster_p_values, p_thresh, 
                        cbp_times=None, times_full=None):
    if cbp_times is None:
        if times_full is None: 
            times_full = range(len(T_obs))
        cbp_times = times_full
    if times_full is None: 
        times_full = cbp_times
    y_max = np.max(np.abs(T_obs)) * np.array([-1.1, 1.1])
    for i_c, c in enumerate(clusters):
        c = c[0]
        if cluster_p_values[i_c] < p_thresh:
            h1 = ax.axvspan(cbp_times[c.start], cbp_times[c.stop - 1],
                            color='r', alpha=0.3)
    hf = ax.plot(cbp_times, T_obs, 'g')
    ax.hlines(0, times_full[0], times_full[-1])
    ax.legend((h1,), (u'p < %s' % p_thresh,), loc='upper right', ncol=1, prop={'size': 9})
    ax.set(xlabel="Time (s)", ylabel="T-values",
            ylim=y_max, xlim=times_full[np.array([0,-1])])
    #fig.tight_layout(pad=0.5)
    ax.axvspan(0, 0.2, color='grey', alpha=0.3)
    ax.axvspan(2.2, 2.3, color='grey', alpha=0.3)
    ax.vlines([-0.8, 0,0.2,2.2], *y_max, linestyles='--', colors='k',
                    linewidth=1., zorder=1)
    #ax.set_aspect(0.33)
    ax.set_title('')
    #ax.set_aspect('auto', adjustable='datalim')
    #ax.set(aspect=1.0/ax.get_data_ratio()*0.25, adjustable='box')
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)


def plot_main_eff(cond, cond_dict, data, ax, n_boot=1000):
    sns.lineplot(x='time', y='pwr', hue=cond, data=data, n_boot=n_boot, 
                 palette=[config.colors[l] for l in cond_dict[cond]], ax=ax)
    ytick_range = ax.get_ylim()
    ax.set(xlim=(-1.1, 2.3), ylim=ytick_range)
    ax.set_ylabel('V$^2$')
    ax.set_xlabel('Time (s)')
    ax.axvspan(0, 0.2, color='grey', alpha=0.3)
    ax.axvspan(2.2, 2.5, color='grey', alpha=0.3)
    ax.vlines((-0.8, 0, 0.2, 2.2), ymin=ytick_range[0], ymax=ytick_range[1], 
            linestyles='dashed')
    ax.hlines(0, xmin=-1.1, xmax=2.3)



def get_tfr(epos, picks='all', average=True, freqs=None):
    if freqs is None:
        freqs = np.concatenate([np.arange(6, 26, 1)])  # , np.arange(16,30,2)])
    n_cycles = freqs / 2.  # different number of cycle per frequency
    power = mne.time_frequency.tfr_morlet(epos, picks=picks, freqs=freqs,
                                          n_cycles=n_cycles, use_fft=True,
                                          return_itc=False, average=average,
                                          decim=1, n_jobs=-2)
    return power


def get_mean_pwrdiff_per_trial(subID, freqs_to_avg=np.arange(8,13),
                               t_start=None, t_stop=None, pwr_style='induced',
                               part_epo='fulllength', 
                               picks=config.chans_CDA_all):
    epos_ = helpers.load_data(subID, config.path_epos_sorted + '/' +
                              part_epo + '/collapsed', '-epo')
    
    # Shift time, so that 0 == Stimulus Onset:
    epos_ = epos_.shift_time(-config.times_dict['cue_dur'])
    
    if pwr_style == 'induced':
        epos_ = epos_.subtract_evoked()

    event_dict = helpers.get_event_dict(epos_.event_id)

    sub_dfs = list()

    for load in ['LoadLow', 'LoadHigh']:
        for ecc in ['EccS', 'EccM', 'EccL']:
            epos_cond = epos_[event_dict[load]][event_dict[ecc]]
            # Grab trial numbers (correct for collapsing):
            trial_nums = [tn-720 if tn>=720 else tn for tn in epos_cond.selection]
            # correct for zero indexing:
            trial_nums = [t+1 for t in trial_nums]
            # Get TFRs and calc lat power:
            tfrs_cond = get_tfr(epos_cond,
                                   picks=picks, 
                                   average=False, 
                                   freqs=freqs_to_avg)
            tfrs_cond.apply_baseline(baseline=(-(config.times_dict['cue_dur']+0.3), 
                                               -(config.times_dict['cue_dur']+0.1)), 
                                     mode='mean')
            tfr_lat = get_lateralized_power_difference(tfrs_cond, 
                                                       config.chans_CDA_dict['Contra'], 
                                                       config.chans_CDA_dict['Ipsi'])
            # Crop to cluster times: 
            tfr_lat.crop(t_start, t_stop)
            tfr_lat_df = tfr_lat.to_data_frame()
            tfr_lat_df = tfr_lat_df.loc[:, ['epoch'] + tfr_lat.ch_names]
            tfr_lat_df = tfr_lat_df.groupby('epoch').agg('mean').reset_index()
            tfr_lat_df['mean_pwr_diff'] = tfr_lat_df.loc[:, tfr_lat.ch_names].mean(axis=1)
            tfr_lat_df['c_StimN'] = load
            tfr_lat_df['c_Ecc'] = ecc
            tfr_lat_df['trial_num'] = [trial_nums[i] for i in tfr_lat_df.epoch]
            tfr_lat_df = tfr_lat_df.drop('epoch', axis=1)
            
            # store in list
            sub_dfs.append(tfr_lat_df)

    sub_df = pd.concat(sub_dfs, axis=0)
    sub_df['subID'] = subID
    return(sub_df)


def get_condition_pwrdiff_df(factor, cond_dict, sub_list_str):
    df_list = list()
    for cond in cond_dict[factor]:
        tfr_list = [load_avgtfr(subID, cond, pwr_style, part_epo, 
                    baseline=(-(config.times_dict['cue_dur']+0.3), 
                            -(config.times_dict['cue_dur']+0.1)), 
                    mode='mean') for subID in sub_list_str]

        diffs_ = [get_lateralized_power_difference(tfr, 
                                                config.chans_CDA_dict['Contra'], 
                                                config.chans_CDA_dict['Ipsi']) for 
                tfr in tfr_list]

        frqs_idx = [(8 <= diffs_[0].freqs) & (diffs_[0].freqs <= 12)]
        diffs_mean = [d.data[:, frqs_idx[0], :].mean(axis=(0, 1)) for d in diffs_]
        plt_df = pd.DataFrame(np.array(diffs_mean).swapaxes(1, 0),
                              columns=sub_list_str)
        plt_df['time'] = times
        plt_df[factor] = cond
        df_list.append(plt_df)

    df_load_concat = pd.concat(df_list)
    df_load_long = df_load_concat.melt(id_vars=['time', factor], 
                                       var_name='subID', 
                                       value_name='pwr')
    return(df_load_long)


#%%
sub_list = np.setdiff1d(np.arange(1, 28), config.ids_missing_subjects +
                        config.ids_excluded_subjects)               
sub_list_str = ['VME_S%02d' % sub for sub in sub_list]

part_epo = 'fulllength'
pwr_style = 'induced'  # 'evoked' # 
cond_dict = {'Load': ['LoadLow', 'LoadHigh'], 
             'Ecc': ['EccS', 'EccM', 'EccL']}


#%% get list with avg TFRs for all trials and conditions: !!! THIS BLOCK TAKES AGES
tfr_list = [load_avgtfr(subID, 'all', pwr_style, part_epo, 
                        baseline=(-(config.times_dict['cue_dur']+0.3), 
                                  -(config.times_dict['cue_dur']+0.1)), 
                        mode='mean') 
            for subID in sub_list_str]

all_conds = [c for fac in cond_dict for c in cond_dict[fac]]
tfr_by_cond = {cond: [load_avgtfr(subID, cond, pwr_style, part_epo, 
                    baseline=(-(config.times_dict['cue_dur']+0.3), 
                                -(config.times_dict['cue_dur']+0.1)), 
                    mode='mean') 
                    for subID in sub_list_str]
               for cond in all_conds}

grand_avgtfr_all = mne.grand_average(tfr_list)

#%% Set up params:
times = grand_avgtfr_all.times
freqs = grand_avgtfr_all.freqs

# %%
# Calc overall difference between high and low load:
load = 'LoadLow'
side = 'Ipsi'
ga = defaultdict(mne.EvokedArray)
for load in ['LoadLow', 'LoadHigh']:
    tmp = tfr_by_cond[load]
    ga[load] = mne.grand_average(tmp)

diff_ga_data = ga['LoadHigh'].data - ga['LoadLow'].data

info = ga['LoadHigh'].info
diff_ga = mne.time_frequency.AverageTFR(info, diff_ga_data, times, freqs, nave=21)

diff_diff = get_lateralized_power_difference(diff_ga, config.chans_CDA_dict['Contra'], 
                                                      config.chans_CDA_dict['Ipsi'])

fig, ax = plt.subplots(1, figsize=(6,4))
tf_contra = plot_tfr_side(ax, diff_diff, picks=config.chans_CDA_dict['Contra'], 
            tmin=-1.1, tmax=2.3, title=side, cbar=True, 
            vmin=-6e-10, vmax=6e-10)


# %% CSP decoding pipe
''' This block can probably entirely be deleted.

def decode(sub_list_str, conditions, event_dict, save_scores = True, part_epo = 'stimon', 
           signaltype='collapsed'):
    # Code from: 
    # https://mne.tools/stable/auto_examples/decoding/plot_decoding_csp_timefreq.html#sphx-glr-auto-examples-decoding-plot-decoding-csp-timefreq-py

    contrast_str = '_vs_'.join(conditions)
    scoring = 'accuracy'
    cv_folds = 5

    clf = make_pipeline(CSP(n_components=6, reg=None, log=True, norm_trace=False),
                        LinearDiscriminantAnalysis())
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Classification & time-frequency parameters
    tmin = -0.5 # -config.times_dict['cue_dur']
    tmax =  2.3  # config.times_dict['stim_dur'] + config.times_dict['retention_dur']
    n_cycles = None  # how many complete cycles: used to define window size
    w_size = 0.5
    w_overlap = 0.5 # how much shall the windows overlap [value in [0,1]; 0: no overlap, 1: full overlap]
    min_freq = 8
    max_freq = 14
    n_freqs = 3  # how many frequency bins to use


    # Assemble list of frequency range tuples
    freqs = np.linspace(min_freq, max_freq, n_freqs + 1)  # assemble frequencies
    freq_ranges = list(zip(freqs[:-1], freqs[1:]))  # make freqs list of tuples


    if ((n_cycles is not None) and (w_size is None)): 
        # Infer window spacing from the max freq and number of cycles to avoid gaps
        window_spacing = (n_cycles / np.max(freqs) / 2.)
        centered_w_times = np.arange(tmin, tmax, window_spacing)[1:]
    elif (((w_size is not None)) and (n_cycles is None)): 
        assert 0 <= float(w_overlap or -1) < 1, f'Invalid value for w_overlap: {w_overlap}'
        step_size = w_size * (1 - w_overlap)
        centered_w_times = np.arange(tmin + (w_size / 2.), tmax - (w_size / 2), step_size)
    else: 
        raise ValueError(f'Invalid combination of values for w_size and n_cylces. Exactly one must be None.')

    n_windows = len(centered_w_times)

    tf_scores_list = list()
    for subID in sub_list_str:
        part_epo = part_epo

        print(f'Running {subID}')

        X_epos, y, t = get_sensordata(subID, part_epo, signaltype, conditions, event_dict)
        # init scores
        tf_scores = np.zeros((n_freqs, n_windows))

        # Loop through each frequency range of interest
        for freq, (fmin, fmax) in enumerate(freq_ranges):

            print(f'Freq. {freq} of {len(freq_ranges)}')

            if (w_size is None):
                # Infer window size based on the frequency being used
                w_size = n_cycles / ((fmax + fmin) / 2.)  # in seconds

            # Apply band-pass filter to isolate the specified frequencies
            X_epos_filter = X_epos.copy().filter(fmin, fmax, n_jobs=-2, fir_design='firwin')

            # Roll covariance, csp and lda over time
            for t, w_time in enumerate(centered_w_times):

                # Center the min and max of the window
                w_tmin = w_time - w_size / 2.
                w_tmax = w_time + w_size / 2.

                # Crop data into time-window of interest
                X = X_epos_filter.copy().crop(w_tmin, w_tmax).get_data()

                # Save mean scores over folds for each frequency and time window
                tf_scores[freq, t] = np.mean(cross_val_score(estimator=clf, X=X, y=y,
                                                            scoring='accuracy', cv=cv,
                                                            n_jobs=-2), axis=0)
        tf_scores_list.append(tf_scores)

        if save_scores:
            sub_scores_ = np.asarray(tf_scores_list)
            fpath = op.join(config.path_decod_tfr, part_epo, signaltype, contrast_str, 'scores')
            helpers.chkmk_dir(fpath)
            fname = op.join(fpath, 'scores_per_sub.npy')
            np.save(fname, sub_scores_)
            np.save(fname[:-4] + '__times' + '.npy', centered_w_times)
            np.save(fname[:-4] + '__freqs' + '.npy', freq_ranges)
            del(fpath, fname)


        # save info:
        if save_scores:
            info_dict = {'tmin': tmin, 
                         'tmax': tmax, 
                         'n_cycles': n_cycles, 
                         'w_size': w_size,
                         'w_overlap': w_overlap,
                         'min_freq': min_freq, 
                         'max_freq': max_freq,
                         'n_freqs': n_freqs,
                         'cv_folds': cv_folds, 
                         'scoring': scoring}
            fpath = op.join(config.path_decod_tfr, part_epo, signaltype, contrast_str)
            fname = op.join(fpath, 'info.json')
            with open(fname, 'w+') as outfile:  
                json.dump(info_dict, outfile)

    return tf_scores_list, centered_w_times
 

# %%
a, b = decode(sub_list_str, ['EccS', 'EccL'], config.event_dict)

# %%
res_load = decode(sub_list_str, ['LoadLow', 'LoadHigh'], config.event_dict)
res_load_eccL = decode(sub_list_str, ['LoadLowEccL', 'LoadHighEccL'], config.event_dict)
res_load_eccS = decode(sub_list_str, ['LoadLowEccS', 'LoadHighEccS'], config.event_dict)
res_load_eccM = decode(sub_list_str, ['LoadLowEccM', 'LoadHighEccM'], config.event_dict)
res_ecc = decode(sub_list_str, ['EccM', 'EccL'], config.event_dict) 

# %% Plot decoding results

def load_res_dectfr(conditions, part_epo='stimon', signaltype='collapsed'):
    contrast_str = '_vs_'.join(conditions)
    fpath = op.join(config.path_decod_tfr, part_epo, signaltype, contrast_str, 'scores')
    fname = op.join(fpath, 'scores_per_sub.npy')
    res = np.load(fname)
    times = np.load(fname[:-4] + '__times.npy')
    freqs = np.load(fname[:-4] + '__freqs.npy')
    return(res, times, freqs)


def plot_im_dectfr(scores_avg, conditions, times, freqs):
    fig, ax = plt.subplots(1,1)
    im = plt.imshow(scores_avg, origin='lower', cmap='Greens')
    ax.set_yticks(range(len(freqs)))
    ax.set_yticklabels([int(np.mean(f)) for f in freqs])
    ax.set_xticks(range(len(times)))
    ax.set_xticklabels(times)
    cbar = ax.figure.colorbar(im, ax=ax)
    plt.title('_vs_'.join(conditions))
    return(fig)


def plot_line_dectfr(scores_avg, conditions, times):
    fig, ax = plt.subplots(1,1)
    im = plt.plot(scores_avg[1,:])
    ax.set_xticks(range(len(times)))
    ax.set_xticklabels(times)


conditions = ['LoadLow', 'LoadHigh']

a, times, freqs = load_res_dectfr(conditions)
scores_avg = np.mean(a, axis=0)

plot_im_dectfr(scores_avg, conditions, times, freqs)
plot_line_dectfr(scores_avg, conditions, times)



conditions = ['EccS', 'EccL']

a, times, freqs = load_res_dectfr(conditions)
scores_avg = np.mean(a, axis=0)

plot_im_dectfr(scores_avg, conditions, times, freqs)
plot_line_dectfr(scores_avg, conditions, times)

'''

#%%###########################################################################################
# Plot TF diag per hemisphere across all conditions:
for side in ['Contra', 'Ipsi']:
    fig, ax = plt.subplots(1, figsize=(6,4))
    tf_contra = plot_tfr_side(ax, grand_avgtfr_all, picks=config.chans_CDA_dict[side], 
                tmin=-1.1, tmax=2.3, title=side, cbar=True, 
                vmin=-6e-10, vmax=6e-10)

    # Save it: 
    fpath = op.join(config.path_plots, 'TFR', part_epo)
    helpers.chkmk_dir(fpath)
    fname = op.join(fpath, f'grandavgTFR_{side}.png')
    # fig.savefig(fname, bbox_inches="tight")
##############################################################################################
##############################################################################################


#%% Calculate the difference between the hemispheres:
diff_avgtfr_all = get_lateralized_power_difference(grand_avgtfr_all, 
                                                 config.chans_CDA_dict['Contra'], 
                                                 config.chans_CDA_dict['Ipsi'])

##############################################################################################
fig, ax = plt.subplots(1, figsize=(6,4))
tf_contra = plot_tfr_side(ax, diff_avgtfr_all, picks='all', 
            tmin=-1.1, tmax=2.3, title='Contra - Ipsi', cbar=True, 
            vmin=-4e-10, vmax=4e-10, cmap='PRGn')

# Save it: 
fpath = op.join(config.path_plots, 'TFR', part_epo)
helpers.chkmk_dir(fpath)
fname = op.join(fpath, f'grandavgTFR_Difference.png')
fig.savefig(fname, bbox_inches="tight")

# Create version with red box around classical alpha range (8-12Hz):
rect = mpl.patches.Rectangle((-1.1, 8), 2.3 - -1.1, 4, 
                             linewidth=3, edgecolor='r', facecolor='none', alpha=0.5)
ax.add_patch(rect)
# Save it: 
fpath = op.join(config.path_plots, 'TFR', part_epo)
helpers.chkmk_dir(fpath)
fname = op.join(fpath, f'grandavgTFR_Difference_classAlpha.png')
fig.savefig(fname, bbox_inches="tight")
##############################################################################################
##############################################################################################


#%% Get the difference per subject: 

diffs_avgtfr_all = [get_lateralized_power_difference(tfr, 
                                                   config.chans_CDA_dict['Contra'], 
                                                   config.chans_CDA_dict['Ipsi']) for 
                    tfr in tfr_list]


##############################################################################################
#%% Make single subject plots:
fig, ax = plt.subplots(7,3)
for tfr_, ax_ in zip(diffs_avgtfr_all, ax.reshape(-1)):
    plot_tfr_side(ax_, tfr_, picks='all', 
            tmin=-1.1, tmax=2.3, title='Contra - Ipsi', cbar=False, 
            cmap='PRGn')
    ax_.xaxis.label.set_visible(False)
    ax_.yaxis.label.set_visible(False)
for ax_ in ax[:-1].reshape(-1):
    ax_.xaxis.set_visible(False)
ax[3,0].yaxis.label.set_visible(True)
ax[6,1].xaxis.label.set_visible(True)

# Save it: 
fpath = op.join(config.path_plots, 'TFR', part_epo)
helpers.chkmk_dir(fpath)
fname = op.join(fpath, 'singlesubs_avgTFR_Difference.png')
fig.savefig(fname, bbox_inches="tight")
##############################################################################################
##############################################################################################


#%% Extract standard alpha (8-12Hz):
frqs_idx = [(8 <= diffs_avgtfr_all[0].freqs) & (diffs_avgtfr_all[0].freqs <= 12)]
diffs_stdalpha_mean = [d.data[:, :, :].mean(axis=(0,1)) for d in diffs_avgtfr_all]

##############################################################################################
# plot it in 2D: 
plt_df = pd.DataFrame(np.array(diffs_stdalpha_mean).swapaxes(1,0), 
                      columns=sub_list_str)
plt_df['time'] = times
plt_df_long = plt_df.melt(id_vars='time', var_name='subID', value_name='pwr')
plt_df_long['hue'] = 'Alpha Power: Contra - Ipsi'

fig, axes = plt.subplots(2, figsize=(24,4), sharex=True)
ax = axes[0]
sns.lineplot(x='time', y='pwr', hue='hue', data=plt_df_long, n_boot=100, ax=ax)
ytick_range = ax.get_ylim()
ax.set(xlim=(-1.1, 2.3), ylim=ytick_range)
ax.set_ylabel('µV^2/Hz')
#ax.set_xlabel('Time (s)')
#ytick_vals = np.arange(*np.round(ytick_range), 2)
#ax.yaxis.set_ticks(ytick_vals)
ax.axvspan(0, 0.2, color='grey', alpha=0.3)
ax.axvspan(2.2, 2.5, color='grey', alpha=0.3)
ax.vlines((-0.8, 0, 0.2, 2.2), ymin=ytick_range[0], ymax=ytick_range[1], 
          linestyles='dashed')
ax.hlines(0, xmin=-1.1, xmax=2.3)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[1:], labels=labels[1:], loc=1, prop={'size': 9})
ax.xaxis.label.set_visible(False)
##############################################################################################
##############################################################################################


#%% run CBP test on classical alpha data (sub x times):
# crop to time of interest: 
mask_time = [(times >= -0.8) & (times <= 2.2)]
data = np.array(diffs_stdalpha_mean)[:, mask_time[0]]
T_obs, clusters, c_pvals = run_cbp_test(data, 
                                        p_initial=0.05)


##############################################################################################
# add it to figure:
ax = axes[1]
plot_cbp_result_cda(ax, T_obs, clusters, c_pvals, 0.05, 
                         cbp_times=times[mask_time[0]], 
                         times_full=np.arange(-1.1, 2.3, 0.002))

# Save it: 
fpath = op.join(config.path_plots, 'TFR', part_epo)
helpers.chkmk_dir(fpath)
fname = op.join(fpath, 'classAlpha_Difference.png')
fig.savefig(fname, bbox_inches="tight")
##############################################################################################
##############################################################################################

#%% Export single trial data about lateral. difference in alpha power 
# for the CDA ROI during the sign cluster:

# For the retention intervall, we look at the 2nd cluster.
cluster_dict = {'name': 'retention', 
                't_start': times[mask_time[0]][clusters[1]][0], 
                't_stop': times[mask_time[0]][clusters[1]][-1]}

mean_pwrdiff_list = list()
for sub in sub_list:
    subID = 'VME_S%02d' % sub
    mean_pwrdiff_df = get_mean_pwrdiff_per_trial(subID, freqs_to_avg=np.arange(8,13), 
                                                 t_start= cluster_dict['t_start'], 
                                                 t_stop= cluster_dict['t_stop'])
    # Write subject data to disk:
    fpath = op.join(config.path_tfrs_summaries, pwr_style, cluster_dict['name'])
    helpers.chkmk_dir(fpath)
    fname = op.join(fpath, subID + f'-mean_pwrdiff_{cluster_dict["name"]}.csv')
    mean_pwrdiff_df.to_csv(fname, index=False)
    mean_pwrdiff_list.append(mean_pwrdiff_df)

mean_pwrdiff_all_df = pd.concat(mean_pwrdiff_list)
# Write subject data to disk:
fpath = op.join(config.path_tfrs_summaries, pwr_style, cluster_dict['name'], 
                'global_summary')
helpers.chkmk_dir(fpath)
fname = op.join(fpath, f'allsubjects-mean_pwrdiff_{cluster_dict["name"]}.csv')
mean_pwrdiff_all_df.to_csv(fname, index=False)


##############################################################################################
#%% compare the conditions: 



            

load_df_long = get_condition_pwrdiff_df('Load', cond_dict, sub_list_str)
ecc_df_long = get_condition_pwrdiff_df('Ecc', cond_dict, sub_list_str)


#Plot main effect Load:
fig, ax = plt.subplots(1, figsize=(12,4))
plot_main_eff('Load', cond_dict, load_df_long, ax)
ax.legend(title='Size Memory Array', labels=['2', '4'], loc=1, prop={'size': 9})

# Save it: 
fpath = op.join(config.path_plots, 'TFR', part_epo)
helpers.chkmk_dir(fpath)
fname = op.join(fpath, 'mainEff_load.png')
fig.savefig(fname, bbox_inches="tight")

# Plot main effect Ecc:
fig, ax = plt.subplots(1, figsize=(12,4))
plot_main_eff('Ecc', cond_dict, ecc_df_long, ax)
ax.legend(title='Eccentricity', labels=['4°', '9°', '14°'], 
          loc=1, prop={'size': 9}, ncol=3)

# Save it: 
fpath = op.join(config.path_plots, 'TFR', part_epo)
helpers.chkmk_dir(fpath)
fname = op.join(fpath, 'mainEff_ecc.png')
fig.savefig(fname, bbox_inches="tight")

##############################################################################################
##############################################################################################


##############################################################################################
#%% run rep-meas ANOVA on power: 

# building on code from: 
# https://mne.tools/dev/auto_tutorials/stats-source-space/plot_stats_cluster_time_frequency_repeated_measures_anova.html#sphx-glr-auto-tutorials-stats-source-space-plot-stats-cluster-time-frequency-repeated-measures-anova-py


#%%
df_list = list()
for load in cond_dict['Load']:
    for ecc in cond_dict['Ecc']:
        tfr_list = [load_avgtfr(subID, load+ecc, pwr_style, part_epo, 
                    baseline=(-(config.times_dict['cue_dur']+0.3), 
                            -(config.times_dict['cue_dur']+0.1)), 
                    mode='percent') for subID in sub_list_str]


        diffs_ = [get_lateralized_power_difference(tfr, 
                                                config.chans_CDA_dict['Contra'], 
                                                config.chans_CDA_dict['Ipsi']) for 
                tfr in tfr_list]
        for d in diffs_:
            dat_ = d.crop(0.2, 2.2)
            df_list.append(dat_.data.mean(axis=0))
    times = dat_.times
    freqs = dat_.freqs
#%%

# Setup parameters:
decim = 2
factor_levels = [2, 3]
effects = 'A*B'
n_levels = np.multiply(*factor_levels)
n_freqs = df_list[0].shape[-2]
n_times = df_list[0].shape[-1]
n_subs = int(len(df_list) / n_levels)

# Shape data matrix: subjects x effects x n_freqs*n_times
subtfr_array = np.asarray(df_list)
subtfr_mway_data = subtfr_array.reshape(n_levels, n_subs, n_freqs*n_times)
subtfr_mway_data = subtfr_mway_data.swapaxes(0, 1)

fvals, pvals = f_mway_rm(subtfr_mway_data, factor_levels, effects=effects)
effect_labels = ['Load', 'Ecc', 'Load x Ecc']

# Plot result:
for effect, sig, effect_label in zip(fvals, pvals, effect_labels):
    f = plt.figure()
    # show naive F-values in gray
    plt.imshow(effect.reshape(n_freqs, n_times), cmap=plt.cm.gray, extent=[times[0],
               times[-1], freqs[0], freqs[-1]], aspect='auto',
               origin='lower')
    # create mask for significant Time-frequency locations
    effect[sig >= 0.05] = np.nan
    plt.imshow(effect.reshape(n_freqs, n_times), cmap='RdBu_r', extent=[times[0],
               times[-1], freqs[0], freqs[-1]], aspect='auto',
               origin='lower')
    cb = plt.colorbar()
    cb.set_label(label='F', rotation=0)
    ytick_range = f.axes[0].get_ylim()
    ytick_vals = np.arange(*np.round(ytick_range), 2)
    f.axes[0].yaxis.set_ticks(ytick_vals)
    plt.xlabel('Time (ms)')
    plt.ylabel('Frequency (Hz)')
    plt.title(r"Induced lateralized power for '%s' (%s)" % (effect_label, 'CDA-ROI'))
    plt.show()


# Use CBP test to correct for multiple-comparisons correction:

# We need to do this separately for each effect:
effects='A:B'

def stat_fun(*args):
    return f_mway_rm(np.reshape(args, (len(sub_list), n_levels, n_freqs, len(times))), 
                     factor_levels=factor_levels,
                     effects=effects, return_pvals=False)[0]


# The ANOVA returns a tuple f-values and p-values, we will pick the former.
pthresh = 0.05  # set threshold rather high to save some time
f_thresh = f_threshold_mway_rm(len(sub_list), factor_levels, effects,
                               pthresh)
tail = 1  # f-test, so tail > 0
n_permutations = 1000  # Save some time (the test won't be too sensitive ...)
T_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_test(
    subtfr_mway_data, stat_fun=stat_fun, threshold=f_thresh, tail=tail, n_jobs=-2,
    n_permutations=n_permutations, buffer_size=None, out_type='mask')

# Plot it:
T_obs_plot = np.ones_like(T_obs) *np.nan
for c, p in zip(clusters, cluster_p_values):
    if p < .05: 
        T_obs_plot[c] = T_obs[c]

f = plt.figure()
for f_image, cmap in zip([T_obs, T_obs_plot], [plt.cm.gray, 'RdBu_r']):
    plt.imshow(f_image.reshape(20, 1951), cmap=cmap, extent=[times[0], times[-1],
               freqs[0], freqs[-1]], aspect='auto',
               origin='lower')
cb = plt.colorbar()
cb.set_label(label='F', rotation=0)
ytick_range = f.axes[0].get_ylim()
ytick_vals = np.arange(*np.round(ytick_range), 2)
f.axes[0].yaxis.set_ticks(ytick_vals)
plt.xlabel('Time (ms)')
plt.ylabel('Frequency (Hz)')
plt.title("Induced lateralized power for 'Eccentricity' (%s)\n"
          " cluster-level corrected (p <= 0.05)" % 'CDA-ROI')

plt.show()


#%% To look into the direction(s) of the effect, we extract the relevant data per condition:

obs_sign_dict = {cond: [] for cond in cond_dict['Ecc']}

for ecc in cond_dict['Ecc']:
    # tfr_list = [load_avgtfr(subID, ecc, pwr_style, part_epo, 
    #             baseline=(-(config.times_dict['cue_dur']+0.3), 
    #                     -(config.times_dict['cue_dur']+0.1)), 
    #             mode='mean') for subID in sub_list_str]

    diffs_ = [get_lateralized_power_difference(tfr, 
                                            config.chans_CDA_dict['Contra'], 
                                            config.chans_CDA_dict['Ipsi']) 
              for tfr in tfr_by_cond[ecc]]


    for d in diffs_:
        dat = d.data.mean(axis=0)
        dat = dat.reshape(len(d.freqs) * len(d.times))
        obs_sign = np.zeros_like(dat) 
        for c, p in zip(clusters, cluster_p_values):
            if p < .05: 
                obs_sign[c] = dat[c]
        obs_sign = obs_sign.reshape(n_freqs, len(times))

        obs_sign_dict[ecc].append(obs_sign)
#%%

# identify the clusters:

template = measurements.label(np.nan_to_num(obs_sign_dict['EccS'][0], nan=0))
# find largest cluster:
cluster_counts = np.bincount(template[0].flatten())[1:] #ignore first value (zeroes = background)
index_largest_cluster = cluster_counts.argmax() + 1  # add 1 to skip the zero in the template

haa = (template[0] == index_largest_cluster)

avgs = list()
dd = pd.DataFrame()
for ecc in cond_dict['Ecc']:
    avg = np.stack(obs_sign_dict[ecc], axis=-1)
    avg = avg[5, 1400, :]
    #avg = np.nanmean(avg, axis=0)
    #avg = np.nanmean(avg, axis = (0,1))
    avgs.append(avg)
    dd[ecc] = avg
#    abs_avg = np.nanmean(avg)

stata = [dd[x] for x in dd]
F, p = stats.f_oneway(*stata)

dd.melt().boxplot(by='variable')
ee = dd.melt()

len(obs_sign_dict['EccS'])


#%%


# run CBP test on TF data of the CDA ROI (sub x freq x times)
diffs_allfreqs = [d.data[:, :, :].mean(axis=(0)) for d in diffs_avgtfr_all]


T_obs, clusters, c_pvals = run_cbp_test(np.array(diffs_allfreqs), 
                                        p_initial=0.05)

vmax = np.max(np.abs(T_obs))
vmin = -vmax

# Create new stats image with only significant clusters
T_obs_plot = np.nan * np.ones_like(T_obs)
for c, p_val in zip(clusters, c_pvals):
    if p_val <= 0.05:
        T_obs_plot[c] = T_obs[c]


fig, ax = plt.subplots(1, figsize=(6,4))
plt.imshow(T_obs, cmap=plt.cm.gray,
           extent=[times[0], times[-1], freqs[0], freqs[-1]],
           aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
plt.imshow(T_obs_plot, cmap=plt.cm.RdBu_r,
           extent=[times[0], times[-1], freqs[0], freqs[-1]],
           aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
ytick_range = ax.get_ylim()
ytick_vals = np.arange(*np.round(ytick_range), 2)
ax.yaxis.set_ticks(ytick_vals)
ax.axvspan(0, 0.2, color='grey', alpha=0.5)
ax.axvspan(2.2, 2.3, color='grey', alpha=0.5)
ax.vlines((-0.8, 0, 0.2, 2.2), ymin=ytick_range[0], ymax=ytick_range[-1], 
          linestyles='dashed')
cb = plt.colorbar()
cb.set_label(label='T', rotation=0)
plt.xlabel('Time (ms)')
plt.ylabel('Frequency (Hz)')
plt.title(f'Contra - Ipsi (T stats)')

# Save it: 
fpath = op.join(config.path_plots, 'TFR', part_epo)
helpers.chkmk_dir(fpath)
fname = op.join(fpath, 'CBP_avgTFR_Diff_classAlpha_p001.png')
fig.savefig(fname, bbox_inches="tight")



# %%
