

# %% load libs:
from collections import defaultdict
from os import path as op
import json
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

from scipy import stats

import mne
from mne import EvokedArray
# from mne.epochs import concatenate_epochs
from mne.decoding import (SlidingEstimator,  # GeneralizingEstimator,
                          cross_val_multiscore, LinearModel, get_coef)
from mne.stats import permutation_cluster_1samp_test, f_mway_rm, f_threshold_mway_rm

from library import config, helpers


# %% Functions:

def get_epos(subID, epo_part, signaltype, condition, event_dict):
    if signaltype == 'uncollapsed':
        fname = op.join(config.path_rejepo, subID + '-' + epo_part +
                        '-postica-rejepo' + '-epo.fif')
    elif signaltype in ['collapsed', 'difference']:
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


def avg_time(data, step=25, times=None):
    orig_shape = data.shape
    n_fill = step - (orig_shape[-1] % step)
    fill_shape = np.asarray(orig_shape)
    fill_shape[-1] = n_fill
    fill = np.ones(fill_shape) * np.nan
    data_f = np.concatenate([data, fill], axis=-1)
    data_res = np.nanmean(data_f.reshape(*orig_shape[:2], -1, step), axis=-1)

    if times is not None:
        f_times = np.r_[times, [np.nan] * n_fill]
        n_times = np.nanmean(f_times.reshape(-1, step), axis=-1)
        return data_res, n_times
    else:
        return data_res


def batch_trials(epos, batch_size):
    n_trials = len(epos)
    n_batches = int(n_trials / batch_size)
    rnd_seq = np.arange(n_trials)
    np.random.shuffle(rnd_seq)
    rnd_seq = rnd_seq[:n_batches * batch_size]
    rnd_seq = rnd_seq.reshape(-1, batch_size)
    batches = [epos[b].average() for b in rnd_seq]
    return(batches)


def get_data(subID, epo_part, signaltype, conditions, event_dict,
             batch_size=1, smooth_winsize=1):
    epos_dict = defaultdict(dict)
    for cond in conditions:
        epos_dict[cond] = get_epos(subID,
                                   epo_part=epo_part,
                                   signaltype=signaltype,
                                   condition=cond,
                                   event_dict=event_dict)

    times = epos_dict[conditions[0]][0].copy().times

    # Setup data:
    if batch_size > 1:
        batches = defaultdict(list)
        for cond in conditions:
            batches[cond] = batch_trials(epos_dict[cond], batch_size)
            batches[cond] = np.asarray([b.data for b in batches[cond]])

        X = np.concatenate([batches[cond].data for cond in conditions], axis=0)
        n_ = {cond: batches[cond].shape[0] for cond in conditions}

    else:
        X = mne.concatenate_epochs([epos_dict[cond] for cond in conditions])
        X = X.get_data()
        n_ = {cond: len(epos_dict[cond]) for cond in conditions}

    if smooth_winsize > 1:
        X, times_n = avg_time(X, smooth_winsize, times=times)
    else:
        times_n = times

    y = np.r_[np.zeros(n_[conditions[0]]),
              np.concatenate([(np.ones(n_[conditions[i]]) * i)
                              for i in np.arange(1, len(conditions))])]

    return X, y, times_n


def decode(sub_list_str, conditions, epo_part='stimon', signaltype='collapsed',
           event_dict=config.event_dict, n_rep_sub=100, shuffle_labels=False,
           batch_size=10, smooth_winsize=5, save_single_rep_scores=False,
           save_scores=True, save_patterns=False):

    contrast_str = '_vs_'.join(conditions)
    scoring = 'roc_auc' # 'accuracy'
    cv_folds = 5


    clf = make_pipeline(StandardScaler(),
                        LinearModel(LogisticRegression(solver='liblinear',
                                                       penalty='l2',
                                                       random_state=42,
                                                       verbose=False)))

    se = SlidingEstimator(clf,
                          scoring=scoring,
                          n_jobs=-2,
                          verbose=0)

    sub_scores = list()
    sub_scores_per_rep = list()
    sub_coef = list()
    times_n = list()

    for sub in sub_list_str:
        print(f'### RUNING SUBJECT {sub}')
        all_scores = list()
        all_coef = list()
        for i in np.arange(n_rep_sub):
            X, y, times_n = get_data(sub,
                                     epo_part=epo_part,
                                     signaltype=signaltype,
                                     conditions=conditions,
                                     event_dict=event_dict,
                                     batch_size=batch_size,
                                     smooth_winsize=smooth_winsize)
            if shuffle_labels:
                np.random.shuffle(y)
            for i in np.unique(y):
                print(f'Size of class {i}: {np.sum(y == i)}\n')
            scores = cross_val_multiscore(se, X=X, y=y, cv=cv_folds, verbose=0)
            scores = np.mean(scores, axis=0)
            all_scores.append(scores)
            se.fit(X, y)
            coef = get_coef(se, 'patterns_', inverse_transform=True)
            all_coef.append(coef)

        sub_scores.append(np.asarray(all_scores).mean(axis=0))
        sub_coef.append(np.asarray(all_coef).mean(axis=0))

        # save shizzle:
        if save_single_rep_scores:
            if len(sub_scores_per_rep) == 0:
                sub_scores_per_rep = np.asarray(all_scores)
            else:
                sub_scores_per_rep = np.concatenate([sub_scores_per_rep,
                                                    np.asarray(all_scores)],
                                                    axis=0)

            fpath = op.join(config.path_decod_temp, contrast_str, 'single_rep_data')
            helpers.chkmk_dir(fpath)
            fname = op.join(fpath,
                            f'reps{n_rep_sub}_' \
                            f'swin{smooth_winsize}_batchs{batch_size}.npy')
            np.save(fname, sub_scores_per_rep)
            np.save(fname[:-4] + '__times' + '.npy', times_n)
            del(fpath, fname)

        # save accuracies:
        if save_scores:
            sub_scores_ = np.asarray(sub_scores)
            fpath = op.join(config.path_decod_temp, epo_part, signaltype, contrast_str, 'scores')
            helpers.chkmk_dir(fpath)
            fname = op.join(fpath, 'scores_per_sub.npy')
            np.save(fname, sub_scores_)
            np.save(fname[:-4] + '__times' + '.npy', times_n)
            del(fpath, fname)


        # save patterns:
        if save_patterns:
            sub_patterns = np.asarray(sub_coef)
            fpath = op.join(config.path_decod_temp, epo_part, signaltype, contrast_str, 'patterns')
            helpers.chkmk_dir(fpath)
            fname = op.join(fpath, 'patterns_per_sub.npy')
            np.save(fname, sub_patterns)
            np.save(fname[:-4] + '__times' + '.npy', times_n)
            del(fpath, fname)

        # save info:
        if save_scores or save_patterns or save_single_rep_scores:
            info_dict = {'n_rep_sub': n_rep_sub, 
                         'batch_size': batch_size, 
                         'smooth_winsize': smooth_winsize, 
                         'cv_folds': cv_folds, 
                         'scoring': scoring}
            fpath = op.join(config.path_decod_temp, epo_part, signaltype, contrast_str)
            fname = op.join(fpath, 'info.json')
            with open(fname, 'w+') as outfile:  
                json.dump(info_dict, outfile) 
            
    return sub_scores, sub_coef, times_n


def plot_score_per_factor(factor, data, scoring='roc_auc', sign_clusters=[], p_lvl=0.01, 
                          plt_dict=None, ax=None, n_boot=1000):

    
    sns.lineplot(x='time', 
                 y='score', 
                 hue=factor, 
                 data=data, 
                 n_boot=n_boot, 
                 palette=config.colors, 
                 ax=ax)
    ytick_range = ax.get_ylim()
    ax.set(xlim=(plt_dict['xmin'], plt_dict['xmax']), ylim=ytick_range)
    if scoring == 'roc_auc':
        scoring_str = 'ROC AUC'
    else: 
        scoring_str = scoring
    ax.set_ylabel(scoring_str)
    ax.set_xlabel('Time (s)')
    ax.axvspan(plt_dict['t_stimon'], plt_dict['t_stimon']+0.2, color='grey', alpha=0.3)
    ax.axvspan(plt_dict['t_stimon']+ 2.2, plt_dict['t_stimon'] + 2.5, color='grey', alpha=0.3)
    ax.vlines((plt_dict['t_stimon'], plt_dict['t_stimon']+0.2, plt_dict['t_stimon']+2.2),
              ymin=ytick_range[0], ymax=ytick_range[1], 
              linestyles='dashed')
    ax.hlines(0.5, xmin=plt_dict['xmin'], xmax=plt_dict['xmax'])
    p_lvl_str = 'p < .' + str(p_lvl).split('.')[-1]
    for sc in sign_clusters:
        xmin = sc[0]
        xmax = sc[-1]
        ax.hlines(ytick_range[0] + 0.05*np.ptp(ytick_range), xmin=xmin, xmax=xmax, color='purple', 
                 label=p_lvl_str)
    handles, labels = ax.get_legend_handles_labels()
    print(labels)
    n_sgn_clu = None if len(sign_clusters) <= 1 else -(len(sign_clusters)-1)
    ax.legend(handles=handles[0:n_sgn_clu], labels=labels[0:n_sgn_clu])



def run_cbp_test(data):
    # number of permutations to run
    n_permutations = 1000 
    # set initial threshold
    p_initial = 0.05
    # set family-wise p-value
    p_thresh = 0.05
    connectivity = None
    tail = 1.  # for one-sided test

    # set cluster threshold
    n_samples = len(data)
    threshold = -stats.t.ppf(p_initial / (1 + (tail == 0)), n_samples - 1)
    if np.sign(tail) < 0:
        threshold = -threshold

    cluster_stats = mne.stats.permutation_cluster_1samp_test(
        data, threshold=threshold, n_jobs=config.n_jobs, verbose=False, tail=tail,
        step_down_p=0.0005, adjacency=connectivity,
        n_permutations=n_permutations, seed=42, out_type='mask')

    T_obs, clusters, cluster_p_values, _ = cluster_stats
    return(T_obs, clusters, cluster_p_values)

# %% setup params:

mne.set_log_level('WARNING')


# plotting:
plt_dict = defaultdict(dict)
pp = {'t_stimon':  0,
      'xmin': -0.2,
      'xmax': 2.3}
plt_dict['stimon'] = pp

# structuring data:
sub_list = np.setdiff1d(np.arange(1, 28), config.ids_missing_subjects +
                        config.ids_excluded_subjects)               
sub_list_str = ['VME_S%02d' % sub for sub in sub_list]

event_dict = config.event_dict
cond_dict = {'Load': ['LoadLow', 'LoadHigh'],
             'Ecc': ['EccS', 'EccM', 'EccL']}



# %% Decode load across all eccentricities:

decod_results_load = defaultdict(dict)

conditions = ['LoadLow', 'LoadHigh']
contrast_str = '_vs_'.join(conditions)
sc_, pat_, ts_ = decode(sub_list_str, 
                        conditions=conditions,
                        epo_part='stimon', 
                        signaltype='collapsed',
                        event_dict=config.event_dict, 
                        n_rep_sub=10,
                        shuffle_labels=False,
                        batch_size=5,
                        smooth_winsize=10,
                        save_single_rep_scores=False,
                        save_patterns=False,
                        save_scores=False)
decod_results_load['acc'] = sc_
decod_results_load['patterns'] = pat_
decod_results_load['times'] = ts_


sc_, _, _ = decode(sub_list_str, 
                   conditions=conditions,
                   epo_part='stimon', 
                   signaltype='collapsed',
                   event_dict=config.event_dict, 
                   n_rep_sub=10,
                   shuffle_labels=True,
                   batch_size=5,
                   smooth_winsize=10,
                   save_single_rep_scores=False,
                   save_patterns=False,
                   save_scores=False)
decod_results_load['random'] = sc_

# %% Plot the results:
# Prepare data for plotting with seaborn:
epo_part = 'stimon'
signaltype = 'difference'

fpath = op.join(config.path_decod_temp, epo_part, signaltype, contrast_str, 'scores')
            
            # fname = op.join(fpath, 'scores_per_sub.npy')
            # np.save(fname, sub_scores_)
            # np.save(fname[:-4] + '__times' + '.npy', times_n)


times = decod_results_load['times']
acc = np.asarray(decod_results_load['acc'])
acc_df = pd.DataFrame(acc)
acc_df.columns = times
acc_df_long = acc_df.melt(var_name='time', value_name='score')  # put into long format
acc_df_long['decoding target'] = 'Load'

chance = np.asarray(decod_results_load['random'])
chance_df = pd.DataFrame(chance)
chance_df.columns = times
chance_df_long = chance_df.melt(var_name='time', value_name='score')  # put into long format
chance_df_long['decoding target'] = 'Random'

data_plot = pd.concat([acc_df_long, chance_df_long])

# %%

# run CBP:

data = np.asarray(decod_results_load['acc']) - np.asarray(decod_results_load['random'])
t_values, clusters, p_values = run_cbp_test(data)
p_val_cbp = 0.05
idx_sign_clusters = np.argwhere(p_values<p_val_cbp)
sign_cluster_times = [times[clusters[idx[0]]][[0,-1]] for idx in idx_sign_clusters]


# %%


# Plot it:
fig, ax = plt.subplots(1, figsize=(6,4))
plot_score_per_factor('decoding target', data=data_plot, sign_clusters=sign_cluster_times, p_lvl=p_val_cbp,
plt_dict=plt_dict['stimon'], n_boot=10, ax=ax)
handles, labels = ax.get_legend_handles_labels()
p_lvl_str = "$\it{p}$ < ." + str(p_val_cbp).split('.')[-1]
ax.legend(title='Decoding Target', 
          handles = handles,
          labels=['Load: 2 vs 4', 'chance', p_lvl_str], loc=1, prop={'size': 9})


# %% Plot patterns:


# get dummy epos file to get electrode locations
dummy_epos = get_epos('VME_S01', 'stimon', 'collapsed','LoadLow', event_dict)
sub_patterns = np.asarray(decod_results_load['patterns'])
# normalize them by l2 norm to allow fair average across subjects:
sub_patterns = sub_patterns / np.linalg.norm(sub_patterns, axis=1, ord=2, keepdims=True)
sub_patterns_avg = sub_patterns.mean(axis=0) 
# normalize per timebin
sub_patterns_avg = sub_patterns_avg / np.linalg.norm(sub_patterns_avg, axis=0, ord=2, keepdims=True)
sub_patterns_evo = EvokedArray(sub_patterns_avg, dummy_epos.info)
sub_patterns_evo.times = decod_results_load['times']
sub_patterns_evo.plot_topomap(times = [0.25, 0.55, 0.85, 1.15, 1.5, 2.0], scalings=1, units='', 
                                title=config.labels['Load'])



# %% decode load per eccentricity:

decod_results_load = defaultdict(dict)

for ecc in cond_dict['Ecc']:
    conditions = ['LoadLow' + ecc, 'LoadHigh' + ecc]
    contrast_str = '_vs_'.join(conditions)
    sc_, pat_, ts_ = decode(sub_list_str, 
                            conditions=conditions,
                            epo_part='stimon', 
                            signaltype='difference',
                            event_dict=config.event_dict, 
                            n_rep_sub=1,
                            batch_size=10,
                            smooth_winsize=5,
                            save_single_rep_scores=False,
                            save_patterns=True,
                            save_scores=True)
    decod_results_load[ecc]['acc'] = sc_
    decod_results_load[ecc]['patterns'] = pat_
    decod_results_load[ecc]['times'] = ts_

# %% Plot the results:


# Prepare data for plotting with seaborn:
results_df_list = list()
for ecc in cond_dict['Ecc']:
    times = decod_results_load[ecc]['times']
    acc = np.asarray(decod_results_load[ecc]['acc'])
    acc_df = pd.DataFrame(acc)
    acc_df.columns = times
    df = acc_df.melt(var_name='time', value_name='score')  # put into long format
    df['Ecc'] = ecc
    results_df_list.append(df)
data_plot = pd.concat(results_df_list)

# Plot it:
fig, ax = plt.subplots(1, figsize=(6,4))
plot_score_per_factor('Ecc', data=data_plot, plt_dict=plt_dict['stimon'], n_boot=10, ax=ax)
ax.legend(title='Eccentricity', labels=['4°', '9°', '14°'], loc=1, prop={'size': 9})

# %% Plot the corresponding patterns per eccentricity level:

for ecc in cond_dict['Ecc']:
# get dummy epos file to get electrode locations
    dummy_epos = get_epos('VME_S01', 'stimon', 'difference','LoadLow', event_dict)
    sub_patterns = np.asarray(decod_results_load[ecc]['patterns'])
    # normalize them by l2 norm to allow fair average across subjects:
    sub_patterns = sub_patterns / np.linalg.norm(sub_patterns, axis=1, ord=2, keepdims=True)
    sub_patterns_avg = sub_patterns.mean(axis=0) 
    # normalize per timebin
    sub_patterns_avg = sub_patterns_avg / np.linalg.norm(sub_patterns_avg, axis=0, ord=2, keepdims=True)
    sub_patterns_evo = EvokedArray(sub_patterns_avg, dummy_epos.info)
    sub_patterns_evo.times = decod_results_load['EccL']['times']
    sub_patterns_evo.plot_topomap(times = [0.25, 0.55, 0.85, 1.15], scalings=1, units='', 
                                  title=config.labels[ecc])


# %%
2+2
# %%
