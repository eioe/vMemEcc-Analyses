

#%% load libs:
from collections import defaultdict
from os import path as op
from mne.decoding.base import LinearModel
from mne.epochs import concatenate_epochs
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

import mne
from mne import EvokedArray
from mne.decoding import (SlidingEstimator, GeneralizingEstimator,
                          cross_val_multiscore, LinearModel, get_coef)

from library import config, helpers


#%% Functions:

def get_epos(subID, epo_part cond, event_dict):
    fname = op.join(config.path_rejepo, subID + '-' + epo_part + '-postica-rejepo' + '-epo.fif')
                    #'difference', subID + '-epo.fif')
    epos = mne.read_epochs(fname, verbose=False)
    epos = epos.pick_types(eeg=True)
    uppers = [l.isupper() for l in cond]
    if (np.sum(uppers) > 2): 
        cond_1 = cond[:np.where(uppers)[0][2]]
        cond_2 = cond[np.where(uppers)[0][2]:]
        selection = epos[event_dict[cond_1]][event_dict[cond_2]]
    else:
        selection = epos[event_dict[cond]]
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


def get_data(subID, conditions, batch_size=1, smooth_winsize=1):
    epos_dict = defaultdict(dict)
    for cond in conditions:
        epos_dict[cond] = get_epos(subID,
                                   'stimon',
                                   cond,
                                   event_dict) 

    times = epos_dict[cond][0].copy().times

    # Setup decoding pipeline:
    if batch_size > 1:
        batches = defaultdict(list)
        for cond in conditions:
            batches[cond] = batch_trials(epos_dict[cond], 15)
            batches[cond] = np.asarray([b.data for b in batches[cond]])

        X = np.concatenate([batches[cond].data for cond in conditions], axis=0)
        n_ = {cond: batches[cond].shape[0] for cond in conditions}

    else:
        X = mne.concatenate_epochs([epos_dict[cond] for cond in conditions])
        X = X.get_data()
        n_ = {cond: len(epos_dict[cond]) for cond in conditions}

    if smooth_winsize > 1:
        X, times_n = avg_time(X, 25, times=times)
    else:
        times_n = times

    y = np.r_[np.zeros(n_[conditions[0]]),
              np.concatenate([(np.ones(n_[conditions[i]]) * i) 
                              for i in np.arange(1, len(conditions))])]

    return X, y, times_n


#%% setup params:


# plotting:
plt_dict = defaultdict(dict)
pp = {'t_stimon':  0, 
      'xmin': -0.2, 
      'xmax': 2.3}
plt_dict['stimon'] = pp

# structuring data:
event_dict = config.event_dict
cond_dict = {'Load': ['LoadLow', 'LoadHigh'], 
             'Ecc': ['EccS', 'EccM', 'EccL']}


#%% load data:


sub_list = np.setdiff1d(np.arange(1, 28), config.ids_missing_subjects +
                        config.ids_excluded_subjects)               
sub_list_str = ['VME_S%02d' % sub for sub in sub_list]


epo_part = 'stimon'
conditions = ['LoadLowEccS', 'LoadHighEccS']
contrast_str = '_vs_'.join(conditions)
batch_size = 10
smooth_winsize = 10
n_rep_sub = 1
save_single_rep = False
save_patterns = False

clf = make_pipeline(StandardScaler(),
                    LinearModel(LogisticRegression(solver='liblinear',
                                                   penalty='l2', 
                                                   random_state=42)))

se = SlidingEstimator(clf,
                      scoring='accuracy',
                      n_jobs=-2,
                      verbose='warning')

sub_results = list()
sub_coef = list()

for sub in sub_list_str:
    count = sub_list_str.index(sub)
    print(f'### RUNING SUBJECT {sub}')
    all_scores = list()
    all_coef = list()
    for i in np.arange(n_rep_sub):
        X, y, times_n = get_data(sub,
                                 conditions=conditions, #cond_dict['Load']
                                 batch_size=batch_size,
                                 smooth_winsize=smooth_winsize)
        scores = cross_val_multiscore(se, X=X, y=y, cv=5, verbose='warning')
        scores = np.mean(scores, axis=0)
        all_scores.append(scores)
        # get 
        se.fit(X, y)
        coef = get_coef(se, 'patterns_', inverse_transform=True)
        all_coef.append(coef)

    avg_scores = np.asarray(all_scores).mean(axis=0)
    sub_results.append(avg_scores)
    sub_coef.append(np.asarray(all_coef).mean(axis=0))

    # save shizzle:
    if save_single_rep:
        if count == 0:
            sub_scores_per_rep = np.asarray(all_scores)
            #sub_scores_per_rep = np.expand_dims(sub_scores_per_rep, 0)
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
        del(fpath, fname)

    # save patterns:
    if save_patterns:
        sub_patterns = np.asarray(sub_coef)
        fpath = op.join(config.path_decod_temp, contrast_str, 'patterns')
        helpers.chkmk_dir(fpath)
        fname = op.join(fpath, 'patterns_per_sub.npy')
        np.save(fname, sub_patterns)
        del(fpath, fname)
    

#%%

sub_avg = np.mean(np.asarray(sub_results), axis=0)
sub_res = np.asarray(sub_results)

# Plot
fig, ax = plt.subplots()
ax.plot(times_n, sub_avg, label='score')
ax.plot(times_n, np.swapaxes(sub_res, 1, 0), alpha=0.1)
ax.axhline(1. / len(np.unique(y)), color='k', linestyle='--', label='chance')
ax.set_xlabel('Times')
ax.set_ylabel('accuracy')  
ax.legend()
ax.axvline(.0, color='k', linestyle='-')
ax.set_title(f'Sensor space decoding: {" vs. ".join(conditions)}')

# %%
subID = 'VME_S01'
epos = get_epos(subID, 'stimon', 'LoadLow', event_dict)
sub_coef_avg = np.asarray(sub_coef).mean(axis=0)
evo = EvokedArray(sub_coef_avg, epos.info)
evo.times = times_n
evo.plot_topomap(times = [0.024, 0.074, 0.124, 0.174, 0.224, 0.274, 0.324, 0.374, 0.424])
evo.plot_joint(times = [0.024, 0.074, 0.174, 0.224, 0.274, 0.324, 0.374, 0.424, 0.574])
evo.plot_topomap(times = np.arange(0.574, 2.074, 0.1))
evo.plot_topomap(times = np.arange(-0.574, -0.24, 0.1))
# %%
