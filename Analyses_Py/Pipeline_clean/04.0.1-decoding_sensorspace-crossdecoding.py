# %%
# %% load libs:
from collections import defaultdict
from os import path as op
import sys
import json
import numpy as np
import seaborn as sns

from sklearn.model_selection import check_cv, BaseCrossValidator, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
import multiprocessing as mp

from scipy import stats

import mne
# from mne.epochs import concatenate_epochs
from mne.decoding import (SlidingEstimator, GeneralizingEstimator,
                          cross_val_multiscore, LinearModel, get_coef)

from library import config, helpers

# %%

def get_epos(subID, epo_part, signaltype, condition, event_dict, picks_str):
    if signaltype == 'uncollapsed':
        fname = op.join(config.paths['03_preproc-rejectET'],
                        epo_part,
                        'cleaneddata',
                        f"{subID}-{epo_part}-rejepo-epo.fif")
    elif signaltype in ['collapsed', 'difference']:
        fname = op.join(config.paths['03_preproc-pooled'],
                        epo_part,
                        signaltype,
                        f"{subID}-{epo_part}-{signaltype}-epo.fif")
    else:
        raise ValueError(f'Invalid value for "signaltype": {signaltype}')
    epos = mne.read_epochs(fname, verbose=False)
    epos = epos.pick_types(eeg=True)

    # pick channel selection:
    if (picks_str is not None) and (picks_str != 'All'):
        roi_dict = mne.channels.make_1020_channel_selections(epos.info)
        picks = [epos.ch_names[idx] for idx in roi_dict[picks_str]]
        epos.pick_channels(picks, ordered=True)

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


def batch_trials(epos, batch_size, randomize=True, random_state=None):
    n_trials = len(epos)
    n_batches = int(n_trials / batch_size)
    rnd_seq = np.arange(n_trials)
    if randomize:
        rng = np.random.default_rng(random_state)
        rng.shuffle(rnd_seq)
    rnd_seq = rnd_seq[:n_batches * batch_size]
    rnd_seq = rnd_seq.reshape(-1, batch_size)
    batches = [epos[b].average() for b in rnd_seq]
    return(batches)


def shuffle_samples(data, conds, n_, random_state=None):
    """
    Shuffle samples in a NumPy array based on condition labels.

    Parameters
    ----------
    data : np.ndarray
        The NumPy array containing the samples to be shuffled along the first axis. The data is expected to be sorted by condition (along the first axis). Otherwise this will create wrong results!
    conds : list of str
        The list of condition labels corresponding to each sample.
    n_ : dict
        A dictionary mapping condition labels to the number of samples for each condition.
    random_state : int, optional
        Seed to use for random number generation. If not specified, the default NumPy generator will be used.

    Returns
    -------
    np.ndarray
        A new NumPy array containing the shuffled samples.
    """
    
    # check inputs:
    if not isinstance(n_, dict):
        raise TypeError(f'n_ must be a dict, not {type(n_)}')
    # check if all conditions are present in n_:
    if not all([cond in n_.keys() for cond in conds]):
        raise ValueError(f'All conditions must be present in n_.')

    shuffled_idx = np.array([], dtype=int)
    for i, cond in enumerate(conds):
        start = int(np.sum([n_[c] for c in conds[:i]]))
        idx = np.arange(start, start+n_[cond])
        rng = np.random.default_rng(random_state)
        rng.shuffle(idx)
        shuffled_idx = np.concatenate([shuffled_idx, np.array(idx)], dtype=int)
    data_shuffled = data[shuffled_idx]
    return data_shuffled


def get_data(subID, epo_part, signaltype, conditions, event_dict,
             batch_size=1, smooth_winsize=1, picks_str=None, randomize=True, random_state=None):
    epos_dict = defaultdict(dict)
    for cond in conditions:
        epos_dict[cond] = get_epos(subID,
                                   epo_part=epo_part,
                                   signaltype=signaltype,
                                   condition=cond,
                                   event_dict=event_dict,
                                   picks_str=picks_str)

    times = epos_dict[conditions[0]][0].copy().times
    info = epos_dict[conditions[0]][0].info

    # Setup data:
    if batch_size > 1:
        batches = defaultdict(list)
        for cond in conditions:
            batches[cond] = batch_trials(epos_dict[cond], batch_size, randomize=randomize, random_state=random_state)
            batches[cond] = np.asarray([b.data for b in batches[cond]])

        X = np.concatenate([batches[cond].data for cond in conditions], axis=0)
        n_ = {cond: batches[cond].shape[0] for cond in conditions}

    else:
        X = mne.concatenate_epochs([epos_dict[cond] for cond in conditions])
        X = X.get_data()
        n_ = {cond: len(epos_dict[cond]) for cond in conditions}

    if randomize:
        X = shuffle_samples(X, conditions, n_, random_state=random_state)

    if smooth_winsize > 1:
        X, times_n = avg_time(X, smooth_winsize, times=times)
    else:
        times_n = times

    y = np.r_[np.zeros(n_[conditions[0]]),
              np.concatenate([(np.ones(n_[conditions[i]]) * i)
                              for i in np.arange(1, len(conditions))])]

    return X, y, times_n, info

# %%
def concat_train_test(
    subID,
    epo_part,
    signaltype,
    conditions_target,
    condition_train,
    condition_test,
    event_dict,
    batch_size=3,
    smooth_winsize=10,
    picks_str=None,
    randomize=True,
    random_state=None,
):
    if condition_train != condition_test:
        conditions_train = [f"{condition_train}{c}" for c in conditions_target]
        conditions_test = [f"{condition_test}{c}" for c in conditions_target]
        X_train_all, y_train_all, times_n, info = get_data(
            subID, epo_part, signaltype, conditions_train, event_dict,batch_size=batch_size, smooth_winsize=smooth_winsize, picks_str=picks_str, randomize=randomize, random_state=random_state
        )
        X_test_all, y_test_all, _, _ = get_data(
            subID, epo_part, signaltype, conditions_test, event_dict,
            batch_size=batch_size, smooth_winsize=smooth_winsize, picks_str=picks_str, randomize=randomize, random_state=random_state
        )

        X = np.concatenate([X_train_all, X_test_all], axis=0)
        y = np.concatenate([y_train_all, y_test_all], axis=0)
        groups = np.concatenate([len(X_train_all) * [0], len(X_test_all) * [1]])
    else:
        conditions_traintest = [f"{condition_train}{c}" for c in       
                                conditions_target]
        X, y, times_n, info = get_data(
            subID, epo_part, signaltype, conditions_traintest, event_dict,
            batch_size=batch_size, smooth_winsize=smooth_winsize, picks_str=picks_str, randomize=randomize, random_state=random_state
        )
        groups = None
    return X, y, groups, times_n, info


# %%


class CrossDecodSplitter(BaseCrossValidator):
    def __init__(self, n_splits):
        self.n_splits = n_splits
        
    def split(self, X, y, groups):
        # throw error if groups does not contain exactly 2 unique values
        if len(np.unique(groups)) != 2:
            raise ValueError('groups must contain exactly 2 unique values')
        # thow error if groups is not sorted. sorting should be done in a way that groups[0] always stays the first value after sorting
        groups_sorted = np.sort(groups)
        if groups_sorted[0] != groups[0]:
            groups_sorted = np.flip(groups_sorted)
        if not np.all(groups_sorted == groups):
            raise ValueError('groups vector must be sorted')
            

        idx_0 = np.where(groups == groups[0])[0]
        idx_1 = np.where(groups == groups[-1])[0]

        idx_cv = StratifiedKFold(n_splits=self.n_splits)

        for fold_0, fold_1 in zip(idx_cv.split(idx_0, y[idx_0]),
                                  idx_cv.split(idx_1, y[idx_1])):
            yield idx_0[fold_0[0]], idx_1[fold_1[1]]
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

# %%
def decode_core(X, y, groups, info,
                scoring='roc_auc',
                temp_gen=False,
                n_cv_folds=5,
                cv_random_state=None):
    

    clf = make_pipeline(mne.decoding.Scaler(info),
                    mne.decoding.Vectorizer(),
                    LinearModel(
                        LogisticRegression(solver='liblinear',
                                            random_state=42,
                                    verbose=False)))

    if temp_gen:
        gen_str = 'gen_temp'
        se = GeneralizingEstimator(clf,
                                   scoring=scoring,
                                   n_jobs=15,
                                   verbose=0)
    else:
        gen_str = ''
        se = SlidingEstimator(clf,
                              scoring=scoring,
                              n_jobs=15,
                              verbose=0)
    if groups is None:
        cv = StratifiedKFold(n_splits=n_cv_folds)
    else:
        cv = CrossDecodSplitter(n_splits=n_cv_folds)
    scores = cross_val_multiscore(se, X, y, cv=cv, groups=groups, n_jobs=n_cv_folds)

    se.fit(X, y)
    patterns = get_coef(se, 'patterns_', inverse_transform=True)

    return scores, patterns

    


def gen_save_path(contrast_str,
                  epo_part='stimon',
                  signaltype='collapsed',
                  gen_str='',
                  scoring='roc_auc',
                  picks_str=None,
                  labels_shuffled=False,
                  cross_decod=False,
                  crossing_str='',
                 ):
    shuf_labs = 'labels_shuffled' if labels_shuffled else ''
    cross_decod_str = 'cross_decod_ecc' if cross_decod else ''
    picks_str = 'picks' if picks_str is not None else ''

    path_save = op.join(config.paths['06_decoding-sensorspace'], epo_part,
                        signaltype, contrast_str, gen_str, cross_decod_str, crossing_str, scoring, picks_str, shuf_labs)
    return path_save


def save_scores(subID, scores, times_n, path_save):
    fpath = op.join(path_save, 'scores')
    helpers.chkmk_dir(fpath)
    fname = op.join(fpath, f'{subID}-scores_per_sub.npy')
    np.save(fname, scores)
    np.save(fname[:-4] + '__times' + '.npy', times_n)
    

def save_patterns(subID, patterns, times_n, path_save):
    fpath = op.join(path_save, 'patterns')
    helpers.chkmk_dir(fpath)
    fname = op.join(fpath, f'{subID}-patterns_per_sub.npy')
    np.save(fname, patterns)
    np.save(fname[:-4] + '__times' + '.npy', times_n)


def save_info(subID, info_dict, path_save):
    fpath = path_save
    fname = op.join(fpath, f'{subID}-info.json')
    with open(fname, 'w+') as outfile:
        json.dump(info_dict, outfile)


def save_single_rep_scores(subID, sub_scores_per_rep, times_n, path_save):
    fpath = op.join(path_save, 'single_rep_data')
    helpers.chkmk_dir(fpath)
    fname = op.join(fpath,
                    f'{subID}-'
                    f'reps{n_rep_sub}_'
                    f'swin{smooth_winsize}_batchs{batch_size}.npy')
    np.save(fname, sub_scores_per_rep)
    np.save(fname[:-4] + '__times' + '.npy', times_n)




# %%
def run_decoding(c_train, c_test, subID, shuffle_labels=False):
    scores_per_rep = []
    patterns_per_rep = []
    for rep in range(n_rep_sub):
        X, y, groups, times_n, info = concat_train_test(
                    subID=subID,
                    epo_part="stimon",
                    signaltype="collapsed",
                    conditions_target=["LoadLow", "LoadHigh"],
                    condition_train=c_train,
                    condition_test=c_test,
                    event_dict=config.event_dict,
                    batch_size=10,
                    smooth_winsize=10,
                    picks_str=None,
                    randomize=True,
                    random_state=42 + rep,
                )

        if shuffle_labels:
            groups_ = groups if groups is not None else np.zeros(shape=y.shape)
            groups_uniq, n_per_group = np.unique(groups_, return_counts=True)
            n_per_group = {k: v for k, v in zip(groups_uniq, n_per_group)}
            y = shuffle_samples(y, groups_uniq, n_per_group)
            print(f"y shape: {y.shape}")
            print(y)

        scores, patterns = decode_core(
            X, y, groups, info, scoring=scoring, n_cv_folds=5
        )
        scores_per_rep.append(np.mean(scores, axis=0))
        patterns_per_rep.append(patterns)
    scores_sub = np.mean(np.array(scores_per_rep), axis=0)
    patterns_sub = np.mean(np.array(patterns_per_rep), axis=0)

    return (scores_sub, patterns_sub, times_n, subID, c_train, c_test)



# %%

if __name__ == '__main__':

    # Set up parameters:
    batch_size = 10
    smooth_winsize = 10
    n_rep_sub = 100
    n_cv_folds = 5
    scoring = "roc_auc"

    shuffle_labels = True

    sub_list = np.setdiff1d(
        np.arange(1, 28), config.ids_missing_subjects + config.ids_excluded_subjects
    )
    sub_list_str = ["VME_S%02d" % sub for sub in sub_list]

    # when running on the cluster we want parallelization along the subject dimension
    if not helpers.is_interactive(): 
        helpers.print_msg('Running Job Nr. ' + sys.argv[1])
        job_nr = int(float(sys.argv[1]))
        sub_list_str = [sub_list_str[job_nr]]

    scores_all = defaultdict(list)
    patterns_all = defaultdict(list)

    pool = mp.Pool()
    results = []
    for c_train in ["EccS", "EccM", "EccL"]:
        for c_test in ["EccS", "EccM", "EccL"]: 
            for subID in sub_list_str:
                print(f"Running {subID} ... train: {c_train} test: {c_test}")
                result = pool.apply_async(run_decoding,
                                          args=(c_train,
                                                c_test,
                                                subID,    
                                                shuffle_labels))
                results.append(result)
    pool.close()
    pool.join()

    for result in results:
        scores_sub, patterns_sub, times_n, subID, c_train, c_test = result.get()
        scores_all[f"train_{c_train}-test_{c_test}"].append(scores_sub)
        patterns_all[f"train_{c_train}-test_{c_test}"].append(patterns_sub)
        path_save = gen_save_path(
                contrast_str="LoadLow_vs_LoadHigh",
                epo_part="stimon",
                signaltype="collapsed",
                gen_str="",
                scoring="roc_auc",
                picks_str=None,
                labels_shuffled=shuffle_labels,
                cross_decod=True,
                crossing_str=f"train_{c_train}-test_{c_test}",
            )
        save_scores(subID, scores_sub, times_n, path_save)
        save_patterns(subID, patterns_sub, times_n, path_save)

        info_dict = {
            "n_rep_sub": n_rep_sub,
            "batch_size": batch_size,
            "smooth_winsize": smooth_winsize,
            "cv_folds": n_cv_folds,
            "scoring": scoring,
        }
        save_info(subID, info_dict, path_save)

# %%



