#!/usr/bin/env python
# coding: utf-8
# %%
import os
import os.path as op
import sys
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from datetime import datetime
from scipy import stats
from scipy.ndimage import measurements
import warnings

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import (
    cross_val_score,
    GridSearchCV,
    StratifiedKFold
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder

import mne
from mne.stats import (
    f_mway_rm,
    f_threshold_mway_rm,
    permutation_cluster_1samp_test
)
from mne.decoding import (
    cross_val_multiscore,
    CSP,
    GeneralizingEstimator
)
from library import helpers, config


def get_epos(subID, part_epo, signaltype, condition, event_dict,
             picks_str=None):
    """Load a set of specified epochs.
    
     Parameters
    ----------
    subID : str
        Subject identifier (eg, 'VME_S05')
    part_epo : str
        Part of the epoch. One of: 'fulllength', 'cue', 'stimon'
    signaltype: str
        Processing state of the sensor signal. One of:
            'collapsed': electrode positions flipped for cue left trials
            'uncollapsed': normal electrode positions,
            'difference': difference signal: contra minus ipsilateral
    condition: str
        Experimental condition. Combination of 'Ecc' and 'Load' (eg, 'LoadLow'
        or 'LoadLowEccS')
    event_dict: dict
        Dictionnary explaining the event codes. Normally this can be grabbed
        from config.event_dict
    picks_str: str
        Predefined selection, has to be either 'Left', 'Right', 'Midline' or
        'All'; None (default) is thesame as 'All'

    Returns
    -------
    mne.Epochs
        Array of selected epochs.
    """

    if signaltype == 'uncollapsed':
        fname = op.join(config.paths['03_preproc-rejectET'],
                        part_epo,
                        'cleaneddata',
                        f'{subID}-{part_epo}-rejepo-epo.fif')
    elif signaltype in ['collapsed']:
        fname = op.join(config.paths['03_preproc-pooled'],
                        part_epo,
                        signaltype,
                        f'{subID}-{part_epo}-{signaltype}-epo.fif')
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


def get_sensordata(subID, part_epo, signaltype, conditions, event_dict,
                   picks_str=None):
    """Load a set of specified epochs for classification.

     Parameters
    ----------
    subID : str
        Subject identifier (eg, 'VME_S05')
    part_epo : str
        Part of the epoch. One of: 'fulllength', 'cue', 'stimon'
    signaltype: str
        Processing state of the sensor signal. One of:
            'collapsed': electrode positions flipped for cue left trials
            'uncollapsed': normal electrode positions,
            'difference': difference signal: contra minus ipsilateral
    conditions: list
        List of experimental conditions. Combination of 'Ecc' and 'Load'
        (eg, 'LoadLow' or 'LoadLowEccS')
    event_dict: dict
        Dictionnary explaining the event codes. Normally this can be grabbed
        from config.event_dict
    picks_str: str
        Predefined selection, has to be either 'Left', 'Right', 'Midline' or
        'All'; None (default) is thesame as 'All'

    Returns
    -------
    X_epos: Epochs
        Array of selected epochs, sorted by class (starting with class '0').
    y: list
        Sorted list of labels.
    times_n: array, 1d
        Times of the samples within the single epoch.
    """

    epos_dict = defaultdict(dict)
    for cond in conditions:
        epos_dict[cond] = get_epos(subID,
                                   part_epo=part_epo,
                                   signaltype=signaltype,
                                   condition=cond,
                                   event_dict=event_dict,
                                   picks_str=picks_str)

    times = epos_dict[conditions[0]][0].copy().times

    # Setup data:
    X_epos = mne.concatenate_epochs([epos_dict[cond] for cond in conditions])
    n_ = {cond: len(epos_dict[cond]) for cond in conditions}

    times_n = times

    y = np.r_[np.zeros(n_[conditions[0]]),
              np.concatenate([(np.ones(n_[conditions[i]]) * i)
                              for i in np.arange(1, len(conditions))])]

    return X_epos, y, times_n


def decode(sub_list_str, conditions, event_dict, reps=1, scoring='roc_auc',
           t_min=-0.5, t_max=2.5, min_freq=6, max_freq=26, n_freqs=10,
           w_size=0.5, n_cycles=None, w_overlap=0.5, pwr_style='',
           reg_csp='ledoit_wolf', n_components=6, n_cv_folds=5,
           shuffle_labels=False, save_scores=True,
           save_csp_patterns=True, overwrite=False, part_epo='stimon',
           signaltype='collapsed', picks_str=None):
    """Apply CSP and LDA to perform binary classification from (the power) of
    epoched data.

    Original code from:
    https://mne.tools/stable/auto_examples/decoding/plot_decoding_csp_timefreq.html#
    sphx-glr-auto-examples-decoding-plot-decoding-csp-timefreq-py


    Parameters
    ----------
    sub_list_str : list of str
        List of subject identifiers (eg, 'VME_S05')
    conditions: list
        List of experimental conditions that shall be compared/decoded.
        Combination of 'Ecc' and 'Load' (eg, 'LoadLow' or 'LoadLowEccS')
    event_dict: dict
        Dictionnary explaining the event codes. Normally this can be grabbed
        from config.event_dict
    reps: int
        Number of repetions for the CV procedure.
    scoring: str
        Scoring metric to be used; 'roc_auc' (default), 'accuracy', or
        'balanced_accuracy'
    t_min: float
        Start of the epoch, relative to the time of stimulus onset
        (default: -0.5)
    t_max: float
        End of the epoch, relative to the time of stimulus onset (default: 2.5)
    min_freq: float, int
        Lower bound of lowest freq band to be used (default: 6)
    max_freq: float, int
        Upper bound of highest freq band to be used (default: 26)
    n_freqs:
        Number of freq bands that the interval between min_freq and max_freq
        is split into (default: 10)
    w_overlap: float
        Specifies how much the sliding time windows used for feature
        extraction will overlap (e.g., 0.5 -> 50% overlap). Default: 0.5
        [value in [0,1]; 0: no overlap, 1: full overlap]
    w_size: float
        Width of the sliding window (in s). If `None, `n_cycles`
        needs to be specified. Default: 0.5;
    n_cycles: int
        Allows to specify the width of the sliding window as a function of the
        frequency to keep the same number of oscillations in the window for
        each frequency. This leads to windows of different length for different
        frequencies though.
        If `None`, `w_size` needs to be specified. Default: `None`
    pwr_style:
        If set to 'induced', calculate induced power. Otherwise nothing
        happens.
        Saves induced power to separate subfolder.
    reg_csp: float, str
        Regularization approach to be used when fitting the CSP model. Can be a
        float in [0;1] (for l2 regularization) or the str 'ledoit_wolf' for the
        analytical solution. (default: 'ledoit_wolf')
    n_components: int
        Number of CSP components to be used as features for the LDA.
    n_cv_folds: int
        Number of folds of the CV. Default: 5
    shuffle_labels: bool
        Shuffle the labels to produce a null distribution.
    save_scores: bool, optional
        Shall the decoding scores be written to disk? (default is True).
    save_patterns: bool, optional
        Shall the CSP patterns be written to disk? (default is True).
    overwrite : bool
        Overwrite existing folders (True) or append current datetime to
        foldername (False). (default: False)
    part_epo : str, optional
        Part of the epoch. One of: 'fulllength', 'cue', 'stimon'
        (default is 'stimon').
    signaltype: str
        Processing state of the sensor signal. One of:
            'collapsed': electrode positions flipped for cue left trials
            'uncollapsed': normal electrode positions,
            'difference': difference signal: contra minus ipsilateral
            (default is 'collapsed'.)
    picks_str: str
        Predefined selection, has to be either 'Left', 'Right', 'Midline' or
        'All'; None (default) is thesame as 'All'

    Returns
    -------
    tf_scores_list: list
        list of 2d arrays (freq x time) with the decoding scores per subject
    centered_w_times: list
        list with the times around which the decoding windows were centered.
    """

    contrast_str = '_vs_'.join(conditions)

    cv_folds = n_cv_folds
    n_components = n_components
    reg = reg_csp  # 0.4  # 'ledoit_wolf'

    csp = CSP(n_components=n_components,
              # reg=reg,
              log=True, norm_trace=False,
              component_order='alternate')
    clf = make_pipeline(csp, LinearDiscriminantAnalysis())

    parameters = {
        "csp__reg": reg_csp  # np.logspace(-3, 0, 10)
    }

    # Classification & time-frequency parameters
    tmin = t_min
    tmax = t_max

    # Get datetime identifier for uniqure folder names (if not overwriting):
    datetime_str = datetime.today().strftime('%Y-%m-%d-%H-%M')

    # Assemble list of frequency range tuples
    freqs = np.linspace(min_freq, max_freq, n_freqs + 1)
    # make freqs list of tuples:
    freq_ranges = list(zip(freqs[:-1], freqs[1:]))

    # Setup list of seeds for the repetitions:
    np.random.seed(seed=42)
    rep_seeds = np.random.choice(range(10 * reps), reps)

    if ((n_cycles is not None) and (w_size is None)):
        # Infer window spacing from the max freq and number of cycles to
        # avoid gaps
        window_spacing = (n_cycles / np.max(freqs) / 2.)
        centered_w_times = np.arange(tmin, tmax, window_spacing)[1:]
    elif (((w_size is not None)) and (n_cycles is None)):
        assert 0 <= float(w_overlap or -1) < 1, f'Invalid value for \
                                                  w_overlap: {w_overlap}'
        step_size = w_size * (1 - w_overlap)
        centered_w_times = np.arange(tmin + (w_size / 2.),
                                     tmax - (w_size / 2) + 0.001,
                                     step_size)
    else:
        raise ValueError('Invalid combination of values for w_size and \
                         n_cylces. Exactly one must be None.')

    n_windows = len(centered_w_times)

    tf_scores_list = list()
    tf_patterns_list = list()
    completed_subs = list()
    for subID in sub_list_str:
        part_epo = part_epo

        print(f'Running {subID}')

        sub_folder = subID

        if shuffle_labels:
            shuf_labs = 'labels_shuffled'
        else:
            shuf_labs = ''

        if reg is not None:
            if isinstance(reg, float):
                reg_str = 'shrinkage'+str(reg)
            elif isinstance(reg, str):
                reg_str = reg
            elif isinstance(reg, list):
                reg_str = 'shrinkageCV'
        else:
            reg_str = ''

        if picks_str is not None:
            picks_str_folder = picks_str
        else:
            picks_str_folder = ''
        if not overwrite:
            subject_dir = op.join(config.paths['06_decoding-csp'],
                                  part_epo,
                                  signaltype,
                                  contrast_str,
                                  scoring,
                                  reg_str,
                                  shuf_labs,
                                  sub_folder)
            if op.exists(subject_dir):
                print("Output directory already exists; skipping subject.")
                return

        ######################################################################

        X_epos, y, t = get_sensordata(subID, part_epo, signaltype, conditions,
                                      event_dict, picks_str)

        if pwr_style == 'induced':
            X_epos = X_epos.subtract_evoked()

        n_channels = len(X_epos.ch_names)
        # init scores
        tf_scores = np.zeros((n_freqs, n_windows))
        tf_scores_tmp = np.zeros((reps, n_freqs, n_windows))
        tf_patterns = np.zeros((n_components, n_channels, n_freqs, n_windows))

        # Loop through each frequency range of interest
        for freq, (fmin, fmax) in enumerate(freq_ranges):

            print(f'Freq. {freq} of {len(freq_ranges)}')

            if (w_size is None):
                # Infer window size based on the frequency being used (default
                # behavuior is to use a fixed w_size)
                w_size = n_cycles / ((fmax + fmin) / 2.)  # in seconds

            # Apply band-pass filter to isolate the specified frequencies
            X_epos_filter = X_epos.copy().filter(fmin, fmax, n_jobs=1,
                                                 fir_design='firwin')

            # Roll covariance, csp and lda over time
            for t, w_time in enumerate(centered_w_times):

                # Center the min and max of the window
                w_tmin = w_time - w_size / 2.
                w_tmax = w_time + w_size / 2.

                # Crop data into time-window of interest
                X = X_epos_filter.copy().crop(w_tmin, w_tmax).get_data()

                # Run repeated CV to estimate decoding score:
                for rep, rand_state in enumerate(rep_seeds):
                    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                                         random_state=rand_state)

                    if shuffle_labels:
                        np.random.seed(rand_state)
                        np.random.shuffle(y)

                    # Set up grid search:
                    cclf = GridSearchCV(clf, param_grid=parameters, cv=cv, verbose=False)

                    # Save mean scores over folds for each frequency and time
                    # window for this repetition
                    tf_scores_tmp[rep, freq, t] = np.mean(cross_val_score(estimator=cclf,
                                                                          X=X,
                                                                          y=y,
                                                                          scoring=scoring,
                                                                          cv=cv,
                                                                          n_jobs=-2,
                                                                          verbose=False),
                                                           axis=0)
                if save_csp_patterns:
                    # get CSP patterns - fitted to all data:
                    csp.fit(X, y)
                    patterns_ = getattr(csp, 'patterns_')
                    tf_patterns[:, :, freq, t] = patterns_[:n_components, :]

        tf_scores = tf_scores_tmp
        tf_scores_list.append(tf_scores)
        tf_patterns_list.append(tf_patterns)

        # save info:
        if (save_scores or save_csp_patterns):
            completed_subs.append(subID)
            info_dict = {'subs': completed_subs,
                         'tmin': tmin,
                         'tmax': tmax,
                         'n_cycles': n_cycles,
                         'w_size': w_size,
                         'w_overlap': w_overlap,
                         'min_freq': min_freq,
                         'max_freq': max_freq,
                         'n_freqs': n_freqs,
                         'cv_folds': cv_folds,
                         'reps': reps,
                         'scoring': scoring}

#             if not isinstance(sub_list_str, list):
#                 sub_list_str = [sub_list_str]

#             if len(sub_list_str) > 1:
#                  sub_folder = '-'.join([sub_list_str[0], sub_list_str[-1]])
#             else:
#                  sub_folder = sub_list_str[0]

            sub_folder = subID

            if shuffle_labels:
                shuf_labs = 'labels_shuffled'
            else:
                shuf_labs = ''

            if reg is not None:
                if isinstance(reg, float):
                    reg_str = 'shrinkage'+str(reg)
                elif isinstance(reg, str):
                    reg_str = reg
                elif isinstance(reg, list):
                    reg_str = 'shrinkageCV'
            else:
                reg_str = ''

            if picks_str is not None:
                picks_str_folder = picks_str
            else:
                picks_str_folder = ''

            fpath = op.join(config.paths["06_decoding-csp"],
                            pwr_style,
                            part_epo,
                            signaltype,
                            contrast_str,
                            scoring,
                            reg_str,
                            picks_str_folder,
                            shuf_labs,
                            sub_folder)
            if (op.exists(fpath) and not overwrite):
                path_save = op.join(config.paths["06_decoding-csp"],
                                    pwr_style,
                                    part_epo,
                                    signaltype,
                                    contrast_str + datetime_str,
                                    scoring,
                                    reg_str,
                                    picks_str_folder,
                                    shuf_labs,
                                    sub_folder + datetime_str)
            else:
                path_save = fpath
            helpers.chkmk_dir(path_save)
            fname = op.join(path_save, 'info.json')
            with open(fname, 'w+') as outfile:
                json.dump(info_dict, outfile)

        if save_csp_patterns:
            sub_patterns_ = np.asarray(tf_patterns_list)
            fpath = op.join(path_save, 'patterns')
            helpers.chkmk_dir(fpath)
            fname = op.join(fpath, 'patterns_per_sub.npy')
            np.save(fname, sub_patterns_)
            np.save(fname[:-4] + '__times' + '.npy', centered_w_times)
            np.save(fname[:-4] + '__freqs' + '.npy', freq_ranges)
            del(fpath, fname)

        if save_scores:
            sub_scores_ = np.asarray(tf_scores_list)
            fpath = op.join(path_save, 'scores')
            helpers.chkmk_dir(fpath)
            fname = op.join(fpath, 'scores_per_sub.npy')
            np.save(fname, sub_scores_)
            np.save(fname[:-4] + '__times' + '.npy', centered_w_times)
            np.save(fname[:-4] + '__freqs' + '.npy', freq_ranges)
            del(fpath, fname)

    return tf_scores_list, centered_w_times


# %%
warnings.filterwarnings('ignore')
old_log_level = mne.set_log_level('WARNING', return_old_level=True)
print(old_log_level)


# %%

# set up parameters
decod_params = dict(
    reps=1,
    scoring='roc_auc',
    t_min=-0.5,
    t_max=2.5,
    min_freq=6,
    max_freq=26,
    n_freqs=10,
    w_size=0.5,
    n_cycles=None,
    w_overlap=0.5,
    pwr_style='induced',
    reg_csp=list(np.logspace(-3, 0, 10)),
    n_components=6,
    n_cv_folds=5,
    part_epo='stimon',
    signaltype='collapsed',
    picks_str=None,
    save_scores=True,
    save_csp_patterns=True,
    overwrite=True
)

sub_list = np.setdiff1d(np.arange(1, 28), config.ids_missing_subjects +
                        config.ids_excluded_subjects)
sub_list_str = ['VME_S%02d' % sub for sub in sub_list]

# when running on the cluster we want parallelization along the
# subject dimension
if not helpers.is_interactive():
    helpers.print_msg('Running Job Nr. ' + sys.argv[1])
    job_nr = int(float(sys.argv[1]))
    sub_list_str = [sub_list_str[job_nr]]

sub_list_str = [sub_list_str[0]]

cond_dict = {'Load': ['LoadLow', 'LoadHigh'],
             'Ecc': ['EccS', 'EccM', 'EccL']}

# We only need to extract these values once:
if sub_list_str[0] == 'VME_S01':
    helpers.extract_var("csp_n_freqs", decod_params["n_freqs"],
                        exp_format=".0f")
    helpers.extract_var("csp_freq_min", decod_params["min_freq"],
                        exp_format=".0f")
    helpers.extract_var("csp_freq_max", decod_params["max_freq"],
                        exp_format=".0f")
    helpers.extract_var("csp_timewin_size", decod_params["w_size"])
    helpers.extract_var("csp_timewin_overlap_perc",
                        decod_params["w_overlap"]*100,
                        exp_format=".2f")

# %%


warnings.filterwarnings('ignore')

for shuf_labels_bool in [True, False]:  # [False]:  #

    _ = decode(sub_list_str, ['LoadLow', 'LoadHigh'], config.event_dict,
               shuffle_labels=shuf_labels_bool, **decod_params)
    _ = decode(sub_list_str, ['LoadLowEccL', 'LoadHighEccL'],
               config.event_dict,
               shuffle_labels=shuf_labels_bool, **decod_params)
    _ = decode(sub_list_str, ['LoadLowEccS', 'LoadHighEccS'],
               config.event_dict,
               shuffle_labels=shuf_labels_bool, **decod_params)
    _ = decode(sub_list_str, ['LoadLowEccM', 'LoadHighEccM'],
               config.event_dict,
               shuffle_labels=shuf_labels_bool, **decod_params)
#     _ = decode(sub_list_str, ['EccS', 'EccL'], config.event_dict,
#                shuffle_labels=shuf_labels_bool, **decod_params)
#     _ = decode(sub_list_str, ['EccM', 'EccL'], config.event_dict,
#                shuffle_labels=shuf_labels_bool, **decod_params)
#     _ = decode(sub_list_str, ['EccS', 'EccM'], config.event_dict,
#                shuffle_labels=shuf_labels_bool, **decod_params)

# for shuf_labels_bool in [False, True]:
#     _ = decode(sub_list_str, ['EccS', 'EccM', 'EccL'], config.event_dict,
#                shuffle_labels=shuf_labels_bool, **decod_params)

# _ = decode(sub_list_str,
#            ['LoadLow', 'LoadHigh'],
#            config.event_dict,
#            pwr_style='induced',
#            reps=50,
#            scoring='roc_auc',
#            shuffle_labels=False, overwrite=False)


# #Decode from single hemispheres:

# _ = decode(sub_list_str, ['LoadLow', 'LoadHigh'], config.event_dict,
#            reps=50, scoring='roc_auc',
#            shuffle_labels=False, overwrite=True, picks_str='Left',
#            min_freq=8, max_freq=14, n_freqs=1)

# _ = decode(sub_list_str, ['LoadLow', 'LoadHigh'], config.event_dict,
#            reps=50, scoring='roc_auc',
#            shuffle_labels=False, overwrite=True, picks_str='Right',
#            min_freq=8, max_freq=14, n_freqs=1)

# %%
