
# %% load libs:
from collections import defaultdict
from os import path as op
import json
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.npyio import load

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

# %%

sub_list = np.setdiff1d(np.arange(1, 28), config.ids_missing_subjects +
                        config.ids_excluded_subjects)               
sub_list_str = ['VME_S%02d' % sub for sub in sub_list]

event_dict = config.event_dict
ha_high = list()
ha_low = list()
for sub in sub_list_str:
    print(f'running {sub}')
    he = get_epos(sub, 'stimon', 'collapsed', 'LoadHigh', config.event_dict)
    ha_high.append(he)
    he = get_epos(sub, 'stimon', 'collapsed', 'LoadLow', config.event_dict)
    ha_low.append(he)
# %%
evos = dict()
evos['LoadLow'] = list()
evos['LoadHigh'] = list()

for sub in range(21):
    print(f'running {sub}')
    ll = ha_low[sub].copy().average() 
    lh = ha_high[sub].copy().average() 
    evos['LoadLow'].append(ll)
    evos['LoadHigh'].append(lh)
# %%
fig, ax = plt.subplots(7,3, figsize=(30, 40))
evos_p = dict()
for sub, axs in zip(range(21), ax.reshape(-1)):
    for key in evos:
        evos_p[key] = evos[key][sub]
    ff = mne.viz.plot_compare_evokeds(evos_p, combine='mean', 
                                    #colors = {k: config.colors[k] for k in plt_dict.keys()},
                                    vlines=[0, 0.2, 2.2], 
                                    picks = ['PO3'], axes=axs, show=False)
# mne.viz.plot_compare_evokeds(evos_p, combine='mean', 
#                                 #colors = {k: config.colors[k] for k in plt_dict.keys()},
#                                 vlines=[0, 0.2, 2.2], 
#                                 picks = ['CP2'])
# %%


mne.viz.plot_compare_evokeds(evos, combine='mean', 
                                #colors = {k: config.colors[k] for k in plt_dict.keys()},
                                 vlines=[0, 0.2, 2.2], 
                                 picks = ['P7'])

# %%
