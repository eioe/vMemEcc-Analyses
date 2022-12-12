



"""
Helper script to plot CNV data or contrast it with the CDA data. 
For CNV: load signaltype = "uncollapsed" or "collapsed"
    "collapsed": left channels are contralateral, right channels are ipsilateral
    "uncollapsed": channel positions are absolute (i.e., left channels are contralateral for CueR trials and ipsilateral for CueL trials, and so forth)
For CDA: load signaltype = "difference"

"""


# %% load libs:
from collections import defaultdict
from os import path as op
import json
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool
from functools import partial

import pandas as pd
import seaborn as sns

import mne

from library import config, helpers


# %% Functions:


def get_epos(subID, epo_part, signaltype, condition, event_dict):
    if signaltype == "uncollapsed":
        fname = op.join(
            config.paths["03_preproc-rejectET"],
            epo_part,
            "cleaneddata",
            subID + "-" + epo_part + "-rejepo" + "-epo.fif",
        )
    elif signaltype in ["collapsed", "difference"]:
        fname = op.join(
            config.paths["03_preproc-pooled"],
            epo_part,
            signaltype,
            "-".join([subID, epo_part, signaltype, "epo.fif"]),
        )
    else:
        raise ValueError(f'Invalid value for "signaltype": {signaltype}')
    epos = mne.read_epochs(fname, verbose=False)
    epos = epos.pick_types(eeg=True)
    uppers = [letter.isupper() for letter in condition]
    if np.sum(uppers) > 2:
        cond_1 = condition[: np.where(uppers)[0][2]]
        cond_2 = condition[np.where(uppers)[0][2] :]
        selection = epos[event_dict[cond_1]][event_dict[cond_2]]
    else:
        selection = epos[event_dict[condition]]
    return selection


# %%

sub_list = np.setdiff1d(
    np.arange(1, 28), config.ids_missing_subjects + config.ids_excluded_subjects
)
sub_list_str = ["VME_S%02d" % sub for sub in sub_list]

event_dict = config.event_dict
ha_high = list()
ha_low = list()
ha_all = list()

with Pool(len(sub_list_str)) as p:
    ha_high = p.map(
        partial(
            get_epos,
            epo_part="stimon",
            signaltype="difference",
            condition="LoadHigh",
            event_dict=config.event_dict,
        ),
        [s for s in sub_list_str],
    )

with Pool(len(sub_list_str)) as p:
    ha_low = p.map(
        partial(
            get_epos,
            epo_part="stimon",
            signaltype="difference",
            condition="LoadLow",
            event_dict=config.event_dict,
        ),
        [s for s in sub_list_str],
    )


# %%
evos = dict()
evos["LoadLow"] = list()
evos["LoadHigh"] = list()

for sub in range(21):
    print(f"running {sub}")
    ll = ha_low[sub].copy().average()
    lh = ha_high[sub].copy().average()
    evos["LoadLow"].append(ll)
    evos["LoadHigh"].append(lh)
# %%
fig, ax = plt.subplots(7, 3, figsize=(30, 40))
evos_p = dict()
for sub, axs in zip(range(21), ax.reshape(-1)):
    for key in evos:
        evos_p[key] = evos[key][sub]
    ff = mne.viz.plot_compare_evokeds(
        evos_p,
        combine="mean",
        colors={k: config.colors[k] for k in ("LoadLow", "LoadHigh")},
        vlines=[0, 0.8, 3.0],
        picks=["PO3"],
        axes=axs,
        show=False,
    )

# %%


mne.viz.plot_compare_evokeds(
    evos,
    combine="mean",
    colors={k: config.colors[k] for k in ("LoadLow", "LoadHigh")},
    vlines=[0, 0.2, 2.2],
    picks=["PO3"],
)

# %%
