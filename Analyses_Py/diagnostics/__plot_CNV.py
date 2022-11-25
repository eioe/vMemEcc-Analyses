
# %% load libs:
from collections import defaultdict
from os import path as op
import json
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

import mne

from library import config, helpers


# %% Functions:

def get_epos(subID, epo_part, signaltype, condition, event_dict):
    if signaltype == 'uncollapsed':
        fname = op.join(config.path_rejepo, subID + '-' + epo_part +
                        '-postica-rejepo' + '-epo.fif')
    elif signaltype in ['collapsed', 'difference']:
        fname = op.join(config.paths['03_preproc-pooled'], epo_part, signaltype,
                        '-'.join([subID, epo_part, signaltype, 'epo.fif']))
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
                                 picks = ['PO7'])

# %%
pps = defaultdict(list)
pickspps = defaultdict(list)
for sub in range(21):
    print(f'running {sub}')
    for pick in ('Contra', 'Ipsi'):  # ('LoadLow', 'LoadHigh'):
        for cond in ('LoadLow', 'LoadHigh'):
            if cond == 'LoadLow':
                ppsd = mne.time_frequency.psd_welch(ha_low[sub].copy().pick(config.chans_CDA_dict[pick]), fmin=0.01, fmax=45, average='mean', n_fft=512, verbose=False)
            elif cond == 'LoadHigh':
                ppsd = mne.time_frequency.psd_welch(ha_high[sub].copy().pick(config.chans_CDA_dict[pick]), fmin=0.01, fmax=45, average='mean', n_fft=512, verbose=False)
            pps[cond].append(ppsd)
            pickspps[pick].append(ppsd)

# %%
m_df = pd.DataFrame()
for cond in ('Contra', 'Ipsi'):
    freqs = pickspps[cond][0][1]
    yy = [p[0].mean(axis=0) for p in pickspps[cond]]
    y = np.array(yy).mean(axis=(1))
    df = pd.DataFrame(y).melt(var_name='freq', value_name='pwr')
    df['subidx'] = np.repeat(range(21), 2*46)
    df = df.groupby('subidx').mean()
    #df['freq']=np.repeat(freqs, 21)
    df['pwr_db'] = 10 * np.log10(df.pwr)
    df['cond'] = cond
    m_df = pd.concat([m_df, df])
fig, ax = plt.subplots()
sns.lineplot(data=m_df, x='freq', y='pwr_db', hue='cond', errorbar='se', ax=ax)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Power spectar density (dB)')


# %%
plt.plot(freqs, 10 * np.log10(y))
# %%
ppsd.shape
# %%
len(ppsd)
# %%
ppsd(0)
# %%
