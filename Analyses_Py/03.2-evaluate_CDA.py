import os
import os.path as op
from pathlib import Path
import numpy as np
from collections import defaultdict
import matplotlib
from matplotlib import pyplot as plt
import mne

from scipy import stats
from mne.stats import permutation_cluster_1samp_test

from library import helpers, config


# following code taken from:
# https://github.com/mne-tools/mne-biomag-group-demo/blob/master/scripts/processing/11-group_average_sensors.py


#sub_list = [1, 2] #,2,3,5, 6, 7, 8, 9, 10, 13, 16, 17, 18, 20, 22, 23, 24, 26, 27]
sub_list = np.setdiff1d(np.arange(1,28), config.ids_missing_subjects + config.ids_excluded_subjects)

# all_evokeds = [list() for _ in range(11)] 
# for sub in sub_list: #[3, 7, 22]:
#     subID = 'VME_S%02d' % sub
#     evokeds = mne.read_evokeds(op.join(config.path_evokeds, subID + '-ave.fif'))
#     for idx, evoked in enumerate(evokeds):
#         all_evokeds[idx].append(evoked)  # Insert to the container

#a_evo = dict.fromkeys(config.factor_levels, list(mne.Evoked))

# a_evo = defaultdict(list)
# for sub in sub_list: #[3, 7, 22]:
#     subID = 'VME_S%02d' % sub
#     evokeds = mne.read_evokeds(op.join(config.path_evokeds, subID + '-ave.fif'), verbose=False)
#     for idx, key in enumerate(config.factor_levels):
#         a_evo[key].append(evokeds[idx].crop(-0.4, 2.3))  # Insert to the container


# for idx,subID in enumerate(sub_list):
#     subsub = 'VME_S%02d' % subID
#     for cond in a_evo.keys():
#             #dat = a_evo[cond]._data
#             #tms_idx = np.where((0.400 < evoked_dict[cond].times) & (evoked_dict[cond].times < 1.45))
#             mean_amplitues_dict[cond] = 1000000*np.mean(np.mean(a_evo[cond][idx]._data, 0)[(slice(416, 747, None),)])
#     write_mean_amp_to_file(subsub)



# for idx, evokeds in enumerate(all_evokeds):
#     all_evokeds[idx] = mne.combine_evoked(evokeds, 'nave')  # Combine subjects



for epo_part in ['stimon']:

    evokeds =  defaultdict(list)

    cda_evokeds = []
    for sub in sub_list: #[3, 7, 22]:
        subID = 'VME_S%02d' % sub
        fname = op.join(config.path_epos_sorted, epo_part, 'difference', subID + '-epo.fif')
        epos = mne.read_epochs(fname, verbose=False)
        event_dict = helpers.get_event_dict(epos.event_id)
        epos.apply_baseline((-0.4, 0))
        for cond in ['LoadLow', 'LoadHigh']:
            evoked = epos[event_dict[cond]].average()   
            evokeds[cond].append(evoked)
        cda_evokeds.append(epos.average())
        #cda_evokeds.append(evokeds.crop(-0.3, 2.3))  # Insert to the container

    # Main curve CDA:
    cda_grandavg = mne.grand_average(cda_evokeds) 
    label_dict = dict()
    label_dict['CDA: Contra - Ipsi'] = cda_evokeds

    res = mne.viz.plot_compare_evokeds(label_dict, 
                                combine='mean', 
                                legend=1, 
                                vlines=[0, 0.8], 
                                truncate_xaxis=False
                                )


# Run cb-perm test to find intervall of interest:
n_jobs = 2  # nb of parallel jobs

channel = 'ROI avg'
#idx = contrast.ch_names.index(channel)
c_list = [evo.copy().crop(0.2,2.2) for evo in cda_evokeds]
data = np.array([np.mean(c.data, axis=0) for c in c_list])

# number of permutations to run
n_permutations = 1000  

# set initial threshold
p_initial = 0.01

# set family-wise p-value
p_thresh = 0.01

connectivity = None
tail = 0.  # for two sided test

# set cluster threshold
n_samples = len(data)
threshold = -stats.t.ppf(p_initial / (1 + (tail == 0)), n_samples - 1)
if np.sign(tail) < 0:
    threshold = -threshold

cluster_stats = permutation_cluster_1samp_test(
    data, threshold=threshold, n_jobs=n_jobs, verbose=True, tail=tail,
    step_down_p=0.05, connectivity=connectivity,
    n_permutations=n_permutations, seed=42)

T_obs, clusters, cluster_p_values, _ = cluster_stats

times = 1e3 * c_list[0].times

fig, axes = plt.subplots(2, sharex=True)
ax = axes[0]
ax.plot(times, 1e6 * data.mean(axis=0), label="CDA grand average (ROI)", color='blue')
ax.hlines(0, -400, 2300)
ax.vlines([0,200, 2200],-20,20)
ax.set(title=' ', ylabel="EEG (uV)", ylim=[-2.8, 2.8])
ax.legend()

ax = axes[1]
for i_c, c in enumerate(clusters):
    c = c[0]
    if cluster_p_values[i_c] < p_thresh:
        h1 = ax.axvspan(times[c.start], times[c.stop - 1],
                        color='r', alpha=0.3)
hf = ax.plot(times, T_obs, 'g')
ax.hlines(0, -400, 2300)
ax.legend((h1,), (u'p < %s' % p_thresh,), loc='upper right', ncol=1)
# ax.set(xlabel="time (ms)", ylabel="T-values",
#        ylim=[-10., 10.], xlim=times[[0, -1]] * 000)
fig.tight_layout(pad=0.5)

ax.vlines([0, 200, 2200],-20,20)
plt.show()  




# Main effect Load:
conds = ['LoadLow', 'LoadHigh']
plt_names = ['Load Low', 'Load High']
plt_dict = {nn: evokeds[k] for k, nn in zip(conds, plt_names)}
res = mne.viz.plot_compare_evokeds(plt_dict, 
                             combine='mean', 
                             #colors = {k: config.colors[k] for k in conds},
                             vlines=[0], 
                             ci=True,
                             ylim=dict(eeg=[-1.5,1.5]),
                             title="Memory Load"
                             )


# Main Effect Ecc:
conds = ['EccS', 'EccM', 'EccL']
plt_names = ['4°', '9°', '14°']
plt_dict = {nn: a_evo[k] for k, nn in zip(conds, plt_names)}
res = mne.viz.plot_compare_evokeds(plt_dict, 
                             combine='mean', 
                             vlines=[0], 
                             ci=False,
                             ylim=dict(eeg=[-1.5,1.5]),
                             title="Eccentricity"
                             )


# Interaction I: 

fig, axs = plt.subplots(1,3)

for ecc, tt, idx, leg in zip(['EccS', 'EccM', 'EccL'], 
                        ['Ecc = 4°', 'Ecc = 9°', 'Ecc = 14°'], 
                        [0, 1, 2], 
                        [1, False, False]):
    conds = ['LoadLow'+ecc, 'LoadHigh'+ecc]
    plt_names = ['Load Low', 'Load High']
    plt_dict = {nn: a_evo[k] for k, nn in zip(conds, plt_names)}
    res = mne.viz.plot_compare_evokeds(plt_dict, 
                                combine='mean', 
                                vlines=[0], 
                                ci=False,
                                axes=axs[idx],
                                ylim=dict(eeg=[-1.5,1.5]),
                                title=tt, 
                                legend=leg,
                                show=False
                                )
                        


# Main effect Load:
res = mne.viz.plot_compare_evokeds(dict(High = all_evokeds[config.factor_dict['LoadHigh']].crop(tmin=-0.4, tmax=2.3), 
                                  Low = all_evokeds[config.factor_dict['LoadLow']].crop(tmin=-0.4, tmax=2.3)), 
                             combine='mean', 
                             vlines=[0], 
                             ci=True,
                             ylim=dict(eeg=[-1.5,1.5]),
                             title="Memory Load"
                             )
ff = 'MainEff_Load.png'
res[0].savefig(op.join(config.path_evokeds, 'Plots', ff))


# Main effect Ecc:
res = mne.viz.plot_compare_evokeds(dict(Small = all_evokeds[config.factor_dict['EccS']].crop(tmin=-0.3, tmax=2.3), 
                                        Medium = all_evokeds[config.factor_dict['EccM']].crop(tmin=-0.3, tmax=2.3), 
                                        Large = all_evokeds[config.factor_dict['EccL']].crop(tmin=-0.3, tmax=2.3)), 
                                  combine='mean', 
                                  vlines=[0], 
                                  ci=True,
                                  ylim=dict(eeg=[-1.5,1.5]),
                                  title = 'Eccentricity')
ff = 'MainEff_Ecc.png'
res[0].savefig(op.join(config.path_evokeds, 'plots', ff))

# Interaction:
for ecc, tt in zip(['EccS', 'EccM', 'EccL'], ['Ecc = 4°', 'Ecc = 9°', 'Ecc = 14°']):
    res = mne.viz.plot_compare_evokeds(dict(High = all_evokeds[config.factor_dict['LoadHigh' + ecc]].crop(tmin=-0.3, tmax=2.3),
                                      Low = all_evokeds[config.factor_dict['LoadLow' + ecc]].crop(tmin=-0.3, tmax=2.3)),
                                 combine = 'mean', 
                                 vlines=[0], 
                                 ylim=dict(eeg=[-1.5,1.5]), 
                                 title = tt, 
                                 show=False)
    ff = 'Load_' + ecc + '.png'
    res[0].savefig(op.join(config.path_evokeds, 'plots', ff))

# Interaction II:
for load, tt in zip(['LoadLow', 'LoadHigh'], ['Memory Load Low', 'Memory Load High']):
    plot_dict = dict()
    plot_dict['4°'] = all_evokeds[config.factor_dict[load + 'EccS']].crop(tmin=-0.4, tmax=2.3)
    plot_dict['9°'] = all_evokeds[config.factor_dict[load + 'EccM']].crop(tmin=-0.4, tmax=2.3)
    plot_dict['14°'] = all_evokeds[config.factor_dict[load + 'EccL']].crop(tmin=-0.4, tmax=2.3)
    res = mne.viz.plot_compare_evokeds(plot_dict,
                                 combine = 'mean', 
                                 vlines=[0], 
                                 ylim=dict(eeg=[-1.5,1.5]), 
                                 title = tt, 
                                 show=False)
    ff = 'Ecc_' + load + '.png'
    res[0].savefig(op.join(config.path_evokeds, 'plots', ff))




c_list = []
for evo_l, evo_h in zip(all_evokeds[config.factor_dict['LoadLowEccL']], all_evokeds[config.factor_dict['LoadHighEccL']]):
    contrast = mne.combine_evoked([evo_h, evo_l], weights = 'equal')
    contrast.crop(-0.4, 0.6)
    c_list.append(contrast)

n_jobs = 2  # nb of parallel jobs

channel = 'ROI avg'
#idx = contrast.ch_names.index(channel)
c_list = cda_evokeds
data = np.array([np.mean(c.data, axis=0) for c in c_list])

n_permutations = 1000  # number of permutations to run

# set initial threshold
p_initial = 0.001

# set family-wise p-value
p_thresh = 0.01

connectivity = None
tail = 0.  # for two sided test

# set cluster threshold
n_samples = len(data)
threshold = -stats.t.ppf(p_initial / (1 + (tail == 0)), n_samples - 1)
if np.sign(tail) < 0:
    threshold = -threshold

cluster_stats = permutation_cluster_1samp_test(
    data, threshold=threshold, n_jobs=n_jobs, verbose=True, tail=tail,
    step_down_p=0.05, connectivity=connectivity,
    n_permutations=n_permutations, seed=42)

T_obs, clusters, cluster_p_values, _ = cluster_stats

times = 1e3 * c_list[0].times

fig, axes = plt.subplots(2, sharex=True)
ax = axes[0]
ax.plot(times, 1e6 * data.mean(axis=0), label="CDA grand average (ROI)", color='blue')
ax.hlines(0, -400, 2300)
ax.vlines([0,200, 2200],-20,20)
ax.set(title=' ', ylabel="EEG (uV)", ylim=[-2.8, 2.8])
ax.legend()

ax = axes[1]
for i_c, c in enumerate(clusters):
    c = c[0]
    if cluster_p_values[i_c] < p_thresh:
        h1 = ax.axvspan(times[c.start], times[c.stop - 1],
                        color='r', alpha=0.3)
hf = ax.plot(times, T_obs, 'g')
ax.hlines(0, -400, 2300)
ax.legend((h1,), (u'p < %s' % p_thresh,), loc='upper right', ncol=1)
ax.set(xlabel="time (ms)", ylabel="T-values",
       ylim=[-10., 10.], xlim=contrast.times[[0, -1]] * 1000)
fig.tight_layout(pad=0.5)

ax.vlines([0, 200, 2200],-20,20)
plt.show()  





