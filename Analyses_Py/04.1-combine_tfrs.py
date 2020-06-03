import os.path as op
import numpy as np
import mne
from library import helpers, config


def get_power_difference(pwr_a, pwr_b, picks='all'): 
            if not pwr_a.data.shape == pwr_b.data.shape: 
                raise ValueError('TFRs must have same shape.')
            if picks == 'all': 
                picks = pwr_a.ch_names
            pwr_diff = pwr_a.copy().pick_channels(picks, ordered = True)
            d_a = pwr_a.copy().pick_channels(picks, ordered = True)._data
            d_b = pwr_b.copy().pick_channels(picks, ordered = True)._data
            pwr_diff._data = d_a - d_b
            return pwr_diff


part_epo = 'stimon'

# Combine subjects:
condition = 'all'
cond_str = '-'+condition
#sub_list = [1,2,3,5, 6, 7, 8, 9, 10, 13, 16, 17, 18, 20, 22, 23, 24, 26, 27]
sub_list = np.setdiff1d(np.arange(1,28), config.ids_missing_subjects)
#sub_list = np.arange(1,5) 



all_tfrs = []
for idx, sub in enumerate(sub_list):
    subID = 'VME_S%02d' % sub
    diff = mne.time_frequency.read_tfrs(op.join(config.path_tfrs + '\\induc\\' + part_epo, subID + '-PowDiff' + cond_str + '-tfr.fif'))
    all_tfrs.append(diff[0])  # Insert to the container


# get contrasts: 
all_tfrs = []
for idx, sub in enumerate(sub_list):
    subID = 'VME_S%02d' % sub
    tfr = dict()
    conds = ['EccS', 'EccL']
    for load in conds:#['LoadLow', 'LoadHigh']:
        cond_str = '-'+load
        tfr[load] = mne.time_frequency.read_tfrs(op.join(config.path_tfrs + '\\' + part_epo, subID + '-PowDiff' + cond_str + '-tfr.fif'))
    diff = get_power_difference(tfr[conds[0]][0], tfr[conds[1]][0])
    all_tfrs.append(diff)  # Insert to the container

glob_tfr = all_tfrs[0].copy()
glob_tfr._data = np.stack([all_tfrs[i]._data for i in range(len(sub_list))]).mean(0)
img = glob_tfr.plot(baseline=(-0.3,0), 
                    picks='all', 
                    mode='mean',
                    #vmax = 3e-10, 
                    #vmin = -3e-10,
                    tmin=-0.4, tmax=2.3, 
                    title='contra-ipsi' + '  (' + condition + ')', 
                    cmap='RdBu', 
                    combine='mean')
ff = 'tfr_' + 'avg_' + cond_str + '.png'
img.savefig(op.join(config.path_tfrs, 'plots', ff))



pwr_obj = glob_tfr
epwr = glob_tfr._data.mean(axis=1)

threshold = 2.5
n_permutations = 1000  # Warning: 100 is too small for real-world analysis.
T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(epwr, n_permutations=n_permutations,
                                   threshold=threshold, tail=0)


from mne.stats import permutation_cluster_1samp_test

plt.figure()
plt.subplots_adjust(0.12, 0.08, 0.96, 0.94, 0.2, 0.43)

# Create new stats image with only significant clusters
T_obs_plot = np.nan * np.ones_like(T_obs)
for c, p_val in zip(clusters, cluster_p_values):
    if p_val <= 0.01:
        T_obs_plot[c] = T_obs[c]

times = pwr_obj.times
freqs = pwr_obj.freqs

vmax = np.max(np.abs(T_obs))
vmin = -vmax
plt.subplot(2, 1, 1)
plt.imshow(T_obs, cmap=plt.cm.gray,
           extent=[times[0], times[-1], freqs[0], freqs[-1]],
           aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
plt.imshow(T_obs_plot, cmap=plt.cm.RdBu_r,
           extent=[times[0], times[-1], freqs[0], freqs[-1]],
           aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
plt.colorbar()
plt.xlabel('Time (ms)')
plt.ylabel('Frequency (Hz)')
plt.title('Induced power')# (%s)' % ch_name)

plt.show()

