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
pwr_style = 'evoked'

# Combine subjects:
condition = 'all'
cond_str = '-'+condition
#sub_list = [1,2,3,5, 6, 7, 8, 9, 10, 13, 16, 17, 18, 20, 22, 23, 24, 26, 27]
sub_list = np.setdiff1d(np.arange(1,28), config.ids_missing_subjects + config.ids_excluded_subjects)
#sub_list = np.arange(1,5) 



all_tfrs = []
for idx, sub in enumerate(sub_list):
    subID = 'VME_S%02d' % sub
    diff = mne.time_frequency.read_tfrs(op.join(config.path_tfrs, 
                                                pwr_style, 
                                                'tfr_lists', 
                                                part_epo, 
                                                subID + '-PowDiff-avgTFRs-tfr.h5'), 
                                                condition='all')
    all_tfrs.append(diff)  # Insert to the container


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


plt.subplots(1,1,figsize=(12,12))
ax = plt.axes()
plot_global_pwr(ax)
plt.show()

def plot_global_pwr(ax):
    glob_tfr = all_tfrs[0].copy()
    glob_tfr._data = np.stack([all_tfrs[i]._data for i in range(len(sub_list))]).mean(0)
    glob_tfr.times -= 0.8
    img = glob_tfr.plot(baseline=(-.2,-0.0), 
                        picks='all', 
                        mode='mean',
                        #vmax = 3e-10, 
                        #vmin = -3e-10,
                        tmin=-1, tmax=2.3, 
                        axes=ax,
                        title='contra-ipsi', 
                        cmap='RdBu', 
                        combine='mean', 
                        show=False)
    ax.vlines([-0.8,0,0.2,2.2], *[0,25], linestyles='--', colors='k',
                linewidth=1., zorder=1)
    ax.axvspan(0, 0.2, color='grey', alpha=0.3)
    ax.axvspan(2.2, 2.3, color='grey', alpha=0.3)



ff = 'tfr_' + 'avg_' + cond_str + '.png'
img.savefig(op.join(config.path_tfrs, 'plots', ff))

afreqs = [f for f in glob_tfr.freqs if (f >= config.alpha_freqs[0]) & (f <= config.alpha_freqs[1])]
idx_afreqs = np.isin(glob_tfr.freqs, afreqs)
glob_tfr = all_tfrs[0].copy()
glob_tfr._data = np.stack([all_tfrs[i]._data for i in range(len(sub_list))]).mean(0)
glob_tfr.times -= 0.8

glob_data = glob_tfr.copy().data.mean(axis=0)
dd = glob_data[idx_afreqs,:].mean(axis=0)

plt.subplots(1,1,figsize=(12,4))
ax = plt.axes()

times = glob_tfr.times
y_max = np.max(np.abs(dd)) * np.array([-1.1, 1.1])

hf = ax.plot(times, dd, 'r')
ax.hlines(0, times[0], times[-1])
#ax.legend((hf,), ('alpha power',), loc='upper right', ncol=1, prop={'size': 9})
ax.set(xlabel="Time (s)", ylabel="lateralized alpha power (uV^2/Hz)", xlim=times[np.array([0,-1])])
#fig.tight_layout(pad=0.5)
ax.axvspan(0, 0.2, color='grey', alpha=0.3)
ax.axvspan(2.2, 2.5, color='grey', alpha=0.3)
ax.vlines([0,0.2,2.2], plt.ylim()[0], plt.ylim()[1], linestyles='--', colors='k',
                linewidth=1., zorder=1)
#ax.set_aspect(0.33)
ax.set_title('')
#ax.set_aspect('auto', adjustable='datalim')
ax.set(aspect=1.0/ax.get_data_ratio()*0.25, adjustable='box')
ax.xaxis.label.set_size(9)
ax.yaxis.label.set_size(9)

plt.show()


glob_tfr = all_tfrs[0].copy()
glob_tfr._data = np.stack([all_tfrs[i]._data for i in range(len(sub_list))])


pwr_obj = glob_tfr
epwr = glob_tfr._data.mean(axis=1).mean(axis=1)


from scipy import stats
from mne.stats import permutation_cluster_1samp_test

def run_cbp_test(data):
    # number of permutations to run
    n_permutations = 1000 
    # set initial threshold
    p_initial = 0.05
    # set family-wise p-value
    p_thresh = 0.05
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

a,b,c = run_cbp_test(epwr)



def plot_cbp_result_cda(ax, T_obs, clusters, cluster_p_values, p_thresh, times_full):
    y_max = np.max(np.abs(T_obs)) * np.array([-1.1, 1.1])
    for i_c, c in enumerate(clusters):
        c = c[0]
        if cluster_p_values[i_c] < p_thresh:
            h1 = ax.axvspan(times[c.start], times[c.stop - 1],
                            color='r', alpha=0.3)
    hf = ax.plot(times, T_obs, 'g')
    ax.hlines(0, times_full[0], times_full[-1])
    ax.legend((h1,), (u'p < %s' % p_thresh,), loc='upper right', ncol=1, prop={'size': 9})
    ax.set(xlabel="Time (s)", ylabel="T-values",
            ylim=y_max, xlim=times_full[np.array([0,-1])])
    #fig.tight_layout(pad=0.5)
    ax.axvspan(0, 0.2, color='grey', alpha=0.3)
    ax.axvspan(2.2, 2.3, color='grey', alpha=0.3)
    ax.vlines([0,0.2,2.2], *y_max, linestyles='--', colors='k',
                    linewidth=1., zorder=1)
    #ax.set_aspect(0.33)
    ax.set_title('')
    #ax.set_aspect('auto', adjustable='datalim')
    ax.set(aspect=1.0/ax.get_data_ratio()*0.25, adjustable='box')
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)


plt.subplots(1,1,figsize=(12,4))
ax = plt.axes()

plot_cbp_result_cda(ax, a, b, c, 0.05, times)






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

