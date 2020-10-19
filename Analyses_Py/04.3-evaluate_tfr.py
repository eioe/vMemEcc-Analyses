import os
import os.path as op
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats
import mne
from mne.stats import permutation_cluster_1samp_test
from library import helpers, config


def load_avgtfr(subID, condition, pwr_style='induced', 
                part_epo='fulllength', baseline=None, mode=None): 
    fpath = op.join(config.path_tfrs, pwr_style, 'tfr_lists', part_epo)
    fname = op.join(fpath, subID + '-collapsed-avgTFRs-tfr.h5')
    tfr_ = mne.time_frequency.read_tfrs(fname, condition=condition)
    if baseline is not None:
        tfr_.apply_baseline(baseline=baseline, mode=mode)
    return tfr_


def plot_tfr_side(ax, tfr, picks, cbar=True, tmin=None, tmax=None, 
                  vmin=None, vmax=None, title='', cmap='RdBu'):
        ha = tfr.copy().crop(tmin, tmax).plot(axes=ax, 
                                      show=False, 
                                      colorbar=cbar,
                                      picks=picks, 
                                      combine='mean', 
                                      title=title, 
                                      vmax=vmax, 
                                      vmin=vmin, 
                                      cmap=cmap)
        ytick_range = ax.get_ylim()
        ytick_vals = np.arange(*np.round(ytick_range), 2)
        ax.yaxis.set_ticks(ytick_vals)
        ax.axvspan(0, 0.2, color='grey', alpha=0.3)
        ax.axvspan(2.2, 2.5, color='grey', alpha=0.3)
        ax.vlines((-0.8, 0, 0.2, 2.2), ymin=-1000, ymax=10000, linestyles='dashed')
        #ax.set_title('uV^2/Hz')
        return ha

def get_lateralized_power_difference(pwr_, picks_contra, picks_ipsi):
    if not len(picks_contra) == len(picks_ipsi):
        raise ValueError('Picks must be of same length.')
    pwr_diff = pwr_.copy().pick_channels(picks_contra, ordered=True)
    pwr_ordered_chans = pwr_.copy().reorder_channels(picks_contra + picks_ipsi)
    # keep flexible to use for data with 3 (AvgTFR) and 4 (EpoTFR) dimensions: 
    d_contra = pwr_ordered_chans._data[..., :len(picks_contra), :, :]
    d_ipsi = pwr_ordered_chans._data[..., len(picks_contra):, :, :]
    pwr_diff._data = d_contra - d_ipsi
    return pwr_diff


def run_cbp_test(data, p_initial = 0.05, p_thresh = 0.05):
    # number of permutations to run
    n_permutations = 1000 
    # set initial threshold
    p_initial = p_initial
    # set family-wise p-value
    p_thresh = p_thresh
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


def plot_cbp_result_cda(ax, T_obs, clusters, cluster_p_values, p_thresh, 
                        cbp_times=None, times_full=None):
    if cbp_times is None:
        if times_full is None: 
            times_full = range(len(T_obs))
        cbp_times = times_full
    if times_full is None: 
        times_full = cbp_times
    y_max = np.max(np.abs(T_obs)) * np.array([-1.1, 1.1])
    for i_c, c in enumerate(clusters):
        c = c[0]
        if cluster_p_values[i_c] < p_thresh:
            h1 = ax.axvspan(cbp_times[c.start], cbp_times[c.stop - 1],
                            color='r', alpha=0.3)
    hf = ax.plot(cbp_times, T_obs, 'g')
    ax.hlines(0, times_full[0], times_full[-1])
    ax.legend((h1,), (u'p < %s' % p_thresh,), loc='upper right', ncol=1, prop={'size': 9})
    ax.set(xlabel="Time (s)", ylabel="T-values",
            ylim=y_max, xlim=times_full[np.array([0,-1])])
    #fig.tight_layout(pad=0.5)
    ax.axvspan(0, 0.2, color='grey', alpha=0.3)
    ax.axvspan(2.2, 2.3, color='grey', alpha=0.3)
    ax.vlines([-0.8, 0,0.2,2.2], *y_max, linestyles='--', colors='k',
                    linewidth=1., zorder=1)
    #ax.set_aspect(0.33)
    ax.set_title('')
    #ax.set_aspect('auto', adjustable='datalim')
    #ax.set(aspect=1.0/ax.get_data_ratio()*0.25, adjustable='box')
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)


def plot_main_eff(cond, cond_dict, data, ax, n_boot=1000):
    sns.lineplot(x='time', y='pwr', hue=cond, data=data, n_boot=n_boot, 
    palette=[config.colors[l] for l in cond_dict[cond]], ax=ax)
    ytick_range = ax.get_ylim()
    ax.set(xlim=(-1.1, 2.3), ylim=ytick_range)
    ax.set_ylabel('V$^2$')
    ax.set_xlabel('Time (s)')
    ax.axvspan(0, 0.2, color='grey', alpha=0.3)
    ax.axvspan(2.2, 2.5, color='grey', alpha=0.3)
    ax.vlines((-0.8, 0, 0.2, 2.2), ymin=ytick_range[0], ymax=ytick_range[1], 
            linestyles='dashed')
    ax.hlines(0, xmin=-1.1, xmax=2.3)



def get_tfr(epos, picks='all', average=True, freqs=None):
    if freqs is None:
        freqs = np.concatenate([np.arange(6, 26, 1)])  # , np.arange(16,30,2)])
    n_cycles = freqs / 2.  # different number of cycle per frequency
    power = mne.time_frequency.tfr_morlet(epos, picks=picks, freqs=freqs,
                                          n_cycles=n_cycles, use_fft=True,
                                          return_itc=False, average=average,
                                          decim=1, n_jobs=-2)
    return power


def get_mean_pwrdiff_per_trial(subID, freqs_to_avg = np.arange(8,13), 
                               t_start=None, t_stop=None, pwr_style='induced', part_epo='fulllength', 
                               picks=config.chans_CDA_all):
    epos_ = helpers.load_data(subID, config.path_epos_sorted + '/' +
                              part_epo + '/collapsed', '-epo')
    
    # Shift time, so that 0 == Stimulus Onset:
    epos_ = epos_.shift_time(-config.times_dict['cue_dur'])
    
    if pwr_style == 'induced':
        epos_ = epos_.subtract_evoked()

    event_dict = helpers.get_event_dict(epos_.event_id)

    sub_dfs = list()

    for load in ['LoadLow', 'LoadHigh']:
        for ecc in ['EccS', 'EccM', 'EccL']:
            epos_cond = epos_[event_dict[load]][event_dict[ecc]]
            # Grab trial numbers (correct for collapsing):
            trial_nums = [tn-720 if tn>=720 else tn for tn in epos_cond.selection]
            # correct for zero indexing:
            trial_nums = [t+1 for t in trial_nums]
            # Get TFRs and calc lat power:
            tfrs_cond = get_tfr(epos_cond,
                                   picks=picks, 
                                   average=False, 
                                   freqs=freqs_to_avg)
            tfrs_cond.apply_baseline(baseline=(-(config.times_dict['cue_dur']+0.3), 
                                               -(config.times_dict['cue_dur']+0.1)), 
                                     mode='mean')
            tfr_lat = get_lateralized_power_difference(tfrs_cond, 
                                                       config.chans_CDA_dict['Contra'], 
                                                       config.chans_CDA_dict['Ipsi'])
            # Crop to cluster times: 
            tfr_lat.crop(t_start, t_stop)
            tfr_lat_df = tfr_lat.to_data_frame()
            tfr_lat_df = tfr_lat_df.loc[:, ['epoch'] + tfr_lat.ch_names]
            tfr_lat_df = tfr_lat_df.groupby('epoch').agg('mean').reset_index()
            tfr_lat_df['mean_pwr_diff'] = tfr_lat_df.loc[:, tfr_lat.ch_names].mean(axis=1)
            tfr_lat_df['c_StimN'] = load
            tfr_lat_df['c_Ecc'] = ecc
            tfr_lat_df['trial_num'] = [trial_nums[i] for i in tfr_lat_df.epoch]
            tfr_lat_df = tfr_lat_df.drop('epoch', axis=1)
            
            # store in list
            sub_dfs.append(tfr_lat_df)

    sub_df = pd.concat(sub_dfs, axis=0)
    sub_df['subID'] = subID
    return(sub_df)




sub_list = np.setdiff1d(np.arange(1, 28), config.ids_missing_subjects +
                        config.ids_excluded_subjects)               
sub_list_str = ['VME_S%02d' % sub for sub in sub_list]

part_epo = 'fulllength'
pwr_style = 'induced'  # 'evoked' # 


# get list with avg TFRs for all trials:
tfr_list = [load_avgtfr(subID, 'all', pwr_style, part_epo, 
                        baseline=(-(config.times_dict['cue_dur']+0.3), 
                                  -(config.times_dict['cue_dur']+0.1)), 
                        mode='mean') 
            for subID in sub_list_str]

grand_avgtfr_all = mne.grand_average(tfr_list)
times = grand_avgtfr_all.times
freqs = grand_avgtfr_all.freqs


##############################################################################################
# Plot TF diag per hemisphere across all conditions:
for side in ['Contra', 'Ipsi']:
    fig, ax = plt.subplots(1, figsize=(6,4))
    tf_contra = plot_tfr_side(ax, grand_avgtfr_all, picks=config.chans_CDA_dict[side], 
                tmin=-1.1, tmax=2.3, title=side, cbar=True, 
                vmin=-6e-10, vmax=6e-10)

    # Save it: 
    fpath = op.join(config.path_plots, 'TFR', part_epo)
    helpers.chkmk_dir(fpath)
    fname = op.join(fpath, f'grandavgTFR_{side}.png')
    fig.savefig(fname, bbox_inches="tight")
##############################################################################################
##############################################################################################


# Calculate the difference between the hemispheres:
diff_avgtfr_all = get_lateralized_power_difference(grand_avgtfr_all, 
                                                 config.chans_CDA_dict['Contra'], 
                                                 config.chans_CDA_dict['Ipsi'])

##############################################################################################
fig, ax = plt.subplots(1, figsize=(6,4))
tf_contra = plot_tfr_side(ax, diff_avgtfr_all, picks='all', 
            tmin=-1.1, tmax=2.3, title='Contra - Ipsi', cbar=True, 
            vmin=-4e-10, vmax=4e-10, cmap='PRGn')

# Save it: 
fpath = op.join(config.path_plots, 'TFR', part_epo)
helpers.chkmk_dir(fpath)
fname = op.join(fpath, f'grandavgTFR_Difference.png')
fig.savefig(fname, bbox_inches="tight")

# Create version with red box around classical alpha range (8-12Hz):
rect = mpl.patches.Rectangle((-1.1, 8), 2.3 - -1.1, 4, 
                             linewidth=3, edgecolor='r', facecolor='none', alpha=0.5)
ax.add_patch(rect)
# Save it: 
fpath = op.join(config.path_plots, 'TFR', part_epo)
helpers.chkmk_dir(fpath)
fname = op.join(fpath, f'grandavgTFR_Difference_classAlpha.png')
fig.savefig(fname, bbox_inches="tight")
##############################################################################################
##############################################################################################


# Get the difference per subject: 

diffs_avgtfr_all = [get_lateralized_power_difference(tfr, 
                                                   config.chans_CDA_dict['Contra'], 
                                                   config.chans_CDA_dict['Ipsi']) for 
                    tfr in tfr_list]


##############################################################################################
# Make single subject plots:
fig, ax = plt.subplots(7,3)
for tfr_, ax_ in zip(diffs_avgtfr_all, ax.reshape(-1)):
    plot_tfr_side(ax_, tfr_, picks='all', 
            tmin=-1.1, tmax=2.3, title='Contra - Ipsi', cbar=False, 
            cmap='PRGn')
    ax_.xaxis.label.set_visible(False)
    ax_.yaxis.label.set_visible(False)
for ax_ in ax[:-1].reshape(-1):
    ax_.xaxis.set_visible(False)
ax[3,0].yaxis.label.set_visible(True)
ax[6,1].xaxis.label.set_visible(True)

# Save it: 
fpath = op.join(config.path_plots, 'TFR', part_epo)
helpers.chkmk_dir(fpath)
fname = op.join(fpath, 'singlesubs_avgTFR_Difference.png')
fig.savefig(fname, bbox_inches="tight")
##############################################################################################
##############################################################################################


# Extract standard alpha (8-12Hz):
frqs_idx = [(8 <= diffs_avgtfr_all[0].freqs) & (diffs_avgtfr_all[0].freqs <= 12)]
diffs_stdalpha_mean = [d.data[:, :, :].mean(axis=(0,1)) for d in diffs_avgtfr_all]

##############################################################################################
# plot it in 2D: 
plt_df = pd.DataFrame(np.array(diffs_stdalpha_mean).swapaxes(1,0), 
                      columns=sub_list_str)
plt_df['time'] = times
plt_df_long = plt_df.melt(id_vars='time', var_name='subID', value_name='pwr')
plt_df_long['hue'] = 'Alpha Power: Contra - Ipsi'

fig, axes = plt.subplots(2, figsize=(24,4), sharex=True)
ax = axes[0]
sns.lineplot(x='time', y='pwr', hue='hue', data=plt_df_long, n_boot=100, ax=ax)
ytick_range = ax.get_ylim()
ax.set(xlim=(-1.1, 2.3), ylim=ytick_range)
ax.set_ylabel('µV^2/Hz')
#ax.set_xlabel('Time (s)')
#ytick_vals = np.arange(*np.round(ytick_range), 2)
#ax.yaxis.set_ticks(ytick_vals)
ax.axvspan(0, 0.2, color='grey', alpha=0.3)
ax.axvspan(2.2, 2.5, color='grey', alpha=0.3)
ax.vlines((-0.8, 0, 0.2, 2.2), ymin=ytick_range[0], ymax=ytick_range[1], 
          linestyles='dashed')
ax.hlines(0, xmin=-1.1, xmax=2.3)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[1:], labels=labels[1:], loc=1, prop={'size': 9})
ax.xaxis.label.set_visible(False)
##############################################################################################
##############################################################################################


# run CBP test on classical alpha data (sub x times):
# crop to time of interest: 
mask_time = [(times >= -0.8) & (times <= 2.2)]
data = np.array(diffs_stdalpha_mean)[:, mask_time[0]]
T_obs, clusters, c_pvals = run_cbp_test(data, 
                                        p_initial=0.05)


##############################################################################################
# add it to figure:
ax = axes[1]
plot_cbp_result_cda(ax, T_obs, clusters, c_pvals, 0.05, 
                         cbp_times=times[mask_time[0]], 
                         times_full=np.arange(-1.1, 2.3, 0.002))

# Save it: 
fpath = op.join(config.path_plots, 'TFR', part_epo)
helpers.chkmk_dir(fpath)
fname = op.join(fpath, 'classAlpha_Difference.png')
fig.savefig(fname, bbox_inches="tight")
##############################################################################################
##############################################################################################

# Export single trial data about lateral. difference in alpha power 
# for the CDA ROI during the sign cluster:

# For the retention intervall, we look at the 2nd cluster.
cluster_dict = {'name': 'retention', 
                't_start': times[mask_time[0]][clusters[1]][0], 
                't_stop': times[mask_time[0]][clusters[1]][-1]}

mean_pwrdiff_list = list()
for sub in sub_list:
    subID = 'VME_S%02d' % sub
    mean_pwrdiff_df = get_mean_pwrdiff_per_trial(subID, freqs_to_avg=np.arange(8,13), 
                                                 t_start= cluster_dict['t_start'], 
                                                 t_stop= cluster_dict['t_stop'])
    # Write subject data to disk:
    fpath = op.join(config.path_tfrs_summaries, pwr_style, cluster_dict['name'])
    helpers.chkmk_dir(fpath)
    fname = op.join(fpath, subID + f'-mean_pwrdiff_{cluster_dict["name"]}.csv')
    mean_pwrdiff_df.to_csv(fname, index=False)
    mean_pwrdiff_list.append(mean_pwrdiff_df)

mean_pwrdiff_all_df = pd.concat(mean_pwrdiff_list)
# Write subject data to disk:
fpath = op.join(config.path_tfrs_summaries, pwr_style, cluster_dict['name'], 
                'global_summary')
helpers.chkmk_dir(fpath)
fname = op.join(fpath, f'allsubjects-mean_pwrdiff_{cluster_dict["name"]}.csv')
mean_pwrdiff_all_df.to_csv(fname, index=False)



# compare the conditions: 

cond_dict = {'Load': ['LoadLow', 'LoadHigh'], 
             'Ecc': ['EccS', 'EccM', 'EccL']}
unit_dict = {'Load': ['LoadLow', 'LoadHigh'], 
             'Ecc': ['EccS', 'EccM', 'EccL']}
            


def get_condition_pwrdiff_df(factor, cond_dict):
    df_list = list()
    for cond in cond_dict[factor]:
        tfr_list = [load_avgtfr(subID, cond, pwr_style, part_epo, 
                    baseline=(-(config.times_dict['cue_dur']+0.3), 
                            -(config.times_dict['cue_dur']+0.1)), 
                    mode='mean') for subID in sub_list_str]

        diffs_ = [get_lateralized_power_difference(tfr, 
                                                config.chans_CDA_dict['Contra'], 
                                                config.chans_CDA_dict['Ipsi']) for 
                tfr in tfr_list]

        frqs_idx = [(8 <= diffs_[0].freqs) & (diffs_[0].freqs <= 12)]
        diffs_mean = [d.data[:, :, :].mean(axis=(0,1)) for d in diffs_]
        plt_df = pd.DataFrame(np.array(diffs_mean).swapaxes(1,0), 
                            columns=sub_list_str)
        plt_df['time'] = times
        plt_df[factor] = cond
        df_list.append(plt_df)

    df_load_concat = pd.concat(df_list)
    df_load_long = df_load_concat.melt(id_vars=['time', factor], 
                                    var_name='subID', 
                                    value_name='pwr')
    return(df_load_long)

load_df_long = get_condition_pwrdiff_df('Load', cond_dict)
ecc_df_long = get_condition_pwrdiff_df('Ecc', cond_dict)


#Plot main effect Load:
fig, ax = plt.subplots(1, figsize=(12,4))
plot_main_eff('Load', cond_dict, load_df_long, ax)
ax.legend(title='Size Memory Array', labels=['2', '4'], loc=1, prop={'size': 9})

# Save it: 
fpath = op.join(config.path_plots, 'TFR', part_epo)
helpers.chkmk_dir(fpath)
fname = op.join(fpath, 'mainEff_load.png')
fig.savefig(fname, bbox_inches="tight")

# Plot main effect Ecc:
fig, ax = plt.subplots(1, figsize=(12,4))
plot_main_eff('Ecc', cond_dict, ecc_df_long, ax)
ax.legend(title='Eccentricity', labels=['4°', '9°', '14°'], 
          loc=1, prop={'size': 9}, ncol=3)

# Save it: 
fpath = op.join(config.path_plots, 'TFR', part_epo)
helpers.chkmk_dir(fpath)
fname = op.join(fpath, 'mainEff_ecc.png')
fig.savefig(fname, bbox_inches="tight")



decim = 2
factor_levels = [2, 3]
effects = 'A*B'
n_freqs = len(freqs)
times = 


df_list = list()
for load in cond_dict['Load']:
    for ecc in cond_dict['Ecc']:
        tfr_list = [load_avgtfr(subID, load+ecc, pwr_style, part_epo, 
                    baseline=(-(config.times_dict['cue_dur']+0.3), 
                            -(config.times_dict['cue_dur']+0.1)), 
                    mode='mean') for subID in sub_list_str]

        diffs_ = [get_lateralized_power_difference(tfr, 
                                                config.chans_CDA_dict['Contra'], 
                                                config.chans_CDA_dict['Ipsi']) for 
                tfr in tfr_list]
        for d in diffs_:
            df_list.append(d.data.mean(axis=0))


aa = np.asarray(df_list)
bb = aa.reshape(21, 6, 20*1951)
bb = bb.reshape(21, 6, 20*1951)

fvals, pvals = f_mway_rm(bb, factor_levels, effects=effects)
effect_labels = ['Load', 'Ecc', 'Load x Ecc']

for effect, sig, effect_label in zip(fvals, pvals, effect_labels):
    f = plt.figure()
    # show naive F-values in gray
    plt.imshow(effect.reshape(20, 1951), cmap=plt.cm.gray, extent=[times[0],
               times[-1], freqs[0], freqs[-1]], aspect='auto',
               origin='lower')
    # create mask for significant Time-frequency locations
    effect[sig >= 0.05] = np.nan
    plt.imshow(effect.reshape(20, 1951), cmap='RdBu_r', extent=[times[0],
               times[-1], freqs[0], freqs[-1]], aspect='auto',
               origin='lower')
    cb = plt.colorbar()
    cb.set_label(label='F', rotation=0)
    ytick_range = f.axes[0].get_ylim()
    ytick_vals = np.arange(*np.round(ytick_range), 2)
    f.axes[0].yaxis.set_ticks(ytick_vals)
    plt.xlabel('Time (ms)')
    plt.ylabel('Frequency (Hz)')
    plt.title(r"Induced lateralized power for '%s' (%s)" % (effect_label, 'CDA-ROI'))
    plt.show()


effects='B'

def stat_fun(*args):
    return f_mway_rm(np.reshape(args, (21, 6, 20, 1951)), factor_levels=factor_levels,
                     effects=effects, return_pvals=False)[0]


# The ANOVA returns a tuple f-values and p-values, we will pick the former.
pthresh = 0.0005  # set threshold rather high to save some time
f_thresh = f_threshold_mway_rm(21, factor_levels, effects,
                               pthresh)
tail = 1  # f-test, so tail > 0
n_permutations = 1000  # Save some time (the test won't be too sensitive ...)
T_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_test(
    dd, stat_fun=stat_fun, threshold=f_thresh, tail=tail, n_jobs=-2,
    n_permutations=n_permutations, buffer_size=None, out_type='mask')


good_clusters = np.where(cluster_p_values < .05)[0]
T_obs_plot = np.ones_like(T_obs) *np.nan
for c, p in zip(clusters, cluster_p_values):
    if p < .05: 
        T_obs_plot[c] = T_obs[c]
        print('ahah')
#T_obs_plot[np.array(clusters[2:])] = np.nan

f = plt.figure()
for f_image, cmap in zip([T_obs, T_obs_plot], [plt.cm.gray, 'RdBu_r']):
    plt.imshow(f_image, cmap=cmap, extent=[times[0], times[-1],
               freqs[0], freqs[-1]], aspect='auto',
               origin='lower')
cb = plt.colorbar()
cb.set_label(label='F', rotation=0)
ytick_range = f.axes[0].get_ylim()
ytick_vals = np.arange(*np.round(ytick_range), 2)
f.axes[0].yaxis.set_ticks(ytick_vals)
plt.xlabel('Time (ms)')
plt.ylabel('Frequency (Hz)')
plt.title("Induced lateralized power for 'Eccentricity' (%s)\n"
          " cluster-level corrected (p <= 0.05)" % 'CDA-ROI')

plt.show()


plt.figure()
plt.imshow(T_obs_plot, cmap='RdBu_r', extent=[times[0], times[-1],
               freqs[0], freqs[-1]], aspect='auto',
               origin='lower')



# run CBP test on TF data of the CDA ROI (sub x freq x times)
diffs_allfreqs = [d.data[:, :, :].mean(axis=(0)) for d in diffs_avgtfr_all]


T_obs, clusters, c_pvals = run_cbp_test(np.array(diffs_allfreqs), 
                                        p_initial=0.05)

vmax = np.max(np.abs(T_obs))
vmin = -vmax

# Create new stats image with only significant clusters
T_obs_plot = np.nan * np.ones_like(T_obs)
for c, p_val in zip(clusters, c_pvals):
    if p_val <= 0.05:
        T_obs_plot[c] = T_obs[c]


fig, ax = plt.subplots(1, figsize=(6,4))
plt.imshow(T_obs, cmap=plt.cm.gray,
           extent=[-1.1, 2.3, freqs[0], freqs[-1]],
           aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
plt.imshow(T_obs_plot, cmap=plt.cm.RdBu_r,
           extent=[-1.1, 2.3, freqs[0], freqs[-1]],
           aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
ytick_range = ax.get_ylim()
ytick_vals = np.arange(*np.round(ytick_range), 2)
ax.yaxis.set_ticks(ytick_vals)
ax.axvspan(0, 0.2, color='grey', alpha=0.5)
ax.axvspan(2.2, 2.3, color='grey', alpha=0.5)
ax.vlines((-0.8, 0, 0.2, 2.2), ymin=ytick_range[0], ymax=ytick_range[-1], 
          linestyles='dashed')
cb = plt.colorbar()
cb.set_label(label='T', rotation=0)
plt.xlabel('Time (ms)')
plt.ylabel('Frequency (Hz)')
plt.title(f'Contra - Ipsi (T stats)')

# Save it: 
fpath = op.join(config.path_plots, 'TFR', part_epo)
helpers.chkmk_dir(fpath)
fname = op.join(fpath, 'CBP_avgTFR_Diff_classAlpha_p001.png')
fig.savefig(fname, bbox_inches="tight")


