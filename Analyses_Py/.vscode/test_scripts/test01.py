import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import re
import sklearn

import mne
from mne import preprocessing

sub = 'P12'

save_epoched  = False
save_filtered = False
ica_from_disk = False

chans_eog = ['Fp1', 'Fp2', 'AF7', 'AF8', 'FT9', 'FT10']

# Set paths:
data_path = 'D:/Felix/Seafile/Experiments/vMemEcc/Data/PilotData/'
raw_path = os.path.join(data_path, sub, 'EEG')
prepro_path = os.path.join(data_path, 'EEG', '01_preprocessed')

# Get data:
raw_files = [os.path.join(raw_path, f) for f in os.listdir(raw_path) if f.endswith('.vhdr')]
raw = mne.io.concatenate_raws([mne.io.read_raw_brainvision(f, preload=True) for f in raw_files])

# Extract info:
srate = raw.info['sfreq']
events, event_id = mne.events_from_annotations(raw)

# delete crap channel:
raw.load_data().drop_channels('EOGv')

# calculate and add HEOG and VEOG channels:
VEOGl = raw.copy().pick_channels(['Fp1', 'IO1']) #(['Fp1', 'AF7'])
VEOGr = raw.copy().pick_channels(['Fp2', 'IO2']) #(['Fp2', 'AF8'])
dataL = VEOGl.get_data(['Fp1']) - VEOGl.get_data(['IO1']) #(['AF7'])
dataR = VEOGr.get_data(['Fp2']) - VEOGr.get_data(['IO2']) #(['AF8'])
dataVEOG = np.stack((dataL,dataR), axis=0).mean(0)
HEOG = raw.copy().pick_channels(['LO1', 'LO2']) #(['FT9', 'FT10'])
#dataHEOG = HEOG.get_data(['FT9']) - HEOG.get_data(['FT10'])
dataHEOG = HEOG.get_data(['LO1']) - HEOG.get_data(['LO2'])
dataEOG = np.concatenate((dataVEOG, dataHEOG), axis=0)
info = mne.create_info(ch_names=['VEOG', 'HEOG'], sfreq=srate, ch_types=['eog', 'eog'])
rawEOG = mne.io.RawArray(dataEOG, info=info)
raw.add_channels([rawEOG], force_update_info=True)

# band pass filter:
filtered = raw.load_data().filter(l_freq=0.01, h_freq=40)
if save_filtered:
    filtered.save(fname=os.path.join(prepro_path, sub+'-filtered-raw.fif'), overwrite=True)

# Define relevant events:
targ_evs = [i for i in np.arange(150, 174)]
epo_keys = ['CueL', 'CueR', 'LoadLow', 'LoadHigh', 'EccS', 'EccM', 'EccL']

event_dict = {key: [] for key in epo_keys}
for ev in targ_evs:
    ev0 = ev - 150
    if (ev0 % 2) == 0:
        event_dict['CueL'].append(str(ev))
    else:
        event_dict['CueR'].append(str(ev))

    if (ev0 % 8) < 4:
        event_dict['LoadLow'].append(str(ev))
    else:
        event_dict['LoadHigh'].append(str(ev))
    
    if (ev0 % 24) < 8:
        event_dict['EccS'].append(str(ev))
    elif (ev0 % 24) > 15:
        event_dict['EccL'].append(str(ev))
    else:
        event_dict['EccM'].append(str(ev))

# Epoch data:
epoched = mne.Epochs(filtered, events, event_id=targ_evs, tmin=-0.5, tmax=2.2, baseline=(None,0),
                    preload=True)

epoched_cue = mne.Epochs(filtered, events, event_id=targ_evs, tmin=-1.5, tmax=0.4, baseline=(None,-1),
                    preload=True)

epoched_tot = mne.Epochs(filtered, events, event_id=targ_evs, tmin=-1.3, tmax=2.2, baseline=(None,-0.8),
                    preload=True)


epoched = epoched_tot.copy()

# reject epochs (here I use manual selection from matlab):
bad_epos = {'P09': [758, 28, 70, 78, 118, 122, 249, 506, 541, 542, 545, 685, 713, 820, 821], 
            'P08': [72, 98, 114, 115, 116, 117, 123, 141, 252, 257, 289, 346, 389, 390, 492, 575, 576, 680, 742, 743, 744, 845, 851], 
            'P07': [46, 59, 106, 163, 237, 247, 251, 281, 304, 338, 344, 406, 413, 428, 470, 471, 472, 473, 483, 623, 658, 785, 827]}
epoched.drop(indices=bad_epos[sub])



if save_epoched:
    epoched.save(fname=os.path.join(prepro_path, sub+'TOT-rej-epo.fif'))

### Load epoched data (after epo rejection)?
epoched_from_disk = False
if epoched_from_disk:
    epoched = mne.read_epochs(fname=os.path.join(prepro_path, sub+'-rej-epo.fif'))

### Load ICA data (after comp rejection)?
if ica_from_disk:
    ica = mne.preprocessing.read_ica(fname=os.path.join(prepro_path, sub+'-ica.fif.'))
else:
    # make copy for ICA (note: you should better make a copy from raw and filter cont data and re-epoch)
    filt_epo = epoched.copy()
    filt_epo.load_data().filter(l_freq=1., h_freq=None)
    ica = mne.preprocessing.ICA(method='infomax')
    ica.fit(filt_epo)
    ica.save(fname=os.path.join(prepro_path, sub+'-ica.fif.'))


## Reject components:

# Via correlation w/ EOG channels:
EOGexclude = []
for ch in ('VEOG', 'HEOG'):
    eog_indices, eog_scores = ica.find_bads_eog(epoched, ch_name=ch, threshold=3)
    EOGexclude.extend(eog_indices)
    #ica.plot_scores(eog_scores)

# Plot marked components:
ica.plot_components(inst=filtered, picks=EOGexclude)
# ica.plot_properties(raw, picks=EOGexclude)
ica.plot_sources(filtered, picks = [0, 1, 2, 3, 4, 5, 6])

# if ok: transfer to ICA obj:
ica.exclude = EOGexclude

# and kick out components:
rejComp = epoched.copy()
ica.apply(rejComp)

# visual comparison:
compare_ica_res = True
if compare_ica_res:
    old = epoched.plot()
    new = rejComp.plot()

save_ICA_cleaned = True
if save_ICA_cleaned:
    epoched.save(fname=os.path.join(prepro_path, sub+'TOT-rejComp-epo.fif'), overwrite=True)


## load ICA cleand:
epoched = mne.read_epochs(fname=os.path.join(prepro_path, sub+'TOT-rejComp-epo.fif'))

############################################################################

chans_CDA = [['P3', 'P5', 'PO3', 'PO7', 'O1'], 
             ['P4', 'P6', 'PO4', 'PO8', 'O2']]

epoched.apply_baseline((-.5,-0.0))

eposLeft = rejComp[event_dict['CueL']]
eposRight = rejComp[event_dict['CueR']]

fig = plt.figure()
plt.plot(eposLeft.times, eposLeft.copy().pick_channels(chans_CDA[0]).get_data().mean(0).mean(0))
plt.axhline(color='black', 
                lw = 0.5, 
                ls = ':')
plt.axvline(x=-0.8, color='black', ls=':')
plt.axvline(color='black', ls=':')
fig.show()

rename_dict = {chans_CDA[0][i]: chans_CDA[1][i]  for i in range(len(chans_CDA[0]))}

eposContraL = eposLeft.copy().pick_channels(chans_CDA[1])
eposContraR = eposRight.copy().pick_channels(chans_CDA[0])
eposContraR.rename_channels(rename_dict)
eposContra = mne.concatenate_epochs([eposContraL.copy(), eposContraR], add_offset=False)

eposIpsiL = eposLeft.copy().pick_channels(chans_CDA[0])
eposIpsiR = eposRight.copy().pick_channels(chans_CDA[1])
eposIpsiL.rename_channels(rename_dict)
eposIpsi = mne.concatenate_epochs([eposIpsiL.copy(), eposIpsiR], add_offset=False)

eposIpsiEccS = eposIpsi[event_dict['EccS']]
eposIpsiEccM = eposIpsi[event_dict['EccM']]
eposIpsiEccL = eposIpsi[event_dict['EccL']]
eposContraEccS = eposContra[event_dict['EccS']]
eposContraEccM = eposContra[event_dict['EccM']]
eposContraEccL = eposContra[event_dict['EccL']]

contra = eposContra.average()
ipsi = eposIpsi.average()

contraHi = eposContra[event_dict['LoadHigh']].average()
contraLo = eposContra[event_dict['LoadLow']].average()
ipsiLo = eposIpsi[event_dict['LoadLow']].average()
ipsiHi = eposIpsi[event_dict['LoadHigh']].average()


contra.plot()
ipsi.plot()

mne.viz.plot_compare_evokeds([contra, ipsi], combine='mean', vlines=[-0.8, 0])

fig = plt.figure()
for evoked, ls, legend in zip([contra, ipsi], ['-', '--'], ['contra', 'ipsi']):
    plt.plot(evoked.times, evoked.data.mean(axis=0), linestyle=ls, label=legend)
    plt.legend()
    plt.axhline(color='black', 
                lw = 0.5, 
                ls = ':')
    plt.axvline(color='black', 
                lw = 0.5, 
                ls = ':')
    plt.axvline(-0.8,
    color='black', 
    lw = 0.5, 
    ls = ':')
fig.show()

he = mne.combine_evoked([contra, -ipsi], weights='equal')
he.plot_topo()



### TFR analyses:

frequencies = np.arange(3, 25, 1)
picksR = [eposLeft.ch_names.index(ch) for ch in chans_CDA[1]]
picksL = [eposLeft.ch_names.index(ch) for ch in chans_CDA[0]]
powCo = mne.time_frequency.tfr_morlet(eposLeft, n_cycles=3, picks=picksR, return_itc=False,
                                      freqs=frequencies, decim=3)
powIp = mne.time_frequency.tfr_morlet(eposLeft, n_cycles=3, picks=picksL, return_itc=False,
                                      freqs=frequencies, decim=3)

freqs = np.logspace(*np.log10([6, 35]), num=20)
n_cycles = freqs / 2.  # different number of cycle per frequency
powerI = mne.time_frequency.tfr_morlet(eposIpsi, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=False, decim=3, n_jobs=1)
powerC = mne.time_frequency.tfr_morlet(eposContra, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=False, decim=3, n_jobs=1)

powDiff = powerC.copy()
powDiff._data = powerC.reorder_channels(powerI.info['ch_names'])._data - powerI._data

powSum = powerC.copy()
powSum._data = powerC.reorder_channels(powerI.info['ch_names'])._data + powerI._data

powL = powSum.copy()
powL._data = powDiff._data / powSum._data

f = plt.figure()
plt.fill([-1.5, 2.5, 2.5, -1.5], [0, 0, 4e-10, 4e-10], 'orange', alpha=0.1)
plt.fill([-1.5, 2.5, 2.5, -1.5], [0, 0, -6e-10, -6e-10], 'blue', alpha=0.1)
plt.text([-1.5, -1.5], [3.9e-10, -3.9e-10], ['contra > ipsi', 'contra < ipsi'])
plt.plot(powerC.times, powDiff._data.mean(0).mean(0))
f.show()

powDiff07 = powDiff.copy()
powDiff08 = powDiff.copy()

ips = powerI.plot(baseline=(-.3, -0), picks='all', mode='logratio', tmin=-1.3, tmax=2, title='ipsilateral', vmax=0.5, vmin=-0.5)
cont = powerC.plot(baseline=(-.3, -0), picks='all', mode='logratio', tmin=-1.3, tmax=2, title='contralateral', vmax=0.5, vmin=-0.5)
tot = powL.plot(baseline=None, picks='all', mode='logratio', tmin=-1.3, tmax=2, title='contra-ipsi', cmap='RdBu')
ips.show()
cont.show()

alpCo = eposContra.copy().filter(l_freq=8, h_freq=13).apply_hilbert(envelope=True)
alpIp = eposIpsi.copy().filter(l_freq=8, h_freq=13).apply_hilbert(envelope=True)
alpCoEv = alpCo.copy().average()
alpIpEv = alpIp.copy().average()
alpD = alpIpEv.copy()
alpD.data = alpCoEv.data - alpIpEv.data
alpIp.apply_baseline(baseline=(-0.5, -0.0)).average().plot()
alp = alpD.apply_baseline(baseline=(-0.6, -0)).plot()
alp.show()
tot.show()
alpCo.plot
mne.viz.plot_compare_evokeds([alpCoEv, alpIpEv], combine='mean', vlines=[-0.8, 0])
