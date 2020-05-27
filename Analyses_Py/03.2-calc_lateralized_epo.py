"""
=============================
03. Calculate CDA
=============================

TODO: Write doc
"""

import os
import os.path as op
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mne
from pathlib import Path
from library import helpers, config


def write_mean_amp_to_file(ID):
    #conds = ['LoadHighEccS', 'LoadHighEccM', 'LoadHighEccL', 'LoadLowEccS', 'LoadLowEccM', 'LoadLowEccL']
    #data = [str(mean_amplitues_dict[key] * 1000) for key in conds]
    file_mean_amp = op.join(config.path_evokeds_summaries, ID + '_mean_amp_CDA.csv')
    with open(file_mean_amp, 'w') as ffile:
        for load in ['LoadHigh', 'LoadLow']:
            for ecc in ['EccS', 'EccM', 'EccL']:
                data_txt = ";".join([ID, load, ecc, format(mean_amplitues_dict[load+ecc], '.8f')])
                ffile.write(data_txt + "\n")

sub_list = np.setdiff1d(np.arange(1,28), config.ids_missing_subjects)
#sub_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27]

chans_CDA_dict = {'Left': ['P3', 'P5', 'PO3', 'PO7', 'O1'], 
                  'Right': ['P4', 'P6', 'PO4', 'PO8', 'O2']}
chans_CDA_all = [ch for v in list(chans_CDA_dict.values()) for ch in v]

for subNr in sub_list:
    subID = 'VME_S%02d' % subNr
    # subID = 'VME_S02'

    # Load data:
    data = helpers.load_data(subID + '-stimon-postica', config.path_postICA, '-epo')

    # Make dict summarizing chans in regions 'Left', 'Midline', 'Right'
    region_dict = mne.channels.make_1020_channel_selections(data.info)
    # Second dict that contains the channel names instead of their indexes:
    chnames_eeg = data.copy().pick_types(eeg=True).ch_names
    region_dict_chnames = dict()
    for k in region_dict.keys(): 
        region_dict_chnames[k] = [chnames_eeg[i] for i in region_dict[k]]
    
    # Defina a dict that translates all lateralized channels to their 
    # counterpart on the other hemisphere:
    rename_dict_mirror = {k: v for k,v in zip(region_dict_chnames['Right'] + region_dict_chnames['Left'], 
                                              region_dict_chnames['Left'] + region_dict_chnames['Right'])}


    # Define relevant events:
    #TODO: Put this to helpers:
    targ_evs = [i for i in data.event_id.keys()]
    epo_keys = ['CueL', 'CueR', 'LoadLow', 'LoadHigh', 'EccS', 'EccM', 'EccL']

    event_dict = {key: [] for key in epo_keys}
    for ev in targ_evs:
        ev_int = int(ev[-3:]) #int(ev.split('/S')[1])
        ev0 = ev_int - 150
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

    epos_dict = dict()
    evoked_dict = dict()

    # Separate into epochs dep. on cue direction:
    epos_dict["CueL"] = data.copy()[event_dict['CueL']]
    epos_dict["CueR"] = data.copy()[event_dict['CueR']]

    # We want to pretend that all contralateral channels are on the left side of the head:
    rename_dict_R2L = {k: v for k,v in zip(region_dict_chnames['Right'], region_dict_chnames['Left'])}
    # opposite direction:
    rename_dict_L2R = {k: v for v, k in rename_dict_R2L.items()}

    eposContra_CueL = epos_dict["CueL"].copy().pick_channels(region_dict_chnames['Right'], ordered=True)
    eposContra_CueR = epos_dict["CueR"].copy().pick_channels(region_dict_chnames['Left'], ordered=True)
    eposContra_CueL.rename_channels(rename_dict_R2L)
    # we still have to correct the channel locations: 
    eposContra_CueL.info['chs'] = eposContra_CueR.copy().info['chs']
    eposContra_CueL.info['dig'] = eposContra_CueR.copy().info['dig']

    #rename_dict_mirror_cda = {k: v for k, v in rename_dict_mirror.items() if k in chans_CDA[0] or k in chans_CDA[1]}
    epos_mirrored = epos_dict['CueR'].copy().rename_channels(rename_dict_mirror)
    epos_mirrored.reorder_channels(epos_dict['CueL'].ch_names)
    epos_mirrored.info['chs'] = epos_dict['CueL'].copy().info['chs']
    epos_mirrored.info['dig'] = epos_dict['CueL'].copy().info['dig']

    epos_sorted = mne.concatenate_epochs([epos_dict['CueL'].copy(), epos_mirrored.copy()], add_offset=False)

    epos_CDA = epos_sorted.copy()
    _contra = epos_CDA.copy().pick_types(eeg=True).pick_channels(region_dict_chnames['Right'], ordered=True)
    _ipsi = epos_CDA.copy().pick_types(eeg=True).pick_channels(region_dict_chnames['Left'], ordered=True)
    # make sure that the channels are in the same order:
    assert _contra.ch_names == [rename_dict_mirror[k] for k in _ipsi.ch_names], 'Channel names are not identical'
    data_contra = _contra._data
    data_ipsi = _ipsi._data
    data_diff = data_contra - data_ipsi
    epos_CDA.pick_types(eeg=True).pick_channels(region_dict_chnames['Right'], ordered=True)
    # make sure that the channels are in the same order:
    assert epos_CDA.ch_names == _contra.ch_names
    epos_CDA._data = data_diff

    mne.viz.plot_compare_evokeds([epos_CDA[event_dict['LoadLow']].average(), epos_CDA[event_dict['LoadHigh']].average()], combine='mean', picks = ['F2', 'Fp2', 'AF4', 'FC6', 'FT8'])

    epos_dict['Contra'] = mne.concatenate_epochs([eposContra_CueR.copy(), eposContra_CueL.copy()], add_offset=False)
    epos_dict["RoiContraLoadLow"] = epos_dict["RoiContra"][event_dict['LoadLow']]
    epos_dict["RoiContraLoadHigh"] = epos_dict["RoiContra"][event_dict['LoadHigh']]

    eposIpsiL = epos_dict["CueL"].copy().pick_channels(chans_CDA[0])
    eposIpsiR = epos_dict["CueR"].copy().pick_channels(chans_CDA[1])
    eposIpsiL.rename_channels(rename_dict)
    epos_dict["RoiIpsi"] = mne.concatenate_epochs([eposIpsiL.copy(), eposIpsiR], add_offset=False)
    epos_dict["RoiIpsiLoadLow"] = epos_dict["RoiIpsi"][event_dict['LoadLow']]
    epos_dict["RoiIpsiLoadHigh"] = epos_dict["RoiIpsi"][event_dict['LoadHigh']]


    for roi in ["RoiContra", "RoiIpsi"]:
        for load in ["LoadLow", "LoadHigh"]:
            for ecc in ["EccS", "EccM", "EccL"]:
                epos_dict[roi+load+ecc] = epos_dict[roi][event_dict[load]][event_dict[ecc]]


    epos_dict["CDA"] = epos_dict["RoiContra"].copy()
    epos_dict["CDA"]._data = epos_dict["RoiContra"]._data - epos_dict["RoiIpsi"]._data

    epos_dict["LoadHigh"] = epos_dict["CDA"][event_dict['LoadHigh']]
    epos_dict["LoadLow"] = epos_dict["CDA"][event_dict['LoadLow']]

    epos_dict["EccS"] = epos_dict["CDA"][event_dict['EccS']]
    epos_dict["EccM"] = epos_dict["CDA"][event_dict['EccM']]
    epos_dict["EccL"] = epos_dict["CDA"][event_dict['EccL']]

    epos_dict["LoadHighEccS"] = epos_dict["LoadHigh"][event_dict['EccS']]
    epos_dict["LoadHighEccM"] = epos_dict["LoadHigh"][event_dict['EccM']]
    epos_dict["LoadHighEccL"] = epos_dict["LoadHigh"][event_dict['EccL']]

    epos_dict["LoadLowEccS"] = epos_dict["LoadLow"][event_dict['EccS']]
    epos_dict["LoadLowEccM"] = epos_dict["LoadLow"][event_dict['EccM']]
    epos_dict["LoadLowEccL"] = epos_dict["LoadLow"][event_dict['EccL']]

    # Save Epos:
    for key in epos_dict:
        helpers.save_data(epos_dict[key], subID, 
                        config.path_epos_sorted + '/' + key, append='-epo')



    for load in ['LoadHigh', 'LoadLow']:
        evoked_dict[load] = epos_dict[load].average()

        for ecc in ['EccS', 'EccM', 'EccL']:
            if not ecc in evoked_dict:
                evoked_dict[ecc] = epos_dict[ecc].average()
            
            evoked_dict[load + ecc] = epos_dict[load + ecc].average()

    # store mean amplitudes:
    mean_amplitues_dict = dict()

    for cond in evoked_dict.keys():
        dat = evoked_dict[cond]._data
        tms_idx = np.where((0.400 < evoked_dict[cond].times) & (evoked_dict[cond].times < 1.45))
        mean_amplitues_dict[cond] = np.mean(np.mean(evoked_dict[cond]._data, 0)[tms_idx])



    write_mean_amp_to_file(subID)


    #TODO: Check if we're safe and delete following:
    # evoHi = epos_dict["LoadHigh"].average()
    # evoLo = epos_dict["LoadLow"].average()

    # evoS = EccS.average()
    # evoM = EccM.average()
    # evoL = EccL.average()

    # evoHiS = LoadHighEccS.average()
    # evoHiM = LoadHighEccM.average()
    # evoHiL = LoadHighEccL.average()

    # evoLoS = LoadLowEccS.average()
    # evoLoM = LoadLowEccM.average()
    # evoLoL = LoadLowEccL.average()

    ff = op.join(config.path_evokeds, subID + '-ave.fif')
    mne.write_evokeds(ff, [evoked_dict[coo] for coo in config.factor_levels])

    #TODO: replace with sequence in config.fac_levs 
    ## old sequence (used for S07, S22):
    # [evoHiL, evoHiM, evoHiS, evoLoL, evoLoM, evoLoS, evoHi, evoLo, evoL, evoM, evoS])

