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

subsub = 'VME_S27'
sub_list = [1,2,3,5,6,7,8,9,10,13,16,17,18,20,22,23,24,26,27]


subsub_list = ['VME_S01', 'VME_S02', 'VME_S03', 'VME_S05', 
               'VME_S06', 'VME_S07', 'VME_S08', 'VME_S09', 
               'VME_S10', 
               'VME_S13', 'VME_S16', 
               'VME_S17', 'VME_S18', 'VME_S20', 'VME_S22', 
               'VME_S23', 'VME_S24', 'VME_S26', 'VME_S27']

for subsub in subsub_list:

    chans_CDA = [['P3', 'P5', 'PO3', 'PO7', 'O1'], 
                ['P4', 'P6', 'PO4', 'PO8', 'O2']]


    data = helpers.load_data(subsub + '-cue-postica', config.path_postICA, '-epo')

    # remove baseline:
    # data.apply_baseline((-0.3,0))

    # Keep only CDA channels:
    ch_cda = [ch for sublist in chans_CDA for ch in sublist]
    data.pick_channels(ch_cda)

    # reject bad epochs:
    rej_dict = dict(eeg = 100e-6)
    data.drop_bad(rej_dict)
    perc_bads_dropped = data.drop_log_stats()


    # Define relevant events:
    #TODO: Check if this does the correct thing:
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


    epos_dict["CueL"] = data[event_dict['CueL']]
    epos_dict["CueR"] = data[event_dict['CueR']]

    rename_dict = {chans_CDA[0][i]: chans_CDA[1][i]  for i in range(len(chans_CDA[0]))}

    eposContraL = epos_dict["CueL"].copy().pick_channels(chans_CDA[1])
    eposContraR = epos_dict["CueR"].copy().pick_channels(chans_CDA[0])
    eposContraR.rename_channels(rename_dict)
    epos_dict["RoiContra"] = mne.concatenate_epochs([eposContraL.copy(), eposContraR], add_offset=False)
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
        helpers.save_data(epos_dict[key], subsub, 
                        config.path_epos_sorted_cue + '/' + key, append='-epo')



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

    def write_mean_amp_to_file(ID):
        #conds = ['LoadHighEccS', 'LoadHighEccM', 'LoadHighEccL', 'LoadLowEccS', 'LoadLowEccM', 'LoadLowEccL']
        #data = [str(mean_amplitues_dict[key] * 1000) for key in conds]
        file_mean_amp = op.join(config.path_evokeds_summaries, ID + '_mean_amp_CDA.csv')
        with open(file_mean_amp, 'w') as ffile:
            for load in ['LoadHigh', 'LoadLow']:
                for ecc in ['EccS', 'EccM', 'EccL']:
                    data_txt = ";".join([ID, load, ecc, str(mean_amplitues_dict[load+ecc]*10e6)])
                    ffile.write(data_txt + "\n")

    #write_mean_amp_to_file(subsub)


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

    ff = op.join(config.path_evokeds_cue, subsub + '-ave.fif')
    mne.write_evokeds(ff, [evoked_dict[coo] for coo in config.factor_levels])

    #TODO: replace with sequence in config.fac_levs 
    ## old sequence (used for S07, S22):
    # [evoHiL, evoHiM, evoHiS, evoLoL, evoLoM, evoLoS, evoHi, evoLo, evoL, evoM, evoS])

    # following code ztaken from:
    # https://github.com/mne-tools/mne-biomag-group-demo/blob/master/scripts/processing/11-group_average_sensors.py

    all_evokeds = [list() for _ in range(11)] 
    for sub in sub_list: #[3, 7, 22]:
        subID = 'VME_S%02d' % sub
        evokeds = mne.read_evokeds(op.join(config.path_evokeds_cue, subID + '-ave.fif'))
        for idx, evoked in enumerate(evokeds):
            all_evokeds[idx].append(evoked)  # Insert to the container

    for idx, evokeds in enumerate(all_evokeds):
        all_evokeds[idx] = mne.combine_evoked(evokeds, 'nave')  # Combine subjects

    print(f'\n###################\nPercent of bad epos dropped: {perc_bads_dropped} \n###################\n')

    # Main effect Load:
    res = mne.viz.plot_compare_evokeds(dict(High = all_evokeds[config.factor_dict['LoadHigh']], 
                                    Low = all_evokeds[config.factor_dict['LoadLow']]), 
                                combine='mean', 
                                vlines=[0, 0.8], 
                                ci=True,
                                #ylim=dict(eeg=[-1.5,1.5]),
                                title="Memory Load"
                                )
    ff = 'MainEff_Load.png'
    res[0].savefig(op.join(config.path_evokeds_cue, 'Plots', ff))


    # Main effect Ecc:
    """
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
    """

    # Interaction:
    """ 
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
        """





