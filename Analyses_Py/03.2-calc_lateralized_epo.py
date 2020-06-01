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

    for epo_part in ['cue']:

        # Load data:
        data = helpers.load_data(subID + '-' + epo_part + '-postica', config.path_postICA, '-epo')

        # Make dict summarizing chans in regions 'Left', 'Midline', 'Right'
        region_dict = mne.channels.make_1020_channel_selections(data.info)
        # Second dict that contains the (eeg!) channel names instead of their indexes:
        chnames_eeg = data.copy().pick_types(eeg=True).ch_names
        region_dict_chnames = dict()
        for k in region_dict.keys(): 
            region_dict_chnames[k] = [chnames_eeg[i] for i in region_dict[k]]
        
        # Defina a dict that translates all lateralized channels to their 
        # counterpart on the other hemisphere:
        rename_dict_mirror = {k: v for k,v in zip(region_dict_chnames['Right'] + region_dict_chnames['Left'], 
                                                region_dict_chnames['Left'] + region_dict_chnames['Right'])}

        # Define relevant events:
        event_dict = helpers.get_event_dict(data.event_id.keys())

        epos_dict = dict()
        evoked_dict = dict()

        # Separate into epochs dep. on cue direction:
        epos_dict["CueL"] = data.copy()[event_dict['CueL']]
        epos_dict["CueR"] = data.copy()[event_dict['CueR']]

        # We "mirror" all electrodes for the trials with a leftward pointing cue to collapse 
        # over the 2 cue directions: 
        epos_mirrored = epos_dict['CueL'].copy().rename_channels(rename_dict_mirror)
        # Establish same order:
        epos_mirrored.reorder_channels(epos_dict['CueR'].ch_names)
        # Overwrite localization info:
        epos_mirrored.info['chs'] = epos_dict['CueR'].copy().info['chs']
        epos_mirrored.info['dig'] = epos_dict['CueR'].copy().info['dig']
        # Change sign of HEOG - so that by convention all eye movements to the "right" are now 
        # going to the stimulus:
        idx_heog = epos_mirrored.ch_names.index('HEOG') 
        epos_mirrored._data[:,idx_heog,:] *= -1

        # Combine the two conditions:
        epos_collapsed = mne.concatenate_epochs([epos_dict['CueR'].copy(), epos_mirrored.copy()], add_offset=False)
        # Notice that: 
        #   (A) .drop_log and .selection of this object now have double the length.
        #   (B) Channels on the "right" are now "Ipsi-", chans on the left are "Contralateral" to 
        #       the cued stimulus. 

        for _dict in [region_dict, region_dict_chnames]:
            for n, o in zip(['Ipsi', 'Contra'], ['Right', 'Left']):
                _dict.update({n: _dict[o]})   

        # Now: Construct an epos object, that contains only the difference waves.
        # These will be stored in the channels over the left hemisphere.
        # Other channels will be dropped.
        epos_CDA = epos_collapsed.copy()
        _contra = epos_CDA.copy().pick_types(eeg=True).pick_channels(region_dict_chnames['Contra'], ordered=True)
        _ipsi = epos_CDA.copy().pick_types(eeg=True).pick_channels(region_dict_chnames['Ipsi'], ordered=True)
        # make sure that the channels are in the same order:
        assert _contra.ch_names == [rename_dict_mirror[k] for k in _ipsi.ch_names], 'Channel names are not identical'
        data_contra = _contra._data
        data_ipsi = _ipsi._data
        data_diff = data_contra - data_ipsi
        epos_CDA.pick_types(eeg=True).pick_channels(region_dict_chnames['Left'], ordered=True)
        # make sure that the channels are in the same order:
        assert epos_CDA.ch_names == _contra.ch_names
        epos_CDA._data = data_diff

        ## You can use this to visualize:
        # mne.viz.plot_compare_evokeds([epos_CDA[event_dict['EccL']].average(), epos_CDA[event_dict['EccS']].average()], combine='mean', picks = chans_CDA_dict['Left'])

        helpers.save_data(epos_collapsed, subID, 
                        config.path_epos_sorted + '/' + epo_part + '/' + 'collapsed', 
                        append='-epo')
                        
        helpers.save_data(epos_CDA, subID, 
                        config.path_epos_sorted + '/' + epo_part + '/' + 'difference', 
                        append='-epo')
                        
    

#TODO: Write out mean amplitudes (per trial) and evokeds

    # for load in ['LoadHigh', 'LoadLow']:
    #     evoked_dict[load] = epos_dict[load].average()

    #     for ecc in ['EccS', 'EccM', 'EccL']:
    #         if not ecc in evoked_dict:
    #             evoked_dict[ecc] = epos_dict[ecc].average()
            
    #         evoked_dict[load + ecc] = epos_dict[load + ecc].average()

    # # store mean amplitudes:
    # mean_amplitues_dict = dict()

    # for cond in evoked_dict.keys():
    #     dat = evoked_dict[cond]._data
    #     tms_idx = np.where((0.400 < evoked_dict[cond].times) & (evoked_dict[cond].times < 1.45))
    #     mean_amplitues_dict[cond] = np.mean(np.mean(evoked_dict[cond]._data, 0)[tms_idx])



    # write_mean_amp_to_file(subID)


    # ff = op.join(config.path_evokeds, subID + '-ave.fif')
    # mne.write_evokeds(ff, [evoked_dict[coo] for coo in config.factor_levels])

