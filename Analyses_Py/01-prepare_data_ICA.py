"""
=============================
01. Prepare data for ICA
=============================

Prepares EEG data for further processing steps. 

TODO: Write doc
"""

import os
import os.path as op
import numpy as np
import mne
from pathlib import Path

# define dummy subject:
#subsub = 'VME_S27'
# define subject:
subsub_list = ['VME_S26', 'VME_S27'] #'VME_S12', 'VME_S25'
               # ['VME_S01', 'VME_S02', 'VME_S03', 'VME_S05', 
               # 'VME_S06', 'VME_S07', 'VME_S08', 'VME_S09', 
               # 'VME_S10', 
               #'VME_S13', 'VME_S16', 
               #'VME_S17', 'VME_S18', 'VME_S20', 'VME_S22', 
               #'VME_S23', 'VME_S24', 


# set paths:
path_study = Path(os.getcwd()).parents[1] #str(Path(__file__).parents[2])
# note: returns Path object >> cast for string

path_data = os.path.join(path_study, 'Data')
path_inp = os.path.join(path_data, 'DataMNE', 'EEG', '00_raw')
path_outp_ev = op.join(path_data, 'DataMNE', 'EEG', '01_events')
path_outp_prep = op.join(path_data, 'DataMNE', 'EEG', '02_prepared')
path_outp_filt = op.join(path_data, 'DataMNE', 'EEG', '03_filt')
path_outp_epo = op.join(path_data, 'DataMNE', 'EEG', '04_epo')

for pp in [path_outp_ev, path_outp_prep, path_outp_filt, path_outp_epo]:
    if not op.exists(pp):
        os.makedirs(pp)
        print('creating dir: ' + pp)

def get_events(subID, save_eeg=True, save_eve_to_file=True):
    fname_inp = op.join(path_inp, subID + '-raw.fif')
    raw = mne.io.read_raw_fif(fname_inp)
    events, event_id = mne.events_from_annotations(raw)
    fname_eve = op.join(path_outp_ev, subID + '-eve.fif')
    mne.write_events(fname_eve, events)
    return raw, events, event_id


def calc_eog_chans(data_raw):
    # calculate and add HEOG and VEOG channels:
    # TODO: Review if acceptable or needs refactoring

    # if IO1 is not present, the old labels shall be used
    if not 'IO1' in data_raw.ch_names:
        rn_ch_dict = {
            'AF7': 'IO1',
            'AF8': 'IO2', 
            'FT9': 'LO1',
            'FT10': 'LO2' 
        }
        data_raw.rename_channels(rn_ch_dict) 
        print('renaming eog channels.')

    # check which labels were given to EOG electrodes:
    eog_dict = {
        'vertical_chans' : {
            'left': ['Fp1', 'IO1'], 
            'right': ['Fp2', 'IO2']
        }, 
        'horizontal_chans' : {
            'left': ['LO1'], 
            'right': ['LO2']
        }
    }

    # calculate bipolar EOG chans:
    raw.load_data()
    #VEOGl = raw.copy().pick_channels(['Fp1', 'IO1']) 
    #VEOGr = raw.copy().pick_channels(['Fp2', 'IO2']) 
    dataL = raw.get_data(['Fp1']) - raw.get_data(['IO1']) #VEOGl.get_data(['Fp1']) - VEOGl.get_data(['IO1']) 
    dataR = raw.get_data(['Fp2']) - raw.get_data(['IO2']) #VEOGr.get_data(['Fp2']) - VEOGr.get_data(['IO2']) 
    dataVEOG = np.stack((dataL,dataR), axis=0).mean(0)
    #HEOG = raw.copy().pick_channels(['LO1', 'LO2']) 
    dataHEOG = raw.get_data(['LO1']) - raw.get_data(['LO2']) #HEOG.get_data(['LO1']) - HEOG.get_data(['LO2'])
    dataEOG = np.concatenate((dataVEOG, dataHEOG), axis=0)
    info = mne.create_info(ch_names=['VEOG', 'HEOG'], sfreq=raw.info['sfreq'], ch_types=['eog', 'eog'])
    rawEOG = mne.io.RawArray(dataEOG, info=info)
    raw.add_channels([rawEOG], force_update_info=True)
    # set chan type of original channels to EEG:
    ch_type_dict = {
        'IO1': 'eeg',
        'IO2': 'eeg', 
        'LO1': 'eeg', 
        'LO2': 'eeg'
    }
    data_raw.set_channel_types(ch_type_dict)

    return data_raw

def set_ecg_chan(data_raw):
    # TODO: check if this changes the original object
    ch_type_dict = {
        'ECG': 'ecg'
    }
    data_raw.set_channel_types(ch_type_dict)


def save_data(data, filename, path, append=''):
    ff = op.join(path, filename + append + '.fif')
    #print("Saving %s ..." % ff)
    data.save(fname=ff, overwrite=True)

def load_data_raw(filename, path):
    ff = op.join(path, filename + '.fif')
    return mne.io.Raw(ff)
    

def setup_event_structures(events_, event_id_, srate_):
    # TODO: This one is quite wild and could use refacoring!

    # Define relevant events:
    targ_evs_orig = [i for i in np.arange(150, 174)]
    targ_evs = [event_id_['Stimulus/S%03d' % ss] for ss in targ_evs_orig]

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

    # clean from double markers in trials with PostureCalib:
    trig_postcal = event_id_['Stimulus/S228']

    for i in range(len(events_)):
        if events_[i,2] in targ_evs and events_[i+2,2] == trig_postcal:
            events_[i:i+2,2] = 999

    # crop off all markers before first and after last exp block:
    # TODO: refactor this
    key_b1_start = 'Stimulus/S208'
    if (key_b1_start in event_id_):
        trig_b1_start = event_id_[key_b1_start]
    else:
        trig_b1_start = events[0][2] #use first event ever in case of doubt
        print("Warning: No event START BLOCK01 found. Using first event in structure.")
    # same for end of last block:
    key_b10_end = 'Stimulus/S247'
    if (key_b10_end in event_id_):
        trig_b10_end = event_id_[key_b10_end]
    else:
        trig_b10_end = events[-1][2]
        print("Warning: No event END BLOCK10 found. Using last event in structure.")
    rel_evs = events_[:,2]
    idx_start = np.where(rel_evs == trig_b1_start)[0][0] #use first element
    idx_end = np.where(rel_evs == trig_b10_end)[0][-1] #use last element to get last instance
    events_ = events_[idx_start:idx_end+1,:]

    # set up event arrays:
    events_fix_ = np.array([ev for ev in events_ if ev[2] in targ_evs])
    # add duration of blinking interval:
    # FIXME: replace 800 with variable 
    events_fix_[:,0] = events_fix_[:,0] + 800*srate_/1000

    events_cue_ = np.array([ev for ev in events_ if ev[2] == event_id_['Stimulus/S  1']])
    events_stimon_ = np.array([ev for ev in events_ if ev[2] == event_id_['Stimulus/S  2']])
    events_stimon_[:,2] = events_fix_[:,2]
    events_cue_[:,2] = events_fix_[:,2]

    return events_fix_, events_cue_, events_stimon_

#FIXME: event_id
def extract_epochs_ICA(raw_data, events, event_id_):
    # filter the data:
    filtered = raw_data.load_data().filter(l_freq=1, h_freq=40)
    epos_ica_ = mne.Epochs(filtered, 
                        events, 
                        event_id=event_id_, 
                        tmin=-1.8, 
                        tmax=2.2, 
                        baseline=None,
                        preload=False)
    return epos_ica_

#FIXME: event_id
def extract_epochs_stimon(raw_data, events, event_id_):
    # filter the data:
    filtered = raw_data.load_data().filter(l_freq=0.01, h_freq=40)
    epos_stimon_ = mne.Epochs(filtered, 
                        events, 
                        event_id=event_id_, 
                        tmin=-1, 
                        tmax=2.7, 
                        baseline=None,
                        preload=False)
    return epos_stimon_

def extract_epochs_cue(raw_data, events, event_id_):
    # filter the data:
    filtered = raw_data.load_data().filter(l_freq=0.01, h_freq=40)
    epos_cue_ = mne.Epochs(filtered, 
                        events, 
                        event_id=event_id_, 
                        tmin=-1, 
                        tmax=1.5, 
                        baseline=None,
                        preload=False)
    return epos_cue_



######################################################################################################
for subsub in subsub_list:
    raw, events, event_id = get_events(subsub)
    raw = calc_eog_chans(raw)   
    set_ecg_chan(raw)
    #save_data(raw, subsub, path_outp_prep, append='-raw') #TODO: replace by helper func

    srate = raw.info['sfreq']
    events_fix, events_cue, events_stimon = setup_event_structures(events, event_id, srate)

    event_id_fix = {key: event_id[key] for key in event_id if event_id[key] in events_fix[:,2]}
    event_id_cue = {key: event_id[key] for key in event_id if event_id[key] in events_cue[:,2]}
    event_id_stimon = {key: event_id[key] for key in event_id if event_id[key] in events_stimon[:,2]}


    """ 
    epos_ica = extract_epochs_ICA(raw.copy(), events_stimon, event_id_stimon)
    save_data(epos_ica, subsub + '-forica', path_outp_epo, '-epo')

    epos_stimon = extract_epochs_stimon(raw, events_stimon, event_id_stimon)
    save_data(epos_stimon, subsub + '-stimon', path_outp_epo, '-epo')
    """
    epos_cue = extract_epochs_cue(raw.copy(), events_cue, event_id_cue)
    save_data(epos_cue, subsub + '-cue', path_outp_epo, '-epo')
# epoCueOn = mne.Epochs(filtered, 
#                         events, 
#                         event_id=event_id['Stimulus/S  1'], #=targ_evs, #
#                         tmin=-0.5, 
#                         tmax=2.2, 
#                         baseline=(None,0),
#                         preload=True)
