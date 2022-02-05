"""
=============================
01.0 Prepare data for next steps
=============================

Prepares EEG data for further processing steps & extract events. 

"""


import os
import os.path as op
import pickle
import sys
import numpy as np
import mne
from pathlib import Path
from library import helpers, config


def get_data_and_events(subID):
    fname_inp = op.join(config.paths['00_raw'], subID + '-raw.fif')
    raw = mne.io.read_raw_fif(fname_inp)
    events, event_id = mne.events_from_annotations(raw)        
    return raw, events, event_id

def save_events(subID, events, event_id, bad_epos, epo_part):
    fname_eve = op.join(config.paths['01_prepared-events'], '-'.join([subID, epo_part,'eve.fif']))
    mne.write_events(fname_eve, events)
    fname_eve_id = op.join(config.paths['01_prepared-events'], '-'.join([subID, 'event_id.pkl']))
    if event_id is not None:
        with open(fname_eve_id, 'wb') as f:
            pickle.dump(event_id, f)
    fname_bad_epos = op.join(config.paths['01_prepared-events'], '-'.join([subID, 'bad_epos_recording.pkl']))
    with open(fname_bad_epos, 'wb') as f:
        pickle.dump(bad_epos, f)
    return

def calc_eog_chans(data_raw):
    # calculate and add HEOG and VEOG channels:

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

    ## which labels were given to EOG electrodes:
    # 'vertical_chans' : 
    #        'left': ['Fp1', 'IO1'], 
    #        'right': ['Fp2', 'IO2'] 
    # 'horizontal_chans' : 
    #        'left': ['LO1'], 
    #        'right': ['LO2']


    # For a few subjects electrodes 'Fp1' and 'IO1' were mistakenly exchanged. 
    # Let's find out for which and repair it:

    picks = ['Fp1', 'Fp2', 'IO1', 'IO2', 'LO1', 'LO2']
    rr = data_raw.load_data().copy().pick_channels(picks).filter(l_freq = 1, h_freq= 5, picks=['eeg','misc'], verbose=False)
    # Create pseudo epochs to loop over:
    events = mne.make_fixed_length_events(rr, duration=20)
    epochs = mne.Epochs(rr, events, tmin=0.0, tmax=20, baseline = (0,1), verbose= False)
    epochs.load_data().reorder_channels(picks)
    # For each epoch we calculate the correlations between all EOG channels and store it
    holder = [] 
    for epo in epochs:
        epo = epo - epo.copy().mean(axis = 1, keepdims=True)
        tmp = np.corrcoef(epo)
        holder.append(tmp)

    res = np.stack(holder).mean(axis=0)
    
    # correlations of "Fp1" (what could be "IO1")
    idx_fp1 = epochs.ch_names.index('Fp1')
    corrs_fp1 = res[idx_fp1,:]
    # corr with itself is ofc largest, so we put it away
    corrs_fp1[idx_fp1] = -999
    # now get which other chan it correlates most with
    idx_corrmax = np.argmax(corrs_fp1)
    chan_corrmax = epochs.ch_names[idx_corrmax]
    helpers.print_msg(f'What we think is Fp1 actually correlates highly with {chan_corrmax} (r = {str(np.max(corrs_fp1))})).')
    
    if not chan_corrmax == 'Fp2':
        helpers.print_msg('Swopping channels IO1 and Fp1.')
        tmp = data_raw.get_data(picks = ['Fp1', 'IO1'])
        data_raw['Fp1'] = tmp[1]
        data_raw['IO1'] = tmp[0]


    # calculate bipolar EOG chans:
    data_raw.load_data()
    dataL = data_raw.get_data(['Fp1']) - data_raw.get_data(['IO1']) 
    dataR = data_raw.get_data(['Fp2']) - data_raw.get_data(['IO2']) 
    dataVEOG = np.stack((dataL,dataR), axis=0).mean(0)
    
    dataHEOG = data_raw.get_data(['LO1']) - data_raw.get_data(['LO2']) 
    dataEOG = np.concatenate((dataVEOG, dataHEOG), axis=0)
    info = mne.create_info(ch_names=['VEOG', 'HEOG'], sfreq=raw.info['sfreq'], ch_types=['eog', 'eog'])
    rawEOG = mne.io.RawArray(dataEOG, info=info)
    data_raw.add_channels([rawEOG], force_update_info=True)
    # set chan type of original channels to EEG:
    ch_type_dict = {
        'IO1': 'misc',
        'IO2': 'misc', 
        'LO1': 'misc', 
        'LO2': 'misc'
    }
    data_raw.set_channel_types(ch_type_dict)

    return data_raw

def set_ecg_chan(data_raw):
    ch_type_dict = {
        'ECG': 'ecg'
    }
    data_raw.set_channel_types(ch_type_dict)

def load_data_raw(filename, path):
    ff = op.join(path, filename + '.fif')
    return mne.io.Raw(ff)
    

def setup_event_structures(events_, event_id_, srate_):

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

    
    # clean from double markers in restarted trials:
    # same routine as in ET analysis

    # temporary list of trial onsets (fixation Onsets) and StimOnsets:
    tmp_ev_fix = np.array([ev for ev in events_ if ev[2] in targ_evs])
    stimon_times = np.array([ev[0] for ev in events_ if ev[2] == event_id_['Stimulus/S  2']])
    for i in range(1,len(tmp_ev_fix)):
        # We know that for restarted trials the same trial type comes 2x in a row:
        if tmp_ev_fix[i,2] == tmp_ev_fix[i-1,2]: 
            times_between = np.arange(tmp_ev_fix[i-1,0], tmp_ev_fix[i,0])
            # if no StimOnset between these two markers:
            if not len(np.intersect1d(times_between, stimon_times)) > 0:
                # we know that this was a restarted trial, and we overwrite all events between the two 
                # relevant markers:
                idx = np.in1d(events_[:,0], times_between)
                for ee in events_[idx]:
                    ev_name = [n for n,v in event_id_.items() if v == ee[2]]
                    print('Trial restarted: Overwriting event -- ' + str(ev_name))
                events_[idx,2] = 999

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
    events_fix_[:,0] = events_fix_[:,0] + config.times_dict['blink_dur'] * srate_

    events_cue_ = np.array([ev for ev in events_ if ev[2] == event_id_['Stimulus/S  1']])
    events_stimon_ = np.array([ev for ev in events_ if ev[2] == event_id_['Stimulus/S  2']])

    bad_epos = dict()
    # Check if for all relevant events 720 instances were found:
    if any(np.array([len(events_fix_), len(events_cue_), len(events_stimon_)]) < 720):
        
        # Missing triggers that code for the trial type are tricky. It's better to fix that manually instead of silently tring to fix it.
        if (len(events_fix_) != 720): 
            raise ValueError("There is a trigger missing that codes for the trial type. You have to fix this manually, I'm afraid!")
        
        # For CueOnset triggers we can check their distance to the trial onsets (ie, fixation onset triggers). 
        # If no stimonset trigger is found in the relevant time window, this is probably the trial where something went wrong.  
        if (len(events_cue_) != 720): 
            idx = np.argwhere([np.min(np.abs(ev[0] - events_cue_[:,0])) >             
                (config.times_dict['fix_dur'] + 0.1) * srate for ev in events_fix_])
            if (len(idx) != (720 - len(events_cue_))): 
                raise ValueError("There is a timing issue apart from missing triggers. Check your triggers manually!")
            else: 
                for i in idx: 
                    print('Replacing missing trigger for Cue and adding to "bads" list.')
                    new_ev = [int(events_fix_[i[0],0] + config.times_dict['fix_dur'] * srate_), 0, event_id_['Stimulus/S  1']]
                    events_tmp = np.vstack([events_cue_, new_ev])
                    # sort events ascending in time:
                    events_cue_ = events_tmp[events_tmp[:,0].argsort()]
            bad_epos['cue'] = idx.flatten()


        # For StimOnset triggers we can check their distance to the trial onsets (ie, fixation onset triggers). 
        # If no stimonset trigger is found in the relevant time window, this is probably the trial where something went wrong.  
        if (len(events_stimon_) != 720): 
            idx = np.argwhere([np.min(np.abs(ev[0] - events_stimon_[:,0])) >             
            (config.times_dict['fix_dur'] + config.times_dict['cue_dur'] + 0.1) * srate for ev in events_fix_])
            if (len(idx) != (720 - len(events_stimon_))): 
                raise ValueError("There is a timing issue apart from missing triggers. Check your triggers manually!")
            else: 
                for i in idx: 
                    print('Replacing missing trigger for StimOnset and adding to "bads" list.')
                    new_ev = [int(events_fix_[i[0],0] + (config.times_dict['fix_dur'] + config.times_dict['cue_dur']) * srate_), 0, event_id_['Stimulus/S  2']]
                    events_tmp = np.vstack([events_stimon_, new_ev])
                    # sort events ascending in time:
                    events_stimon_ = events_tmp[events_tmp[:,0].argsort()]
            bad_epos['stimon'] = idx.flatten()
    
    # Make sure that trials are in the same order (sorted by time)
    events_fix_ = events_fix_[events_fix_[:,0].argsort()]
    events_stimon_ = events_stimon_[events_stimon_[:,0].argsort()]
    events_cue_ = events_cue_[events_cue_[:,0].argsort()]
    assert len(events_fix_) == len(events_stimon_) == len(events_cue_)
    
    # Add trial type info to last column:
    events_stimon_[:,2] = events_fix_[:,2]
    events_cue_[:,2] = events_fix_[:,2]

    return events_fix_, events_cue_, events_stimon_, bad_epos

#FIXME: event_id
def extract_epochs_ICA(raw_data, events, event_id_, n_jobs = 1):
    # filter the data:
    filtered = raw_data.load_data().filter(l_freq=1, h_freq=40, n_jobs = n_jobs)
    epos_ica_ = mne.Epochs(filtered, 
                        events, 
                        event_id=event_id_, 
                        tmin=-0.6, 
                        tmax=2.3, 
                        baseline=(None,None),
                        preload=False)
    return epos_ica_

#FIXME: event_id
def extract_epochs_stimon(raw_data, events, event_id_, bad_epos_, n_jobs = 1):
    # filter the data:
    filtered = raw_data.load_data().filter(l_freq=0.01, h_freq=40, n_jobs = n_jobs)
    epos_stimon_ = mne.Epochs(filtered, 
                        events, 
                        event_id=event_id_, 
                        tmin=-0.6, 
                        tmax=2.3, 
                        baseline=None,
                        preload=False)
    epos_stimon_.drop(bad_epos_, 'BADRECORDING')
    return epos_stimon_

def extract_epochs_cue(raw_data, events, event_id_, tmin_, tmax_, bad_epos_, n_jobs = 1):
    # filter the data:
    filtered = raw_data.load_data().filter(l_freq=0.01, h_freq=40, n_jobs = n_jobs)
    epos_cue_ = mne.Epochs(filtered, 
                        events, 
                        event_id=event_id_, 
                        tmin=tmin_, 
                        tmax=tmax_, 
                        baseline=None,
                        preload=False)
    epos_cue_.drop(bad_epos_, 'BADRECORDING')
    return epos_cue_

def extract_epochs_fulllength(raw_data, events, event_id_, tmin_, tmax_, bad_epos_, n_jobs = 1):
    # filter the data:
    filtered = raw_data.load_data().filter(l_freq=0.01, h_freq=40, n_jobs = n_jobs)
    epos_cue_ = mne.Epochs(filtered, 
                        events, 
                        event_id=event_id_, 
                        tmin=tmin_, 
                        tmax=tmax_, 
                        baseline=None,
                        preload=False)
    epos_cue_.drop(bad_epos_, 'BADRECORDING')
    return epos_cue_


######################################################################################################

## Full procedure:
sub_list = np.setdiff1d(np.arange(1,config.n_subjects_total+1), config.ids_missing_subjects)
sub_list_str = ['VME_S%02d' % sub for sub in sub_list]

## to run a single subject, modify and uncomment the following line:
# sub_list_str = ['VME_S01']

#sub_list = np.array([sub_list_str[job_nr]])

for idx, subID in enumerate(sub_list_str):
    helpers.print_msg('Processing subject ' + subID + '.')
    
    # Get data:
    raw, events, event_id = get_data_and_events(subID)
    
    # Calculate EOG channels & set chan type:
    raw = calc_eog_chans(raw)   
    
    # Set ECG chan type:
    set_ecg_chan(raw)
    
    # Save prepared data:
    helpers.save_data(raw,
                      subID + '-prepared',
                      config.paths['01_prepared'],
                      append='-raw') 

    print("***Saving events:***")
    # Extract and save events:
    srate = raw.info['sfreq']
    events_fix, events_cue, events_stimon, bad_epos = setup_event_structures(events, event_id, srate)
    save_events(subID, events_stimon, event_id=event_id, bad_epos=bad_epos, epo_part='stimon')
    save_events(subID, events_cue, event_id=None, bad_epos=bad_epos, epo_part='cue') # enough to save event_id once
    save_events(subID, events_fix, event_id=None, bad_epos=bad_epos, epo_part='fix')
    
    

#     event_id_fix    = {key: event_id[key] for key in event_id if event_id[key] in events_fix[:,2]}
#     event_id_cue    = {key: event_id[key] for key in event_id if event_id[key] in events_cue[:,2]}
#     event_id_stimon = {key: event_id[key] for key in event_id if event_id[key] in events_stimon[:,2]}

#     epos_ica = extract_epochs_ICA(raw.copy(), 
#                                   events_stimon, 
#                                   event_id_stimon, 
#                                   n_jobs = config.n_jobs)
#     helpers.save_data(epos_ica,
#                       subID + '-forica',
#                       path_outp_epo, 
#                       '-epo')

#     epos_stimon = extract_epochs_stimon(raw.copy(),
#                                         events_stimon,
#                                         event_id_stimon,
#                                         bad_epos_ = bad_epos.get('stimon',[]),
#                                         n_jobs = config.n_jobs)
#     helpers.save_data(epos_stimon,
#                       subID + '-stimon',
#                       path_outp_epo,
#                       append='-epo')
    
#     epos_cue = extract_epochs_cue(raw.copy(),
#                                   events_cue,
#                                   event_id_cue,
#                                   tmin_ = -0.6, 
#                                   tmax_ = 1,
#                                   bad_epos_ = bad_epos.get('cue', []),
#                                   n_jobs = config.n_jobs)
#     helpers.save_data(epos_cue,
#                       subID + '-cue',
#                       path_outp_epo,
#                       append='-epo')
    
#     epos_fulllength = extract_epochs_fulllength(raw.copy(),
#                                                 events_cue,
#                                                 event_id_cue,
#                                                 tmin_ = -0.6,
#                                                 tmax_ = 3.3,
#                                                 bad_epos_ = np.unique([v for k in bad_epos.keys() for v in bad_epos.get(k, [])]),
#                                                 n_jobs = config.n_jobs)
#     helpers.save_data(epos_fulllength,
#                       subID + '-fulllength',
#                       path_outp_epo,
#                       append='-epo')
    
