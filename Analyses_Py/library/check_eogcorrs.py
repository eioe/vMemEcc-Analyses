
import os
import os.path as op
import numpy as np
import mne
from pathlib import Path
#from library import helpers
from datetime import datetime
from matplotlib import pyplot as plt

# define subject:
#subsub = 'VME_S27'
subsub_list = ['VME_S01', 'VME_S02', 'VME_S03', 'VME_S04', 'VME_S05', 
               'VME_S06', 'VME_S07', 'VME_S08', 'VME_S09', 
               'VME_S10', 'VME_S12', 'VME_S13', 'VME_S15', 'VME_S16', 
               'VME_S17', 'VME_S18', 'VME_S20', 'VME_S21', 'VME_S22', 
               'VME_S23', 'VME_S24', 'VME_S25', 'VME_S26', 'VME_S27'] #'VME_S12', 'VME_S25'
               

# set paths:
path_study = Path(os.getcwd()).parents[1] #str(Path(__file__).parents[2])
# note: returns Path object >> cast for string

path_data = os.path.join(path_study, 'Data')
path_inp = os.path.join(path_data, 'DataMNE', 'EEG', '00_raw')


    # get BP [1; 40Hz] filtered data to train ICA:
    #data_forica = mne.read_epochs(fname=op.join(path_prep_epo, subsub + '-forica-epo.fif'))


def load_data_raw(filename, path):
    ff = op.join(path, filename + '.fif')
    return mne.io.Raw(ff)
   

def get_events(subID, save_eeg=True, save_eve_to_file=True):
    fname_inp = op.join(path_inp, subID + '-raw.fif')
    raw = mne.io.read_raw_fif(fname_inp)
    events, event_id = mne.events_from_annotations(raw)
    #fname_eve = op.join(path_outp_ev, subID + '-eve.fif')
    #mne.write_events(fname_eve, events)
    return raw, events, event_id

prob_bears = []
pb_res = []

for subsub in subsub_list:
    raw, events, event_id = get_events(subsub)

    middle = int(len(raw.times)/2)
    length = 5000
    picks = ['Fp1', 'Fp2', 'IO1', 'IO2']
    rr = raw.load_data().copy().pick_channels(picks).filter(l_freq = 1, h_freq= 5, picks=['eeg','misc'])
    events = mne.make_fixed_length_events(rr, duration=20)
    epochs = mne.Epochs(rr, events, tmin=0.0, tmax=20, baseline = (0,1))


    holder = [] 
    for epo in epochs:
        fp1 = epo
        fp1 = fp1 - fp1.mean(axis = 1, keepdims=True)
        # plt.plot(epochs.times, fp1.transpose())
        # plt.legend(picks)
        # plt.show()
        tmp = np.corrcoef(fp1)
        holder.append(tmp)

    res = np.stack(holder).mean(axis=0)
    cross = np.concatenate([res[0,2:4].flatten(), res[1,2:4].flatten()]).mean() #, [1,2], [1,3]])
    equal = np.mean([res[0,1], res[2,3]]) 
    fp_cor = res[0,1]
    if not fp_cor > 0.8: #(cross<0 and equal > 0):
        print('Danger for ' + subsub)
        prob_bears.append(subsub)
        pb_res.append(res)

print(prob_bears)