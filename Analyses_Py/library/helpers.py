
import os
import os.path as op
from pathlib import Path

import mne


def load_data(filename, path, append):
    if append == '-raw':
        ff = op.join(path, filename + append + '.fif')
        return mne.io.Raw(ff)
    elif append == '-epo':
        ff = op.join(path, filename + append + '.fif')
        return mne.read_epochs(ff) 
    else :
        print('This append (%s) is not yet implemented.' % append)


    

def save_data(data, filename, path, append='', overwrite=True):
    if not op.exists(path):
        os.makedirs(path)
        print('creating dir: ' + path) 
    ff = op.join(path, filename + append + '.fif')
    #print("Saving %s ..." % ff)
    data.save(fname=ff, overwrite=overwrite)



def print_msg(msg):
    n_line_marks = min([len(msg)+20, 100])
    print('\n' + n_line_marks*'#' + '\n' + msg + '\n' + n_line_marks*'#' + '\n')

    
def plot_scaled(data, picks = None):
    data.plot(scalings = dict(eeg=10e-5), 
              n_channels = len(data.ch_names), 
              picks = picks)

# This is specific for this very experiment (vMemEcc)!
def get_event_dict(event_ids): 
    targ_evs = [i for i in event_ids]
    epo_keys = ['CueL', 'CueR', 'LoadLow', 'LoadHigh', 'EccS', 'EccM', 'EccL']

    event_dict = {key: [] for key in epo_keys}
    for ev in targ_evs:
        ev_int = int(ev[-3:]) 
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
        
    return event_dict
