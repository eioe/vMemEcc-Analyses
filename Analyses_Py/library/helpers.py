
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


    

def save_data(data, filename, path, append=''):
    if not op.exists(path):
        os.makedirs(path)
        print('creating dir: ' + path) 
    ff = op.join(path, filename + append + '.fif')
    #print("Saving %s ..." % ff)
    data.save(fname=ff, overwrite=True)