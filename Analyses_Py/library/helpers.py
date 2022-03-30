
from logging import warning
import os
import os.path as op
# from pathlib import Path
import json

import mne
from library import config


def load_data(filename, path, append, verbose=None):
    if append == '-raw':
        ff = op.join(path, filename + append + '.fif')
        return mne.io.Raw(ff, verbose=verbose)
    elif append == '-epo':
        ff = op.join(path, filename + append + '.fif')
        return mne.read_epochs(ff, verbose=verbose)
    else:
        print('This append (%s) is not yet implemented.' % append)


def save_data(data, filename, path, append='', overwrite=True, verbose=None):
    if not op.exists(path):
        os.makedirs(path)
        print('creating dir: ' + path)
    if 'tfr' in append:
        fmt = '.h5'
    else:
        fmt = '.fif'
    ff = op.join(path, filename + append + fmt)
    # print("Saving %s ..." % ff)
    data.save(fname=ff, overwrite=overwrite)


def chkmk_dir(path):
    if not op.exists(path):
        os.makedirs(path)
        print('creating dir: ' + path)


def print_msg(msg):
    n_line_marks = min([len(msg)+20, 100])
    print('\n' + n_line_marks*'#' + '\n' + msg + '\n' + n_line_marks*'#' +
          '\n')


def plot_scaled(data, picks=None):
    data.plot(scalings=dict(eeg=10e-5),
              n_channels=len(data.ch_names),
              picks=picks)


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


def make_epos_dicts(epos, target_dicts, event_dict):
    # Modifies (non-empty) target_dicts in place!
    if isinstance(epos, mne.BaseEpochs):
        epos = [epos]
    if isinstance(target_dicts, dict):
        target_dicts = [target_dicts]
    if len(epos) != len(target_dicts):
        raise ValueError("Epos and target_dicts must be of same length")
    for d in target_dicts:
        if len(d) > 0:
            raise ValueError("You handed in a non-empty target_dict.")
        d = dict()
    # epos_CDA_dict = dict()
    # epos_collapsed_dict = dict()
    for _dict, epos in zip(target_dicts, epos):
        _dict['All'] = epos
        for load in ['LoadHigh', 'LoadLow']:
            _dict[load] = epos[event_dict[load]]

            for ecc in ['EccS', 'EccM', 'EccL']:
                if ecc not in _dict:
                    _dict[ecc] = epos[event_dict[ecc]]
                _dict[load + ecc] = epos[event_dict[load]][event_dict[ecc]]


def extract_var(var, val, path_ev=config.paths['extracted_vars_file'], overwrite=True,
                rm_leading_zero=False, is_pval=False, exp_format='.2f'):

    if op.exists(path_ev): 
        with open(path_ev) as f:
            exp_vars_dict = json.load(f)
    else:
        exp_vars_dict = dict()

    val_str = format(val, exp_format)

    # TODO: handle pvals according to alpha level. For now only leading zero
    # is removed.
    if is_pval or rm_leading_zero:
        val_str = val_str.lstrip('0')

    if var in exp_vars_dict:
        old_val = exp_vars_dict[var]
        if not (old_val == val_str):
            if overwrite:
                warning(f'Overwriting old value of {var} ({old_val}) ' +
                        f'with: {val_str}')
            else:
                warning(f'There is already a value for {var}: {old_val}.' +
                        '\n' +
                        f'Allow overwritng to extract new value: {val_str}\n' +
                        'Skipping export -- keeping old value.')
                return

    exp_vars_dict[var] = val_str

    with open(path_ev, 'w') as f:
        json.dump(exp_vars_dict, f)
