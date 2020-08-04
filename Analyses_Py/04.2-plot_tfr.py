
import os
import os.path as op
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import mne
from pathlib import Path
from library import helpers, config


def get_tfr(epos, picks='all', average=True):
    freqs = np.concatenate([np.arange(6, 26, 1)])#, np.arange(16,30,2)])
    n_cycles = freqs / 2.  # different number of cycle per frequency
    power = mne.time_frequency.tfr_morlet(epos, picks=picks, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                          return_itc=False, average=average, decim=1, n_jobs=-2)
    return power

