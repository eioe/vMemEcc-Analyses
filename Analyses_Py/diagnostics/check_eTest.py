import glob
from matplotlib import pyplot as plt
import numpy as plt
import os

import mne

## Analyze a bv file that includes a bunch of triggers (with the EEG cap lying somewhere) 
## to see whether there are trigger-based effects in the eeg channels.

# triggers:
# 150x:
#    -'S 23'
#    -'S 24'  >>> t['S 24'] = t['S 23'] + 200 (ms)

# load eTest data

path_data = os.path.join('..', '..', 'Data')
path_data_folder = os.path.join(path_data, 'DummyData', 'TriggerArt')
fname = [f for f in os.listdir(path_data_folder) if f.endswith('pin.vhdr')][0]
path_data_file = os.path.join(path_data_folder, fname)

raw = mne.io.read_raw_brainvision(path_data_file)


# get events:
events_,events_id = mne.events_from_annotations(raw)

# filter (0.01 - 40 Hz)?


# epoch to 'S 23'
epochedF = mne.Epochs(raw.copy().load_data(), events=events_, event_id=[1], tmin=-0.03, tmax=0.5)
epoched.plot_image()
# average >> evoked & plot 

evoked = epochedF.average()
evoked.plot_topo()

fig = plt.figure()
plt.plot(evoked.times, evoked.data[23,])
[plt.axvline(x=0.05*i, color='black', ls=':') for i in list(range(8))]
plt.axvline(color='black', ls=':')
fig.show()