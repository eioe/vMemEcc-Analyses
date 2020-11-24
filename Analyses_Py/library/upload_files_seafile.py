import os
from os import path as op
import numpy as np
from subprocess import call

path_sdrive = op.join('S:\\', 'Meine Bibliotheken', 'Experiments', 'vMemEcc')
path_data_sdrive = op.join(path_sdrive, 'Data', 'SubjectData')

path_data_hdd = op.join('D:\\', 'vMemEcc', 'SubjectData_extern', 'SubjectData')

subs = [fold for fold in os.listdir(path_data_hdd) if 'VME_' in fold]

for sub in subs:
    for blocknum in np.arange(4, 14):
        target = op.join(path_data_sdrive, sub, 'EyeTracking', 'Block'+str(blocknum), '000', 'exports')
        if op.exists(target): 
            continue
        source = op.join(path_data_hdd, sub, 'EyeTracking', 'Block'+str(blocknum), '000', 'exports')
        call(["robocopy",source, target, "/S"])
