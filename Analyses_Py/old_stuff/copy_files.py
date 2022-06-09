import os
from pathlib import Path
import shutil


path_sdrive = os.path.join('S:\\', 'Meine Bibliotheken', 'Experiments', 'vMemEcc')
path_data_in = os.path.join(path_sdrive, 'Data', 'SubjectData')

path_extHDD = os

path_study = Path(os.getcwd()).parents[1]
path_out = os.path.join(path_study, 'Data', 'SubjectData')

in_folds = os.listdir(path_data_in)
out_folds = os.listdir(path_out)
subs = [sub for sub in in_folds if sub not in out_folds]
subs.remove('__PXX')

for sub in subs:
    target = os.path.join(path_out, sub, 'EyeTracking')
    os.mkdir(target)
    foldname = os.path.join(path_data_in, sub, 'EEG')
    files = os.listdir(foldname)
    for ff in files:
        ffpath = os.path.join(foldname, ff)
        targetpath = os.path.join(target, ff)
        shutil.copy(ffpath, targetpath)

