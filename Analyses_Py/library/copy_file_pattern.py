import os
from os import path as op

import shutil

path_origin = op.join('C:\\','Users','Felix','Seafile','Experiments','vMemEcc','Data','DataMNE','EEG','04_epo')
path_target = op.join('E:\\','vMemEcc','SubjectData_extern','DataMNE','EEG','04_epo', 'fulllength')
f_list = os.listdir(path_origin)

for f in f_list: 
    print(f)
    if 'fulllength-epo' in f: 
        shutil.copy(op.join(path_origin, f), path_target)