import os

dir_data = os.path.join('S:', 'Meine Bibliotheken', 'Experiments', 'vMemEcc', 'Data', 'SubjectData' )

dirs = [dd for dd in os.listdir(dir_data) if 'VME' in dd]

for dir in dirs:
    dir_s = os.path.join(dir_data, dir)
    s_dict = dict()
    for s_type in ['EEG', 'Unity', 'EyeTracking']:
        dir_s_type = os.path.join(dir_s, s_type)
        if len(os.listdir(dir_s_type)) == 0:
            print(f'Problem with {dir}: no files in {s_type}')

