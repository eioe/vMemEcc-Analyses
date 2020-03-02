%% prepare_PXX
%
% customized preparing script for pilot P06.
%
% Relevant notes: 
% EOG channels: 
%  HEOG: [5 27] % {FT9, FT10}
%  VEOG: [33 61 1 32] % {AF7, AF8, Fp1, Fp2}

% no triggers for resting state: ignore it



%% Set paths and variables:
subject = 'P08';

path_data = '.\Data\PilotData\';
path_data_eeg = [path_data '\' subject '\EEG\'];
path_save_sets = [path_data 'EEG\00_prepared\'];
path_save_RS = [path_data 'EEG\RS\'];
if ~exist(path_save_sets, 'dir'); mkdir(path_save_sets); end
if ~exist(path_save_RS, 'dir'); mkdir(path_save_RS); end



%% Read in data files:
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;

[file, nfiles] = FileFromFolder(path_data_eeg, [], 'vhdr');
for i=1:nfiles
    EEG = pop_loadbv(path_data_eeg, ...
        file(i).name, ...
        [], ...
        []);
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, ...
        1, ... 
        'setname',[subject], ...
        'gui','off'); 
end


% remove crap channel:
EEG = pop_select( EEG,'nochannel',{'EOGv'});

%% Save files:

% Save full file:
EEG = pop_saveset( EEG, ...
    'filename',[EEG.setname '.set'], ...
    'filepath', path_save_sets);

eeglab redraw;
    