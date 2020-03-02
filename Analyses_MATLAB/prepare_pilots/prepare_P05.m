%% prepare_P05
%
% customized preparing script for pilot P05.
%
% Relevant notes: 
% EOG channels: 
%  HEOG: [5 27] % {FT9, FT10}
%  VEOG: [33 61 1 32] % {AF7, AF8, Fp1, Fp2}

% no triggers for first part of B1 (no trigger cable)
% code crashed after RS measurement:
% Files: 
%   P05.eeg:   B1-B3 + RS
%   P05.3.eeg: B4-B6



%% Set paths and variables:
subject = 'P05';

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
        'setname',[subject '.' num2str(i)], ...
        'gui','off'); 
end

% Merge single files:
EEG = pop_mergeset( ALLEEG, [2  1], 0);
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 2, ...
    'setname',subject, ...
    'gui','off');

% remove crap channel:
EEG = pop_select( EEG,'nochannel',{'EOGv'});

%% Save files:

% Save full file:
EEG = pop_saveset( EEG, ...
    'filename',[EEG.setname '.set'], ...
    'filepath', path_save_sets);

% Crop out RS data and save:
% Eyes open:
EEG = pop_rmdat( EEG, {'S224'},[0 330] ,0);
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 4, ...
    'setname',[subject '_rsOpen'], ...
    'gui','off');
EEG = pop_saveset( EEG, ...
    'filename',[EEG.setname '.set'], ...
    'filepath', path_save_RS);

% Eyes closed:
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 5, ...
    'retrieve',4,'study',0);  % back to full SET
EEG = pop_rmdat( EEG, {'S226'},[0 330] ,0);
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 4, ...
    'setname',[subject '_rsClosed'], ...
    'gui','off');
EEG = pop_saveset( EEG, ...
    'filename',[EEG.setname '.set'], ...
    'filepath', path_save_RS);



eeglab redraw;
    