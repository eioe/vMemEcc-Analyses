%% Prepare P04:


path_data = 'Data\PilotData\';
path_save_sets = '\Data\\PilotData\prepared\';
if ~exist(path_save_sets, 'dir'); mkdir(path_save_sets); end
subject = 'P05';
path_data_eeg = [path_data '\' subject '\EEG\'];

[file, nfiles] = FileFromFolder(path_data_eeg, [], 'vhdr');
if (nfiles > 1) 
    warning(['more than 1 EEG file for this subject! Using first one: ' ... 
        file(1).name]);
    file = file(1);
end

[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
EEG = pop_loadbv(path_data_eeg, ...
    file.name, ...
    [], ...
    []);

% remove crap channel:
EEG = pop_select( EEG,'nochannel',{'EOGv'});
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'gui','off'); 

eeglab redraw;

% Cropping the data to (exp) blocks 4-6 as subjected missunderstood task 
% in the earlier blocks:
mrks = {EEG.event.type};

% find onset of B4:
mrkr_B4_on = find(strcmp(mrks, 'S216'));
time_B4_on = EEG.event(mrkr_B4_on).latency  * 1/EEG.srate;

% offset B6:
mrkr_B6_off = find(strcmp(mrks, 'S219'));
time_B6_off = (EEG.event(mrkr_B6_off).latency) * 1/EEG.srate;

% crop:
EEG = pop_select( EEG, 'time',[time_B4_on time_B6_off] );
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 2, ...
    'setname',[EEG.setname '_B4-B6'],'gui','off'); 