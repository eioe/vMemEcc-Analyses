%% Preprocess

subject = 'P09';

%% Set paths:
path_data =         '.\Data\PilotData\';
path_data_eeg =     [path_data 'EEG\00_prepared\'];
path_save_sets =    [path_data 'EEG\01_preprocessed\'];
path_save_evlists = [path_save_sets 'evLists\'];
if ~exist(path_save_sets, 'dir'); mkdir(path_save_sets); end
if ~exist(path_save_evlists, 'dir'); mkdir(path_save_evlists); end

%%  Load data:
[ALLEEG EEG CURRENTSET] = eeglab;
EEG = pop_loadset('filename', [subject '.set'], ...
    'filepath', path_data_eeg);
[ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);

    

%% Preprocessing steps:
% high-pass filter at 0.01Hz:
EEG = pop_eegfiltnew(EEG, 'locutoff',0.01,'plotfreqz',1);
EEG.setname = [EEG.setname '_hp0.01'];
[ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);

% low-pass filter at 45Hz:
EEG = pop_eegfiltnew(EEG, 'hicutoff',40,'plotfreqz',1);
EEG.setname = [EEG.setname '_lp40'];
[ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);

% You might want to save to disk here:
EEG = pop_saveset( EEG, 'filename', ...
    [EEG.setname '.set'], ...
    'filepath', path_save_sets);
            
% overwrite onset triggers ('S234') with informative markers:
% (these have been sent either 2 or 3 markers before.)

targ_evs = {'S150'};
for i=151:197
    ev_name = ['S' num2str(i)];
    targ_evs{end+1} = ev_name;
end

targ_evs_num = [150:197];
targ_evs_num = cellstr(num2str(targ_evs_num'))';

idx_o = find(strcmp({EEG.event.type}, 'S234'));
for i=1:length(idx_o)
    if ismember(EEG.event(idx_o(i)-2).type, targ_evs)
        EEG.event(idx_o(i)).type = EEG.event(idx_o(i)-2).type;
        EEG.event(idx_o(i)-2).type = 'S999';
    elseif ismember(EEG.event(idx_o(i)-3).type, targ_evs)
        EEG.event(idx_o(i)).type = EEG.event(idx_o(i)-3).type;
        EEG.event(idx_o(i)-3).type = 'S999';
    end
end
    

% calculate bip. EOG chans:
% EEG = pop_eegchanoperator(EEG, {'ch65 = ch33 - ((ch1+ch32)/2) label VEOG'});  
% EEG = pop_eegchanoperator(EEG, {'ch66 = ch35 - ch61 label HEOG'});  
% remove unip. EOG chans:
% EEG = pop_select( EEG, 'nochannel',{'AF7' 'AFz' 'AF8'});
% [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1, ...
%    'setname',[EEG.setname '_EOG'],'gui','off'); 

% create basic event list (actually only stripping the S from event nums):
EEG  = pop_creabasiceventlist( EEG , ...
    'AlphanumericCleaning', 'on', ...
    'BoundaryNumeric', { -99 }, ...
    'BoundaryString', { 'boundary' }, ...
    'Eventlist', [path_save_evlists subject '_eList.txt']);

% epoch around mem. array onset and remove baseline 
% (last 200ms of Cue interval):
EEG = pop_epoch( EEG, targ_evs, [-0.2 2.2], ...
    'newname', [EEG.setname], ...
    'epochinfo', 'yes'); 
EEG = pop_rmbase( EEG, [-200 0]);
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, ...
    'setname',[EEG.setname '_epo_blrem'],'gui','off'); 

% manually reject epochs:




% make copy for ICA:
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, ...
    'setname',[subject '_copy4ICA'],'gui','off'); 

% high-pass filter at 1Hz:
[EEG, com, b] = pop_eegfiltnew(EEG, 'locutoff',1,'plotfreqz',1);
EEG.setname = [EEG.setname '_hp1'];

% run INFOMAX ICA:
EEG = pop_runica(EEG, ...
    'extended',1, ...
    'interrupt','on');
EEG.setname = [EEG.setname '_ICA'];
[ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG, CURRENTSET);


% You might want to save to disk here:
EEG = pop_saveset( EEG, ...
    'filename', [EEG.setname '.set'], ...
    'filepath', path_save_sets);

% manual version of SASICA based comp rejection:
EEG = SASICA(EEG);
eeglab redraw;
% show component activations:
pop_eegplot( EEG, 0, 1, 1);
% plot further component properties (mainly to see freqs >50Hz):
pop_prop(EEG, 0, 1:size(EEG.icaweights, 1));

% SASICA stores the results in base workspace via assignin.
% [Info from Niko Busch's pipeline:
% https://github.com/nabusch/Elektro-Pipe
EEG = evalin('base','EEG');

% Store relevant ICA info:
tmp_icawinv     = EEG.icawinv;
tmp_icasphere   = EEG.icasphere;
tmp_icaweights  = EEG.icaweights;
tmp_icachansind = EEG.icachansind;
tmp_rejcomps    = find(EEG.reject.gcompreject);
fprintf('\n\n\n\n\n');
fprintf(['Rejected components:\n' num2str(tmp_rejcomps)])
fprintf('\n\n\n\n\n');

% Save:
EEG = pop_saveset(EEG, ... 
    'filename', [EEG.setname  '_rejcomp.set'], ...
    'filepath', path_save_sets);


% Transfer to original data:
EEG = ALLEEG(EPOSET);
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, ...
    'retrieve', EPOSET, 'study',0); 

EEG.icawinv     = tmp_icawinv;
EEG.icasphere   = tmp_icasphere;
EEG.icaweights  = tmp_icaweights;
EEG.icachansind = tmp_icachansind;
tmp_shape       = size(EEG.data);
tmp_data        = reshape(EEG.data, [tmp_shape(1) tmp_shape(2)*tmp_shape(3)]);
EEG.icaact      = EEG.icaweights * EEG.icasphere * tmp_data;
EEG.icaact      = reshape(EEG.icaact, tmp_shape);


% remove the components from full data:
EEG = pop_subcomp(EEG, tmp_rejcomps, 0);
EEG.setname = [EEG.setname '_rejcomp.set'];
[ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);

% Save:
EEG = pop_saveset(EEG, ... 
    'filename', EEG.setname, ...
    'filepath', path_save_sets);

for i=1:length(EEG.event)
    EEG.event(i).type = str2double(erase(EEG.event(i).type, {'S', ' '}));
end



