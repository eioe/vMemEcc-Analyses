%% Preprocess
subject = 'P06';

% settings:
calc_icaact = false;

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
            


targ_evs = {'S150'};
for i=151:197
    ev_name = ['S' num2str(i)];
    targ_evs{end+1} = ev_name;
end
n_targ_evs = sum(ismember({EEG.event.type}, targ_evs));


%% Use this to remap meaningful events to stim onset:
% overwrite onset triggers ('S234') with informative markers:
%(these have been sent either 2 or 3 markers before.)

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


    % epoch around mem. array onset and remove baseline 
    % (last 200ms of Cue interval):
[EEG accep_evs] = pop_epoch( EEG, targ_evs, [-0.2 2.2], ...
    'newname', [EEG.setname], ...
    'epochinfo', 'yes'); 
EEG = pop_rmbase( EEG, [-200 0]);
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, ...
    'setname',[EEG.setname '_epo_blrem'],'gui','off'); 

% create fields to store info about all rejected epochs:
EEG.etc.reject.eporemain = 1:n_targ_evs;
EEG.etc.reject.epotot = [];

% store indices of epos rejected due to boundary events:
if (length(accep_evs) < n_targ_evs)
    rej_epo_rel = find(~ismember(1:n_targ_evs, accep_evs));
    rej_epo_abs = EEG.etc.reject.eporemain(rej_epo_rel);
    EEG.etc.reject.epoboundary = rej_epo_abs;
    EEG.etc.reject.epotot = [EEG.etc.reject.epotot rej_epo_abs];
    EEG.etc.reject.eporemain = setdiff(EEG.etc.reject.eporemain, rej_epo_abs);
end

% manually reject epochs:
EEG = eeg_checkset( EEG );
pop_eegplot( EEG, 1, 1, 0);
  %%%%%%%%%%% manual marking %%%%%%%%%%%%%%
rej_epo_rel = find(EEG.reject.rejmanual);
EEG = pop_rejepoch( EEG, rej_epo_rel, 1);
% Update rejection fields:
rej_epo_abs = EEG.etc.reject.eporemain(rej_epo_rel);
EEG.etc.reject.epomanual = rej_epo_abs;
EEG.etc.reject.epotot = [EEG.etc.reject.epotot rej_epo_abs];
EEG.etc.reject.eporemain = setdiff(EEG.etc.reject.eporemain, rej_epo_abs);

EEG.setname = [EEG.setname '_rejepo'];
[ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG, CURRENTSET);
idx_eposet = CURRENTSET;

% You might want to save to disk here:
EEG = pop_saveset( EEG, 'filename', ...
    [EEG.setname '.set'], ...
    'filepath', path_save_sets);


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
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, ...
    'retrieve', idx_eposet, 'study',0); 

EEG.icawinv     = tmp_icawinv;
EEG.icasphere   = tmp_icasphere;
EEG.icaweights  = tmp_icaweights;
EEG.icachansind = tmp_icachansind;

if (calc_icaact)
    tmp_shape       = size(EEG.data);
    tmp_data        = reshape(EEG.data, [tmp_shape(1) tmp_shape(2)*tmp_shape(3)]);
    EEG.icaact      = (EEG.icaweights * EEG.icasphere) * tmp_data;
    EEG.icaact      = reshape(EEG.icaact, tmp_shape);
end

% remove the components from full data:
tmp_setname = EEG.setname;
EEG = pop_subcomp(EEG, tmp_rejcomps, 0);
% overwrite weird eeglab naming:
EEG.setname = [tmp_setname '_rejcomp.set'];
[ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);

% change event markers to integers:
for i=1:length(EEG.event)
    EEG.event(i).type = str2double(erase(EEG.event(i).type, {'S', ' '}));
end

% remove all but informative markers:
for i=1:length(EEG.event)
    EEG.event(i).type = str2double(erase(EEG.event(i).type, {'S', ' '}));
end
idx_good = ismember([EEG.event.type], 150:199);
EEG.event = EEG.event(idx_good);

% Save:
EEG = pop_saveset(EEG, ... 
    'filename', EEG.setname, ...
    'filepath', path_save_sets);





