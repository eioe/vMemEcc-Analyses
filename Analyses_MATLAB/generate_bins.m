%% Generate bin lists:

% Markers are labelled according to this logic:
% Trigger t
% t ? [150 ... 197]	

% If 
%   t – 150
%        % 2 = 0      >>> CueDir ==  -1
%        % 2 = 1      >>> CueDir ==  1
%        % 4 < 2      >>> Change ==  true
%        % 4 > 1      >>> Change ==  false
%        % 8 < 4      >>> StimN == 2
%        % 8 > 3      >>> StimN == 4
%        % 16 < 8     >>> cortMag ==  0
%        % 16 > 7     >>> cortMag ==  cM style
%        % 48 < 16    >>> Ecc ==  4°
%        % 48 > 31    >>> Ecc ==  15° / 14°
%        % 48 > 15 &&
%        % 48 < 32    >>> Ecc ==  9.5° / 9°



%% bin with high load:
bin_loadHigh = [];
bin_loadLow = [];
for i=150:197
    s = string(['S ' num2str(i)]);
    if mod(i-150,8)>3
        bin_loadHigh = [bin_loadHigh s];
    else
        bin_loadLow = [bin_loadLow s];
    end
end
        