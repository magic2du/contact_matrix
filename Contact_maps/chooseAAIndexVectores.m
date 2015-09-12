function [selectedData, label]=chooseAAIndexVectores(dataFile, FisherMode)
% datafile format F0_20_F1_20_Sliding_17_11_F0_20_F1_20_Sliding_17_11_ouput
data=load(dataFile);
if strcmpi(FisherMode, 'None')
    selectedData=[];
    selectedData=data(:, 41:227); %41-(40+17*11)
    selectedData=[selectedData, data(:, 268:454)]; %(40+17*11)+1+40: (40+17*11)+(40+17*11)
elseif strcmpi(FisherMode, 'AAIndex1')
    selectedData=[];
    selectedData=data(:, 126:142); % AAindex window =1 for resA 40+17*5+1:40+17*5+17
    selectedData=[selectedData, data(:, 353:369)];  % AAindex window =1 for resB (40+17*11)+40+17*5+1: +17
elseif strcmpi(FisherMode, 'AAIndex3')
    selectedData=[];
    selectedData=data(:, 109:159); % AAindex window =3 for resA 40+17*4+1:40+17*4+17*3
    selectedData=[selectedData, data(:, 336:386)];  % AAindex window =3 for resB 40+17*4+1:40+17*4+17*3+1: +17*3

elseif strcmpi(FisherMode, 'FisherM0')
    selectedData=[];
    selectedData=data(:, 1:20); %F0
    selectedData=[selectedData, data(:, 41:227)]; %1+40: (40+17*11)

    selectedData=[selectedData, data(:, 228:247)]; % F0 (40+17*11)+1: (40+17*11)+20
    selectedData=[selectedData, data(:, 268:454)]; %(40+17*11)+1+40: (40+17*11)+(40+17*11)
elseif strcmpi(FisherMode, 'FisherM1')  
    selectedData=[];
    selectedData=data(:, 21:40); %F1
    selectedData=[selectedData, data(:, 41:227)]; %1+40: (40+17*11)

    selectedData=[selectedData, data(:, 248:267)]; % F1 (40+17*11)+20+1: (40+17*11)+40
    selectedData=[selectedData, data(:, 268:454)]; %(40+17*11)+1+40: (40+17*11)+(40+17*11)
elseif strcmpi(FisherMode, 'FisherM1AA1')  
    selectedData=[];
    selectedData=data(:, 21:40); %F1
    selectedData=[selectedData, data(:, 126:142)]; % AAindex window =1 for resA 40+17*5+1:40+17*5+17

    selectedData=[selectedData, data(:, 248:267)]; % F1 (40+17*11)+20+1: (40+17*11)+40
    selectedData=[selectedData, data(:, 353:369)];  % AAindex window =1 for resB (40+17*11)+40+17*5+1: +17
elseif strcmpi(FisherMode, 'FisherM1AA3')  
    selectedData=[];
    selectedData=data(:, 21:40); %F1
    selectedData=[selectedData, data(:, 109:159)]; % AAindex window =3 for resA 40+17*4+1:40+17*4+17*3

    selectedData=[selectedData, data(:, 248:267)]; % F1 (40+17*11)+20+1: (40+17*11)+40
    selectedData=[selectedData, data(:, 336:386)];  % AAindex window =3 for resB 40+17*4+1:40+17*4+17*3+1: +17*3

elseif strcmpi(FisherMode, 'All')
    selectedData=data(:, 1:end-1);
elseif strcmpi(FisherMode, 'FisherM0ONLY')
    selectedData=[];
    selectedData=data(:, 1:20); %F0
    selectedData=[selectedData, data(:, 228:247)]; % F0 (40+17*11)+1: (40+17*11)+20
elseif strcmpi(FisherMode, 'FisherM1ONLY')
    selectedData=[];
    selectedData=data(:, 21:40); %F1
    selectedData=[selectedData, data(:, 248:267)]; % F1 (40+17*11)+20+1: (40+17*11)+40

end
label=data(:, end);
return;

