function reportResultsAllVectorsChooseNegRand_02NOV2011( ...
                                        FisherMode, SVMMode, Kernel, Date)
%function reportResultsAllVectorsChooseNegRand_02NOV2011()

%{
FisherMode = 'FisherM0'; % or 'FisherM1', 'constFisherM0', 'constFisherM1'
SVMMode = 'SVMLIGHT'; % or 'MATLAB'
Kernel = 'RBF'; % or 'POLY'
Date = '02NOV2011';
%}
%{
FisherMode = 'FisherM1';
SVMMode = 'SVMLIGHT';
Kernel = 'POLY';
Date = '02NOV2011';
%}

resultsFile = ['resultsAllVectorsChooseNegRand_' ...
                        FisherMode '_' SVMMode '_' Kernel '_' Date '.txt'];
%resultsFile = 'debuggind_09NOV2011.txt';
if exist(resultsFile, 'file')
    command = ['rm -f ' resultsFile];
    system(command);
end
fid = fopen(resultsFile, 'w');

%finishedDDIs = ['finishedDDIs_AllVectorsChooseNegRand_' ...
%                        FisherMode '_' SVMMode '_' Kernel '_' Date '.txt'];
finishedDDIs = ...
'finishedDDIs_AllVectorsChooseNegRand_FisherM1_SVMLIGHT_POLY_02NOV2011.txt';
fid2 = fopen(finishedDDIs, 'r');
cell_ddis = textscan(fid2, '%s', 'delimiter', '\n');
cell_ddis = cell_ddis{1};
fclose(fid2);

folder3did = ['/home/du/Protein_Protein_Interaction_Project/' ...
                                        '3did_20NOV2009/dom_dom_ints/'];

for ddi_ctr = 1:length(cell_ddis)
    
ddiName = cell_ddis{ddi_ctr};

ddiPath = [folder3did ddiName '/'];
folderResults = ...
    ['/home/du/Protein_Protein_Interaction_Project/Contact_Matrix_Project/dom_dom_ints/' ddiName '/'];

% load needed data.
ddiStructFile = [ddiPath 'ddi_str_array.mat'];
load(ddiStructFile);
numbPairs = length(ddi_str_array);
clear ddi_str_array;

for pairNbr = 1:numbPairs

try
resultsFile = ...
        [folderResults 'AllVectorsChooseNegRand_02NOV2011_results_' ...
        FisherMode '_' SVMMode '_' Kernel '_pair' num2str(pairNbr) '.mat'];
if exist(resultsFile, 'file')
load(resultsFile);

% 11/09/2011 I realized my way of testing predictions results was wrong in
% that it was forcing sensitivity to be equal to specificity by making the
% number of predicted positve to be the number of known real positives,
% which is cheating. I'll change it so that we predict positive whatever
% lies on the positive side of the hyperplane, negative otherwise.
[posTestI posTestJ] = ind2sub(size(gtContactMatrix), ...
                                            find(gtContactMatrix > 0));
distToHPForRealPos = result(1:length(posTestI));
distToHPForRealNeg = result((length(posTestI)+1):end);
svmPredForRealPos = distToHPForRealPos > 0;
svmPredForRealNeg = distToHPForRealNeg > 0;
svmTP = length(find(svmPredForRealPos == 1));
svmFP = length(find(svmPredForRealNeg == 1));
svmTN = length(find(svmPredForRealNeg == 0));
svmFN = length(find(svmPredForRealPos == 0));
svmSens = svmTP/(svmTP+svmFN);
svmSpec = svmTP/(svmTP+svmFP);
if isnan(svmSpec)
    svmSpec = 0;
end

logline = [ddiName '\t' num2str(pairNbr) ...
            '\t' num2str(auc, '%0.3f') ...
            '\t' num2str(svmSens, '%0.3f') ...
            '\t' num2str(svmSpec, '%0.3f') ...
            '\t' num2str(aucBL, '%0.3f') ...
            '\t' num2str(blSens, '%0.3f') ...
            '\t' num2str(blSpec, '%0.3f') '\n'];

fprintf(fid, logline);
end

catch exc
    c = clock;
    time = [num2str(c(1), '%0.0d') '-' num2str(c(2), '%0.0d') '-' ...
            num2str(c(3), '%0.0d') ' ' num2str(c(4), '%0.0d') ':' ...
            num2str(c(5), '%0.0d')];
    logline = [time ' ERROR: ' ddiName ...
                                    '. Message: ' exc.message '\n'];
    fprintf(logline);
end

end

end

fclose(fid);
                    
return;