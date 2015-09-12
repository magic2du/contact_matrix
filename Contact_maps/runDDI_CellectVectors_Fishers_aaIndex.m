function runDDI_CellectVectors_Fishers_aaIndex_Sep12(ddiName, outputPath)
% some each datamatrix in the output Path.
%}
%{
ddiName = 'PF00385.16_int_PF00385.16';
FisherMode = 'FisherM1';
SVMMode = 'SVMLIGHT';
Kernel = 'POLY';
%}
folder3did = ['/home/du/Protein_Protein_Interaction_Project/' ...
                                        '3did_20NOV2009/dom_dom_ints/'];
ddiPath = [folder3did ddiName '/'];
folderResults = ...
        ['/home/du/Protein_Protein_Interaction_Project/Contact_Matrix_Project/dom_dom_ints/' ddiName '/'];
if ~exist(folderResults, 'dir')
	mkdir(folderResults);
end

% run only one example per cluster.
percThr = 90;
load([ddiPath 'MC' num2str(percThr) '.mat']);
pairsToRun = [];
for clCtr = 1:size(MC, 2)
    pairsInCl = find(MC(:, clCtr) == 1);
    % don't run pairs that were run in other clusters:
    pairsInCl = setdiff(pairsInCl, pairsToRun);
    %randIdx = randi(length(pairsInCl), 1);
    %pairsToRun(end+1) = pairsInCl(randIdx);
    pairsToRun(end+1) = pairsInCl(1);
end
% Save the pairs to run as a file
save([outputPath 'pairsToRun.txt'], 'pairsToRun', '-ascii');
tmpTraining = [];
tmpValidation=[];

% last one as validation, other as training.
for pairCtr = 1:length(pairsToRun)
    
    testPair = pairsToRun(pairCtr);
    testPair

    
    %auc = runSeqPairAllVectorsChooseNegRand_02NOV2011( ...
    %                    ddiName, testPair, FisherMode, SVMMode, Kernel);
    TrainingV = runSeqPair_CellectVectors_Fishers_aaIndex( ...
        ddiName, testPair);
    size(TrainingV)
    tmpTraining=[tmpTraining; TrainingV];
end
%last valication pair
%ValidationV = runSeqPair_CellectVectors_Fishers_aaIndex( ...
    %    ddiName, pairsToRun(end));
%tmpValidation=[tmpValidation; ValidationV];

return;
