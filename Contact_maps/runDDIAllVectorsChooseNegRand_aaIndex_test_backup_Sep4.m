function meanAUC = runDDIAllVectorsChooseNegRand_aaIndex_test_backup_Sep4 ...
                                (ddiName, FisherMode, SVMMode, Kernel)
%function meanAUC = runDDIAllVectorsChooseNegRand_02NOV2011()
%{
ddiName = 'PF00385.16_int_PF00385.16';
FisherMode = 'FisherM0'; % or 'FisherM1', 'constFisherM0', 'constFisherM1'
SVMMode = 'SVMLIGHT'; % or 'MATLAB'
Kernel = 'RBF'; % or 'POLY'
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

AUC_Array = [];
startTime = clock;
for pairCtr = 1:length(pairsToRun)
    
    testPair = pairsToRun(pairCtr);
    
    % if this is taking too long, go on! (limit is 2 hours).
    eTime = etime(clock, startTime);
    if eTime > 8*60*60
        error('This DDI was taking too long.');
    end
    
    %auc = runSeqPairAllVectorsChooseNegRand_02NOV2011( ...
    %                    ddiName, testPair, FisherMode, SVMMode, Kernel);
    auc = runSeqPairAllVectorsChooseNegRand_aaIndex_test_backup_Sep4( ...
        ddiName, testPair, FisherMode, SVMMode, Kernel);
    AUC_Array(end+1, :) = auc;
end
meanAUC = mean(AUC_Array);

%fprintf(['\n\n\nFinished ' ddiName ', ...
 %                       meanAUC = ' num2str(meanAUC, '%0.3f') '.\n\n\n']);

% print results to file.
%summaryFile = [folderResults 'AllVectorsChooseNegRand_02NOV2011_' ...
%                            FisherMode '_' SVMMode '_' Kernel '.summary'];
%save(summaryFile, 'AUC_Array', '-ascii');

return;
