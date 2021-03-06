function runDDI_CellectVectors_Fishers_aaIndex_topology_Sep25(ddiName, outputPath)
% some each datamatrix in the output Path.
%}
%{
ddiName = 'PF00385.16_int_PF00385.16';
FisherMode = 'FisherM1';
SVMMode = 'SVMLIGHT';
Kernel = 'POLY';
%}
%folder3did = ['/home/du/Protein_Protein_Interaction_Project/' ...
%                                        '3did_20NOV2009/dom_dom_ints/'];
folder3did ='/home/du/Protein_Protein_Interaction_Project/3did_15OCT2010/topologyTest/dom_dom_ints/';
ddiPath = [folder3did ddiName '/'];
folderResults = ...
        ['/home/du/Protein_Protein_Interaction_Project/Contact_Matrix_Project/dom_dom_ints/' ddiName '/'];
if ~exist(folderResults, 'dir')
	mkdir(folderResults);
end

% run only one example per cluster.

ddiStructFile = [folder3did ddiName '/ddi_str_array.mat'];
load(ddiStructFile);
pairsToRun=[1: length(ddi_str_array)];
% Save the pairs to run as a file
save([outputPath 'pairsToRun.txt'], 'pairsToRun', '-ascii');

% 
for pairCtr = 1:length(pairsToRun)
    
    testPair = pairsToRun(pairCtr);
    testPair

    
    %auc = runSeqPairAllVectorsChooseNegRand_02NOV2011( ...
    %                    ddiName, testPair, FisherMode, SVMMode, Kernel);
    [TrainingV, numPos, numNeg] = runSeqPair_CellectVectors_Fishers_aaIndex_topology_Sep25( ...
        ddiName, testPair);
	filename=[outputPath 'F0_20_F1_20_Sliding_17_11_F0_20_F1_20_Sliding_17_11_ouput_' num2str(testPair) '.txt']
	save(filename, 'TrainingV', '-ascii');
	filename=[outputPath 'numPos_' num2str(testPair) '.txt']
	save(filename, 'numPos', '-ascii');
	filename=[outputPath 'numNeg_' num2str(testPair) '.txt']
	save(filename, 'numNeg', '-ascii');
end
%last valication pair
%ValidationV = runSeqPair_CellectVectors_Fishers_aaIndex( ...
    %    ddiName, pairsToRun(end));
%tmpValidation=[tmpValidation; ValidationV];

return;
