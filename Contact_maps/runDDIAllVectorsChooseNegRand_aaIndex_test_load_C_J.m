function     [aucBL_Array, blSens_Array, blSpec_Array, ...
                    auc_Array, svmSens_Array, svmSpec_Array, test_f_measure_Array,...
                    trainAccuracy_Array, trainRecall_Array, trainPrecision_Array, train_f_measure_Array]  = runDDIAllVectorsChooseNegRand_aaIndex_test_load_C_J ...
                                (ddiName, FisherMode, SVMMode, Kernel, choseNegRatio, choseNegRatioTest, C, Jj)

folder3did = ['/home/du/Protein_Protein_Interaction_Project/' ...
                                        '3did_20NOV2009/dom_dom_ints/'];
ddiPath = [folder3did ddiName '/'];
folderResults = ...
        ['/home/du/Protein_Protein_Interaction_Project/Contact_Matrix_Project/dom_dom_ints/' ddiName '/'];
if ~exist(folderResults, 'dir')
	mkdir(folderResults);
end

% load sequence numbers  to run.
filePath=['/big/du/Protein_Protein_Interaction_Project/Contact_Matrix_Project/Vectors_Fishers_aaIndex_raw/' ddiName '/' 'pairsToRun.txt'];
pairsToRun= load(filePath);


aucBL_Array = [];
blSens_Array = [];
blSpec_Array = [];
auc_Array = [];
svmSens_Array = [];
svmSpec_Array = [];
test_f_measure_Array = [];
trainAccuracy_Array = [];
trainRecall_Array = [];
trainPrecision_Array = [];
train_f_measure_Array = [];
startTime = clock;
for pairCtr = 1:length(pairsToRun)
    
    testPair = pairsToRun(pairCtr);
    
    % if this is taking too long, go on! (limit is 8 hours).
    if 0 
        eTime = etime(clock, startTime);
        if eTime > 8*60*60
            error('This DDI was taking too long.');
        end
    end
    %auc = runSeqPairAllVectorsChooseNegRand_02NOV2011( ...
    %                    ddiName, testPair, FisherMode, SVMMode, Kernel);
    [aucBL, blSens, blSpec, ...
                    auc, svmSens, svmSpec, test_f_measure,...
                    trainAccuracy, trainRecall, trainPrecision, train_f_measure] = runSeqPairAllVectorsChooseNegRand_aaIndex_test_load_C_J ...
                        (ddiName, testPair, FisherMode, SVMMode, Kernel, choseNegRatio, choseNegRatioTest, C, Jj);

    aucBL_Array(end+1, :) = aucBL;
    blSens_Array(end+1, :) = blSens;
    blSpec_Array(end+1, :) = blSpec;
    auc_Array (end+1, :)= auc;
    svmSens_Array(end+1, :) = svmSens;
    svmSpec_Array(end+1, :) = svmSpec;
    test_f_measure_Array(end+1, :) = test_f_measure;
    trainAccuracy_Array(end+1, :) = trainAccuracy;
    trainRecall_Array(end+1, :) = trainRecall;
    trainPrecision_Array(end+1, :) = trainPrecision;
    train_f_measure_Array(end+1, :) = train_f_measure;
end

%fprintf(['\n\n\nFinished ' ddiName ', ...
 %                       meanAUC = ' num2str(meanAUC, '%0.3f') '.\n\n\n']);

% print results to file.
%summaryFile = [folderResults 'AllVectorsChooseNegRand_02NOV2011_' ...
%                            FisherMode '_' SVMMode '_' Kernel '.summary'];
%save(summaryFile, 'AUC_Array', '-ascii');

return;
