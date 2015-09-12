function     [aucBL_Array, blSens_Array, blSpec_Array, ...
                    auc_Array, svmSens_Array, svmSpec_Array, test_f_measure_Array,...
                    trainAccuracy_Array, trainRecall_Array, trainPrecision_Array, train_f_measure_Array]  = runDDI_Seq_Par_ChooseNegRand_aaIndex_test_load_DL_remote...
                                (ddiName, FisherMode, SVMMode, Kernel, choseNegRatio, choseNegRatioTest)

folder3did = ['/big/du/Protein_Protein_Interaction_Project/Contact_Matrix_Project/Vectors_Fishers_aaIndex_raw/'];
                                        
ddiPath = [folder3did ddiName '/'];
folderResults = ddiPath;
if ~exist(folderResults, 'dir')
	mkdir(folderResults);
end

% load sequence numbers  to run.
filePath=[ddiPath 'pairsToRun.txt'];
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
parfor pairCtr = 1:length(pairsToRun)
    
    testPair = pairsToRun(pairCtr);
    
    % if this is taking too long, go on! (limit is 8 hours).
    if 0 
        eTime = etime(clock, startTime);
        if eTime > 8*60*60
            error('This DDI was taking too long.');
        end
    end

    [aucBL, blSens, blSpec, ...
                    auc, svmSens, svmSpec, test_f_measure,...
                    trainAccuracy, trainRecall, trainPrecision, train_f_measure] = runSeqPairAllVectorsChooseNegRand_aaIndex_test_load_DL_remote ...
                        (ddiName, testPair, FisherMode, SVMMode, Kernel, choseNegRatio, choseNegRatioTest);

    aucBL_Array = [aucBL_Array; aucBL];
    blSens_Array = [blSens_Array; blSens];
    blSpec_Array = [blSpec_Array; blSpec];
    auc_Array = [auc_Array; auc];
    svmSens_Array = [svmSens_Array; svmSens];
    svmSpec_Array = [svmSpec_Array; svmSpec];
    test_f_measure_Array = [test_f_measure_Array; test_f_measure];
    trainAccuracy_Array = [trainAccuracy_Array; trainAccuracy];
    trainRecall_Array = [trainRecall_Array; trainRecall];
    trainPrecision_Array = [trainPrecision_Array; trainPrecision];
    train_f_measure_Array = [train_f_measure_Array; train_f_measure];
end

%fprintf(['\n\n\nFinished ' ddiName ', ...
 %                       meanAUC = ' num2str(meanAUC, '%0.3f') '.\n\n\n']);

% print results to file.
%summaryFile = [folderResults 'AllVectorsChooseNegRand_02NOV2011_' ...
%                            FisherMode '_' SVMMode '_' Kernel '.summary'];
%save(summaryFile, 'AUC_Array', '-ascii');

return;