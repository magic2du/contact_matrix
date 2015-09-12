function     [aucBL_Array, blSens_Array, blSpec_Array, ...
                    auc_Array, svmSens_Array, svmSpec_Array, test_f_measure_Array,...
                    trainAccuracy_Array, trainRecall_Array, trainPrecision_Array, train_f_measure_Array]  = runDDI_AllVectorsChooseNegRand_aaIndex_load_All_cross ...
                                (ddiName, trainPO, trainNE, POTestWhole, NETestWhole, FisherMode, SVMMode, Kernel, choseNegRatio, choseNegRatioTest)

folder3did = ['/home/michael/Documents/Contact_Matrix_Project/Vectors_Fishers_aaIndex_raw/'];
                                        
ddiPath = [folder3did ddiName '/'];
folderResults = ddiPath;
dataFolder=ddiPath;

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

trainGroundTruth = [ones(size(trainPO, 1), 1); ...
                -1*ones(size(trainNE, 1), 1)];
POTest=[];
NETest=[];
blSens = 0;
blSpec = 0;
aucBL = 0;
%%%%%%%%%%%%%get the testing examples To POTest NETest.
filePath=[ddiPath 'pairsToRun.txt'];
pairsToRun= load(filePath);
for idxCtr = 1:length(pairsToRun)
    seqCtr = pairsToRun(idxCtr);

    % create positive and negative training sets.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    dataFile=[dataFolder 'F0_20_F1_20_Sliding_17_11_F0_20_F1_20_Sliding_17_11_ouput_' num2str(seqCtr) '.txt'];
    numPos=load([dataFolder 'numPos_' num2str(seqCtr) '.txt']);
    numNeg=load([dataFolder 'numNeg_' num2str(seqCtr) '.txt']);
    [selectedData, label]=chooseAAIndexVectores(dataFile, FisherMode);
    PO=selectedData(1:numPos, :);
    NE=selectedData(numPos+1: numPos+numNeg, :);

    % chose negatives as in the the ratio default is 1.
    selectedNE=[];
    if choseNegRatio==0
        selectedNE=NE;
    else
        numbNegTrain = size(PO, 1) * choseNegRatioTest;
        if numbNegTrain>numNeg
            numbNegTrain=numNeg;
        end
        r=randperm(size(NE, 1));
        r=r(1:numbNegTrain);
        indTrainNE=sort(r);
        selectedNE = NE(indTrainNE, :);
    end
    POTest{seqCtr}=PO;
    NETest{seqCtr}=selectedNE;
end

%%%%%%%%%%%%%% Traing and Testing.
if strcmp(SVMMode, 'SVMLIGHT')
    % SVMLIGHT.
    % test on training set.
    % SVMLIGHT.
    svmlightFolder = ...
    '/home/du/Protein_Protein_Interaction_Project/3did_20NOV2009/svm_light/';
    folderResults = ...
        ['/home/du/Protein_Protein_Interaction_Project/Contact_Matrix_Project/dom_dom_ints/' ddiName '/'];
    trainFile = ...
        [folderResults FisherMode '_'  SVMMode '_' Kernel '_' num2str(choseNegRatio) '_' num2str(choseNegRatioTest) '_remote_ChooseNegRand_aaIndex_test_load_DL_cross.train'];
    modelFile = ...
        [folderResults FisherMode '_'  SVMMode '_' Kernel '_' num2str(choseNegRatio) '_' num2str(choseNegRatioTest) '_remote_ChooseNegRand_aaIndex_test_load_DL_cross.model'];
    % write_SVM_file(FamilyPO, FamilyNE, trainFile);
    write_SVM_file(trainPO, trainNE, trainFile);
    if strcmp(Kernel, 'RBF')
        command = ...
            [svmlightFolder 'svm_learn -t 2 -g 1 -c ' num2str(C) ' -j ' num2str(Jj) ' '  trainFile ' ' modelFile];
    elseif strcmp(Kernel, 'POLY')
        command = ...
            [svmlightFolder 'svm_learn -t 1 -d 3 -c ' num2str(C) ' -j ' num2str(Jj) ' '  trainFile ' ' modelFile];
    elseif strcmp(Kernel, 'LINEAR')
        command = ...
            [svmlightFolder 'svm_learn -t 0 ' trainFile ' ' modelFile];
    else
        error('ERROR');
    end
    system(command);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%testing%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     % SVMLIGHT.
    % test on training set.
    testFile = ...
        [folderResults FisherMode '_'  SVMMode '_' Kernel '_' num2str(choseNegRatio) '_' num2str(choseNegRatioTest) '_remote_ChooseNegRand_aaIndex_test_load_DL_cross.testOnTrain'];
    resultFile = [folderResults FisherMode '_'  SVMMode '_' Kernel '_' num2str(choseNegRatio) '_' num2str(choseNegRatioTest) '_remote_ChooseNegRand_aaIndex_test_load_DL_cross.resultOnTrain'];
    write_SVM_file(trainPO, trainNE, testFile);
    command = ...
    [svmlightFolder 'svm_classify ' testFile ' ' modelFile ' ' resultFile];
    system(command);
    trainResult = load(resultFile);
    trainAUC = roc(trainResult, trainGroundTruth, 'nofigure');
    % for training performance.
    trainResult0_1=zeros(length(trainResult), 1);
    trainResult0_1(find(trainResult>=0))=1;
    trainGroundTruth(find(trainGroundTruth==-1))=0; % change the format to 0 and 1;
    [accuracy, sensitivity, specificity, precision, recall, f_measure, gmean]=Evaluate(trainGroundTruth, trainResult0_1); %Evaluate(ACTUAL,PREDICTED)
     
    if isnan(precision)
        precision = 0;
    end
    if isnan(f_measure)
        f_measure = 0;
    end
    fprintf('Training AUC: %0.3f%%\n', trainAUC * 100);
    fprintf('Training accuracy: %0.3f%%\n', accuracy * 100);
    fprintf('Training recall: %0.3f%%\n', recall * 100);
    fprintf('Training precision: %0.3f%%\n', precision * 100);
    fprintf('Training f_measure: %0.3f%%\n', f_measure * 100);
    trainAccuracy=accuracy;
    trainRecall=recall;
    trainPrecision=precision;
    train_f_measure=f_measure;
    %
        trainAccuracy_Array(end+1, :) = trainAccuracy;
        trainRecall_Array(end+1, :) = trainRecall;
        trainPrecision_Array(end+1, :) = trainPrecision;
        train_f_measure_Array(end+1, :) = train_f_measure;  
    % test on test set:
    for idxCtr = 1:length(pairsToRun)
        seqCtr = pairsToRun(idxCtr);
        testFile = ...
            [folderResults FisherMode '_'  SVMMode '_' Kernel '_' num2str(choseNegRatio) '_' num2str(choseNegRatioTest) '_remote_ChooseNegRand_aaIndex_test_load_DL_cross.test'];

        resultFile = [folderResults FisherMode '_'  SVMMode '_' Kernel '_' num2str(choseNegRatio) '_' num2str(choseNegRatioTest) '_remote_ChooseNegRand_aaIndex_test_load_DL_cross.result'];

        write_SVM_file(POTest{seqCtr}, NETest{seqCtr}, testFile);
        command = ...
        [svmlightFolder 'svm_classify ' testFile ' ' modelFile ' ' resultFile];
        system(command);
        testResult = load(resultFile);
        testGroundTruth = [ones(size(POTest{seqCtr}, 1), 1); ...
                -1*ones(size(NETest{seqCtr}, 1), 1)];
        auc = roc(testResult, testGroundTruth, 'nofigure');
        % for testing set performance.
        testResult0_1=zeros(length(testResult),1);
        testResult0_1(find(testResult>=0))=1;

        testGroundTruth(find(testGroundTruth==-1))=0; % change the format to 0 and 1;
        [accuracy, sensitivity, specificity, precision, recall, f_measure, gmean]=Evaluate(testGroundTruth, testResult0_1); %Evaluate(ACTUAL,PREDICTED)
        if isnan(precision)
            precision = 0;
        end
        if isnan(f_measure)
            f_measure = 0;
        end
        fprintf([ddiName ' seqPair ' num2str(seqCtr) '\n']);
        fprintf('Testing AUC: %0.3f%%\n', auc * 100);
        fprintf('Testing accuracy: %0.3f%%\n', accuracy * 100);
        fprintf('Testing recall: %0.3f%%\n', recall * 100);
        fprintf('Testing precision: %0.3f%%\n', precision * 100);
        fprintf('Testing f_measure: %0.3f%%\n', f_measure * 100);
        svmSens=recall;
        svmSpec=precision;
        test_f_measure=f_measure;
        % collecting result
        aucBL_Array(end+1, :) = aucBL;
        blSens_Array(end+1, :) = blSens;
        blSpec_Array(end+1, :) = blSpec;
        auc_Array (end+1, :)= auc;
        svmSens_Array(end+1, :) = svmSens;
        svmSpec_Array(end+1, :) = svmSpec;
        test_f_measure_Array(end+1, :) = test_f_measure;
      end
elseif strcmp(SVMMode, 'DL')

    trainGroundTruth = [ones(size(trainPO, 1), 1); ...
				    -1*ones(size(trainNE, 1), 1)];
    testGroundTruth = [ones(size(POTest, 1), 1); ...
				    -1*ones(size(NETest, 1), 1)];

    trainData=[trainPO; trainNE];
    trainLabel=[ones(size(trainPO, 1), 1); ...
				    zeros(size(trainNE, 1), 1)];

    % run the DL algorithms
    [trainResult, testResult]= SAE_AllVectorsChooseNegRand_aaIndex_load_All_cross(trainData, trainLabel, POTestWhole, NETestWhole, POTest, NETest, pairsToRun);
        
    
    trainAUC = roc(trainResult, trainGroundTruth, 'nofigure');
    
    % for training performance.
    trainResult0_1=zeros(length(trainResult), 1);
    trainResult0_1(find(trainResult>=0))=1;
    trainGroundTruth(find(trainGroundTruth==-1))=0; % change the format to 0 and 1;
    [accuracy, sensitivity, specificity, precision, recall, f_measure, gmean]=Evaluate(trainGroundTruth, trainResult0_1); %Evaluate(ACTUAL,PREDICTED)
     
    if isnan(precision)
        precision = 0;
    end
    if isnan(f_measure)
        f_measure = 0;
    end
    fprintf('Training AUC: %0.3f%%\n', trainAUC * 100);
    fprintf('Training accuracy: %0.3f%%\n', accuracy * 100);
    fprintf('Training recall: %0.3f%%\n', recall * 100);
    fprintf('Training precision: %0.3f%%\n', precision * 100);
    fprintf('Training f_measure: %0.3f%%\n', f_measure * 100);
    trainAccuracy=accuracy;
    trainRecall=recall;
    trainPrecision=precision;
    train_f_measure=f_measure;
    % collecting 
        trainAccuracy_Array(end+1, :) = trainAccuracy;
        trainRecall_Array(end+1, :) = trainRecall;
        trainPrecision_Array(end+1, :) = trainPrecision;
        train_f_measure_Array(end+1, :) = train_f_measure;  
    % for testing set performance.
    for idxCtr = 1:length(pairsToRun)
        seqCtr = pairsToRun(idxCtr);
        testGroundTruth = [ones(size(POTest{seqCtr}, 1), 1); ...
			    -1*ones(size(NETest{seqCtr}, 1), 1)];
        auc = roc(testResult{seqCtr}, testGroundTruth, 'nofigure');
        testResult0_1=zeros(length(testResult{seqCtr}),1);
        testResult0_1(find(testResult{seqCtr}>=0))=1;
        testGroundTruth(find(testGroundTruth==-1))=0; % change the format to 0 and 1;
        [accuracy, sensitivity, specificity, precision, recall, f_measure, gmean]=Evaluate(testGroundTruth, testResult0_1); %Evaluate(ACTUAL,PREDICTED)
        if isnan(precision)
            precision = 0;
        end
        if isnan(f_measure)
            f_measure = 0;
        end
        fprintf([ddiName ' seqPair ' num2str(seqCtr) '\n']);
        fprintf('Testing AUC: %0.3f%%\n', auc * 100);
        fprintf('Testing accuracy: %0.3f%%\n', accuracy * 100);
        fprintf('Testing recall: %0.3f%%\n', recall * 100);
        fprintf('Testing precision: %0.3f%%\n', precision * 100);
        fprintf('Testing f_measure: %0.3f%%\n', f_measure * 100);
        svmSens=recall;
        svmSpec=precision;
        test_f_measure=f_measure; 
        %Collecting result
        aucBL_Array(end+1, :) = aucBL;
        blSens_Array(end+1, :) = blSens;
        blSpec_Array(end+1, :) = blSpec;
        auc_Array (end+1, :)= auc;
        svmSens_Array(end+1, :) = svmSens;
        svmSpec_Array(end+1, :) = svmSpec;
        test_f_measure_Array(end+1, :) = test_f_measure;           
    end
    
else 
    error('wrong MODE');
end





%fprintf(['\n\n\nFinished ' ddiName ', ...
 %                       meanAUC = ' num2str(meanAUC, '%0.3f') '.\n\n\n']);

% print results to file.
%summaryFile = [folderResults 'AllVectorsChooseNegRand_02NOV2011_' ...
%                            FisherMode '_' SVMMode '_' Kernel '.summary'];
%save(summaryFile, 'AUC_Array', '-ascii');

return;
