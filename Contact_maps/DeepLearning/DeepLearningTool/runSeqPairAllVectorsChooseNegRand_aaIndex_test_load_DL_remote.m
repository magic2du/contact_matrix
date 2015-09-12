function [aucBL, blSens, blSpec, ...
                    auc, svmSens, svmSpec, test_f_measure,...
                    trainAccuracy, trainRecall, trainPrecision, train_f_measure] = runSeqPairAllVectorsChooseNegRand_aaIndex_test_load_DL_remote ...
                        (ddiName, pairNbr, FisherMode, SVMMode, Kernel,choseNegRatio, choseNegRatioTest)

folder3did = ['/big/du/Protein_Protein_Interaction_Project/Contact_Matrix_Project/Vectors_Fishers_aaIndex_raw/'];
                                        
ddiPath = [folder3did ddiName '/'];
dataFolder=ddiPath;
folderResults = ddiPath;

blSens = 0;
blSpec = 0;
aucBL = 0;
% load the data, find the seq. pairs that will be used for
% training. Leave one out
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
filePath=[ddiPath 'pairsToRun.txt'];
pairs= load(filePath);

trainIdx = [];
for clCtr = 1:length(pairs)
    if pairNbr ~= pairs(clCtr)
        trainIdx(end+1) = pairs(clCtr);
    end
end



% Fisher vectors. For all the training sequence pairs, go to the
% corresponding contact matrix, and get from there positive and negative
% Fisher vector pairs.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

FamilyPO = [];
FamilyNE = [];
trainPO = [];
trainNE = [];

%for seqCtr = setdiff(1:length(contactMapsBag), pairNbr)
for idxCtr = 1:length(trainIdx)
    seqCtr = trainIdx(idxCtr);
    
    % create positive and negative training sets.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\
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
        numbNegTrain = size(PO, 1) * choseNegRatio;
        if numbNegTrain>numNeg
            numbNegTrain=numNeg;
        end
        r=randperm(size(NE, 1));
        r=r(1:numbNegTrain);
        indTrainNE=sort(r);
        selectedNE = NE(indTrainNE, :);
    end
    trainNE=[trainNE; selectedNE];
    trainPO=[trainPO; PO];
end

% Testing.

% create positive and negative test sets.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
POTest=[];
NETest=[];

testFile=[dataFolder 'F0_20_F1_20_Sliding_17_11_F0_20_F1_20_Sliding_17_11_ouput_' num2str(pairNbr) '.txt'];
numPos=load([dataFolder 'numPos_' num2str(pairNbr) '.txt']);
numNeg=load([dataFolder 'numNeg_' num2str(pairNbr) '.txt']);
[selectedData, label]=chooseAAIndexVectores(testFile, FisherMode);
POTest=selectedData(1:numPos, :);
NETest=selectedData(numPos+1: numPos+numNeg, :);
                                      
% randomly choose negative examples to test.
if choseNegRatioTest~=0
    numbNegTest = numPos * choseNegRatioTest;
    if numbNegTest>numNeg
            numbNegTest=numNeg;
    end
    r=randperm(size(NETest, 1));
    r=r(1:numbNegTest);
    indTestNE=sort(r);
    NETest = NETest(indTestNE, :);
end

if strcmp(SVMMode, 'SVMLIGHT')

    %%%%%%%%%%%%%%%%%%%%%% if yes choose reduced ratio for training   %%%%%%%%%%%
    if 0
        n=size(trainNE, 1);
        r=randperm(size(trainNE, 1));
        r=r(1:int32(ceil(n/4)));
            indTrainNE=sort(r);
            selectedNE = trainNE(indTrainNE, :);
        trainNE=selectedNE;
        trainPO= trainPO(indTrainNE, :);
    end
    %%%%%%%%%%%%%%%%%%%%%%
    diary off;
    % SVMLIGHT.
    % test on training set.
    % SVMLIGHT.
    trainGroundTruth = [ones(size(trainPO, 1), 1); ...
                    -1*ones(size(trainNE, 1), 1)];
    testGroundTruth = [ones(size(POTest, 1), 1); ...
                    -1*ones(size(NETest, 1), 1)];
    svmlightFolder ='/home/du/Protein_Protein_Interaction_Project/svm_light_linux64_2013/';
  %  '/home/du/Protein_Protein_Interaction_Project/3did_20NOV2009/svm_light/';
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
    diary off;
    system(command);
    diary on;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%testing%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     % SVMLIGHT.
    % test on training set.
    diary off;
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
    diary on;
    fprintf('Training AUC: %0.3f%%\n', trainAUC * 100);
    fprintf('Training accuracy: %0.3f%%\n', accuracy * 100);
    fprintf('Training recall: %0.3f%%\n', recall * 100);
    fprintf('Training precision: %0.3f%%\n', precision * 100);
    fprintf('Training f_measure: %0.3f%%\n', f_measure * 100);
    trainAccuracy=accuracy;
    trainRecall=recall;
    trainPrecision=precision;
    train_f_measure=f_measure;
    
    diary off;
    % test on test set:
    testFile = ...
        [folderResults FisherMode '_'  SVMMode '_' Kernel '_' num2str(choseNegRatio) '_' num2str(choseNegRatioTest) '_remote_ChooseNegRand_aaIndex_test_load_DL_cross.test'];

    resultFile = [folderResults FisherMode '_'  SVMMode '_' Kernel '_' num2str(choseNegRatio) '_' num2str(choseNegRatioTest) '_remote_ChooseNegRand_aaIndex_test_load_DL_cross.result'];

    write_SVM_file(POTest, NETest, testFile);
    command = ...
    [svmlightFolder 'svm_classify ' testFile ' ' modelFile ' ' resultFile];
    system(command);
    testResult = load(resultFile);
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
    diary on;
    fprintf('Testing AUC: %0.3f%%\n', auc * 100);
    fprintf('Testing accuracy: %0.3f%%\n', accuracy * 100);
    fprintf('Testing recall: %0.3f%%\n', recall * 100);
    fprintf('Testing precision: %0.3f%%\n', precision * 100);
    fprintf('Testing f_measure: %0.3f%%\n', f_measure * 100);
    svmSens=recall;
    svmSpec=precision;
    test_f_measure=f_measure;            
elseif strcmp(SVMMode, 'DL')
    %%%%%%%%%%%%%%%%%%%%%% if yes choose reduced ratio for training   %%%%%%%%%%%
    if 0
        n=size(trainNE, 1);
        r=randperm(size(trainNE, 1));
        r=r(1:int32(ceil(n/2)));
            indTrainNE=sort(r);
            selectedNE = trainNE(indTrainNE, :);
        trainNE=selectedNE;
        trainPO= trainPO(indTrainNE, :);
    end
    %%%%%%%%%%%%%%%%%%%%%%
    diary off;
    trainGroundTruth = [ones(size(trainPO, 1), 1); ...
				    -1*ones(size(trainNE, 1), 1)];
    testGroundTruth = [ones(size(POTest, 1), 1); ...
				    -1*ones(size(NETest, 1), 1)];

    trainData=[trainPO; trainNE];
    trainLabel=[ones(size(trainPO, 1), 1); ...
				    zeros(size(trainNE, 1), 1)];
    testData=[POTest; NETest];
    testLabel=[ones(size(POTest, 1), 1); ...
				    zeros(size(NETest, 1), 1)];
    % run the DL algorithms
    [trainResult, testResult]= SAE_AllVectorsChooseNegRand_aaIndex_test_load_DL_remote(trainData, trainLabel, testData, testLabel);
    diary on;    
    auc = roc(testResult, testGroundTruth, 'nofigure');
    trainAUC = roc(trainResult, trainGroundTruth, 'nofigure');
    
elseif strcmp(SVMMode, 'DLUS')
%%%%%%%%%%%%%%%%%%%%%% Reduced training residue pairs.
    trainGroundTruth = [ones(size(trainPO, 1), 1); ...
				    -1*ones(size(trainNE, 1), 1)];
    testGroundTruth = [ones(size(POTest, 1), 1); ...
				    -1*ones(size(NETest, 1), 1)];

    trainData=[trainPO; trainNE];
    trainLabel=[ones(size(trainPO, 1), 1); ...
				    zeros(size(trainNE, 1), 1)];
    testData=[POTest; NETest];
    testLabel=[ones(size(POTest, 1), 1); ...
				    zeros(size(NETest, 1), 1)];
    % run the DL algorithms
    [trainResult, testResult]= SAE_AllVectorsChooseNegRand_aaIndex_test_load_DLUS_remote(trainPO, trainNE, testData, testLabel);
        
    auc = roc(testResult, testGroundTruth, 'nofigure');
    trainAUC = roc(trainResult, trainGroundTruth, 'nofigure');
elseif strcmp(SVMMode, 'SAESVM')
%%%%%%%%%%%%%%%%%%%%%%This is using SAE to encode than followed by SVM
    trainGroundTruth = [ones(size(trainPO, 1), 1); ...
				    -1*ones(size(trainNE, 1), 1)];
    testGroundTruth = [ones(size(POTest, 1), 1); ...
				    -1*ones(size(NETest, 1), 1)];

    trainData=[trainPO; trainNE];
    trainLabel=trainGroundTruth;
    testData=[POTest; NETest];
    testLabel=testGroundTruth;
    % run the DL algorithms
    [trainResult, testResult]= SAE_AllVectorsChooseNegRand_aaIndex_test_load_SAESVM(trainData, trainLabel, testData, testLabel, Kernel, ddiPath);
        
    auc = roc(testResult, testGroundTruth, 'nofigure');
    trainAUC = roc(trainResult, trainGroundTruth, 'nofigure');  
elseif strcmp(SVMMode, 'DLSTOP')
%%%%%%%%%%%%%%%%%%%%%%This add early stopping for the training.
    trainGroundTruth = [ones(size(trainPO, 1), 1); ...
				    -1*ones(size(trainNE, 1), 1)];
    testGroundTruth = [ones(size(POTest, 1), 1); ...
				    -1*ones(size(NETest, 1), 1)];

    trainData=[trainPO; trainNE];
    trainLabel=[ones(size(trainPO, 1), 1); ...
				    zeros(size(trainNE, 1), 1)];
    testData=[POTest; NETest];
    testLabel=[ones(size(POTest, 1), 1); ...
				    zeros(size(NETest, 1), 1)];
    % run the DL algorithms
    [trainResult, testResult] = SAE_AllVectorsChooseNegRand_aaIndex_test_load_DL_STOP_remote(trainData, trainLabel, testData, testLabel);
        
    auc = roc(testResult, testGroundTruth, 'nofigure');
    trainAUC = roc(trainResult, trainGroundTruth, 'nofigure');
elseif strcmp(SVMMode, 'DBN_STOP')
%%%%%%%%%%%%%%%%%%%%%%This add early stopping for the training.
    trainGroundTruth = [ones(size(trainPO, 1), 1); ...
				    -1*ones(size(trainNE, 1), 1)];
    testGroundTruth = [ones(size(POTest, 1), 1); ...
				    -1*ones(size(NETest, 1), 1)];

    trainData=[trainPO; trainNE];
    trainLabel=[ones(size(trainPO, 1), 1); ...
				    zeros(size(trainNE, 1), 1)];
    testData=[POTest; NETest];
    testLabel=[ones(size(POTest, 1), 1); ...
				    zeros(size(NETest, 1), 1)];
    % run the DL algorithms
    [trainResult, testResult] = DBN_AllVectorsChooseNegRand_aaIndex_test_load_DL_STOP_remote(trainData, trainLabel, testData, testLabel);
        
    auc = roc(testResult, testGroundTruth, 'nofigure');
    trainAUC = roc(trainResult, trainGroundTruth, 'nofigure');
elseif strcmp(SVMMode, 'DL_RE_US')
%%%%%%%%%%%%%%%%%%%%%%This is to reduce the traing sequence pairs for tranning
    trainGroundTruth = [ones(size(trainPO, 1), 1); ...
				    -1*ones(size(trainNE, 1), 1)];
    testGroundTruth = [ones(size(POTest, 1), 1); ...
				    -1*ones(size(NETest, 1), 1)];

    trainData=[trainPO; trainNE];
    trainLabel=[ones(size(trainPO, 1), 1); ...
				    zeros(size(trainNE, 1), 1)];
    testData=[POTest; NETest];
    testLabel=[ones(size(POTest, 1), 1); ...
				    zeros(size(NETest, 1), 1)];
    [reducedTrainingData, reducedTrainingLabel]=getReducedTrainingAndLabel(8, trainIdx, dataFolder, FisherMode, choseNegRatio);
    % run the DL algorithms
    [trainResult, testResultOnReducedModel, testResultOnUSModel] = SAE_AllVectorsChooseNegRand_aaIndex_test_DL_RE_US_remote(trainData, trainLabel, testData, testLabel, reducedTrainingData, reducedTrainingLabel);
        
    auc = roc(testResultOnUSModel, testGroundTruth, 'nofigure');
    testResult=testResultOnUSModel;
    trainAUC = roc(trainResult, trainGroundTruth, 'nofigure');  
    aucBL= roc(testResultOnReducedModel, testGroundTruth, 'nofigure');
elseif strcmp(SVMMode, 'DL_CLUSTER_US')
%%%%%%%%%%%%%%%%%%%%%%Using the clustered sequence for supervised training but use who family for
    trainGroundTruth = [ones(size(trainPO, 1), 1); ...
				    -1*ones(size(trainNE, 1), 1)];
    testGroundTruth = [ones(size(POTest, 1), 1); ...
				    -1*ones(size(NETest, 1), 1)];

    trainData=[trainPO; trainNE];
    trainLabel=[ones(size(trainPO, 1), 1); ...
				    zeros(size(trainNE, 1), 1)];
    testData=[POTest; NETest];
    testLabel=[ones(size(POTest, 1), 1); ...
				    zeros(size(NETest, 1), 1)];
    [wholeTrainingData, wholeTrainingLabel]=getWholeTrainingAndLabelForClustered(pairNbr, dataFolder, FisherMode, choseNegRatio);
    % run the DL algorithms
    if strcmp(Kernel, 'DL_CLUSTER_US') % this is for unsupvervised training
        [trainResult, testResultOnReducedModel, testResultOnUSModel] =SAE_AllVectorsChooseNegRand_aaIndex_test_DL_CLUSTER_US_remote(wholeTrainingData, wholeTrainingLabel, testData, testLabel, trainData, trainLabel);
    elseif strcmp(Kernel, 'SAE_US_SVM')
    %%%% here need to change the lables to -1 and 1
        trainLabel=[ones(size(trainPO, 1), 1); ...
				    -1*ones(size(trainNE, 1), 1)];
        testLabel=[ones(size(POTest, 1), 1); ...
				    -1*ones(size(NETest, 1), 1)];
        [trainResult, testResultOnReducedModel, testResultOnUSModel] =SAE_aaIndex_test_DL_CLUSTER_US_SVM_remote(wholeTrainingData, wholeTrainingLabel, testData, testLabel, trainData, trainLabel, ddiPath);
    end    
    auc = roc(testResultOnUSModel, testGroundTruth, 'nofigure');
    testResult=testResultOnUSModel;
    trainAUC = 0;  
    aucBL= roc(testResultOnReducedModel, testGroundTruth, 'nofigure');
else 
    error('wrong MODE');
end
% here start asses performance of training and testing data and baseline.
% assess performance.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% svm.
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
fprintf('Training accuracy: %0.3f%%\n', accuracy * 100);
fprintf('Training recall: %0.3f%%\n', recall * 100);
fprintf('Training precision: %0.3f%%\n', precision * 100);
fprintf('Training f_measure: %0.3f%%\n', f_measure * 100);
trainAccuracy=accuracy;
trainRecall=recall;
trainPrecision=precision;
train_f_measure=f_measure;
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

fprintf('Testing accuracy: %0.3f%%\n', accuracy * 100);
fprintf('Testing recall: %0.3f%%\n', recall * 100);
fprintf('Testing precision: %0.3f%%\n', precision * 100);
fprintf('Testing f_measure: %0.3f%%\n', f_measure * 100);
svmSens=recall;
svmSpec=precision;
test_f_measure=f_measure;

% save and print results.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


fprintf(['\n\n\n' ddiName ', seqPair ' num2str(pairNbr) ...
            ', auc = ' num2str(auc, '%0.3f') ...
            ', sens = ' num2str(svmSens, '%0.3f') ...
            ', spec = ' num2str(svmSpec, '%0.3f') ...
            ', blAUC = ' num2str(aucBL, '%0.3f') ...
            ', blSens = ' num2str(blSens, '%0.3f') ...
            ', blSpec = ' num2str(blSpec, '%0.3f') '.\n\n\n']);
            diary off;
            diary on;
                    
return;
