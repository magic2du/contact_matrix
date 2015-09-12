function [aucBL, blSens, blSpec, ...
                    auc, svmSens, svmSpec, test_f_measure,...
                    trainAccuracy, trainRecall, trainPrecision, train_f_measure] = runSeqPairAllVectorsChooseNegRand_aaIndex_test_load_DL ...
                        (ddiName, pairNbr, FisherMode, SVMMode, Kernel,choseNegRatio, choseNegRatioTest)



folder3did = ['/home/michael/Documents/Contact_Matrix_Project/Vectors_Fishers_aaIndex_raw/'];
                                        
ddiPath = [folder3did ddiName '/'];
dataFolder=ddiPath;
folderResults = ddiPath;


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
    % SVMLIGHT.
    % test on training set.
        
elseif strcmp(SVMMode, 'DL')

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
    [trainResult, testResult]= SAE_AllVectorsChooseNegRand_aaIndex_test_load_DL(trainData, trainLabel, testData, testLabel);
        
    auc = roc(testResult, testGroundTruth, 'nofigure');
    trainAUC = roc(trainResult, trainGroundTruth, 'nofigure');
elseif strcmp(SVMMode, 'DLUS')
%%%%%%%%%%%%%%%%%%%%%%
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
    [trainResult, testResult]= SAE_AllVectorsChooseNegRand_aaIndex_test_load_DLUS(trainPO, trainNE, testData, testLabel);
        
    auc = roc(testResult, testGroundTruth, 'nofigure');
    trainAUC = roc(trainResult, trainGroundTruth, 'nofigure');    
elseif strcmp(SVMMode, 'SAESVM')
%%%%%%%%%%%%%%%%%%%%%%
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
%%%%%%%%%%%%%%%%%%%%%%
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
    [trainResult, testResult] = SAE_AllVectorsChooseNegRand_aaIndex_test_load_DL_STOP(trainData, trainLabel, testData, testLabel);
        
    auc = roc(testResult, testGroundTruth, 'nofigure');
    trainAUC = roc(trainResult, trainGroundTruth, 'nofigure'); 
else 
    error('wrong MODE');
end
% here start asses performance of training and testing data and baseline.
% assess performance.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


blSens = 0;
blSpec = 0;
aucBL = 0;

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
fprintf('Training AUC: %0.3f%%\n', trainAUC * 100);
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
fprintf('Testing AUC: %0.3f%%\n', auc * 100);
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
