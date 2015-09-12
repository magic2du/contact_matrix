function [aucBL, blSens, blSpec, ...
                    auc, svmSens, svmSpec, test_f_measure,...
                    trainAccuracy, trainRecall, trainPrecision, train_f_measure] = runSeqPairAllVectorsChooseNegRand_aaIndex_test_load_C_J ...
                        (ddiName, pairNbr, FisherMode, SVMMode, Kernel,choseNegRatio, choseNegRatioTest, C, Jj)

% pairNbr is the current sequence number doing the LOO cross validation.
% SVM mode 'SVMLIGHT', 

folder3did = ['/home/du/Protein_Protein_Interaction_Project/' ...
                                        '3did_20NOV2009/dom_dom_ints/'];
ddiPath = [folder3did ddiName '/'];
dataFolder=['/big/du/Protein_Protein_Interaction_Project/Contact_Matrix_Project/Vectors_Fishers_aaIndex_raw/' ddiName '/' ];
folderResults = ...
        ['/home/du/Protein_Protein_Interaction_Project/Contact_Matrix_Project/dom_dom_ints/' ddiName '/'];
if ~exist(folderResults, 'dir')
	mkdir(folderResults);
end

% load the data, find the seq. pairs that will be used for
% training. Leave one out
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
filePath=['/big/du/Protein_Protein_Interaction_Project/Contact_Matrix_Project/Vectors_Fishers_aaIndex_raw/' ddiName '/' 'pairsToRun.txt'];
pairs= load(filePath);

trainIdx = [];
for clCtr = 1:length(pairs)
    if pairNbr ~= pairs(clCtr)
        trainIdx(end+1) = pairs(clCtr);
    end
end

% get contact matrix baseline pred. from the training seq. pairs. Baseline
% prediction will come from the family, except the pair being tested.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
contactMatrixPath = ...
        ['/home/du/Protein_Protein_Interaction_Project/Contact_Matrix_Project/ContactMapExamples/' ...
                                        ddiName '/contactMapsBag.mat'];
load(contactMatrixPath); % contactMapsBag.
contactMapsBag(end) = []; % the last one is the average.
contactMatrixAv = zeros(size(contactMapsBag{1}));
numbContacts = [];
for idxCtr = 1:length(trainIdx)
    % average.
    contactMatrixAv = contactMatrixAv + contactMapsBag{trainIdx(idxCtr)};
    % number of contacts in the matrix
    numbContacts(end+1) = length(find(contactMapsBag{trainIdx(idxCtr)}));
end
avNumbContacts = round(mean(numbContacts));

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


% Train SVM.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%numbNegTrain = size(FamilyPO, 1);
% question: how many negatives? clustering?

%r=randperm(size(FamilyNE, 1));
%r=r(1:numbNegTrain);
%indTrainNE=sort(r);
%FamilyNE = FamilyNE(indTrainNE, :);

if strcmp(SVMMode, 'SVMLIGHT')
% SVMLIGHT.
svmlightFolder = ...
'/home/du/Protein_Protein_Interaction_Project/3did_20NOV2009/svm_light/';

trainFile = ...
    [folderResults FisherMode '_'  SVMMode '_' Kernel '_' num2str(choseNegRatio) '_' num2str(C) '_' num2str(Jj) '_AllVectorsChooseNegRand_aaIndex_test.train'];
modelFile = ...
    [folderResults FisherMode '_'  SVMMode '_' Kernel '_' num2str(choseNegRatio) '_' num2str(C) '_' num2str(Jj) '_AllVectorsChooseNegRand_aaIndex_test.model'];
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
        [svmlightFolder 'svm_learn -t 0 -c ' num2str(C) ' -j ' num2str(Jj) ' ' trainFile ' ' modelFile];
else
	error('ERROR');
end
system(command);

else
% MATLAB.
%Training = [FamilyPO; FamilyNE];
Training = [trainPO; trainNE];

Group = [ones(size(trainPO, 1), 1); zeros(size(trainNE, 1), 1)];
if strcmp(Kernel, 'RBF')
	SVMStruct = svmtrain(Training, Group, 'kernel_function', 'rbf');
elseif strcmp(Kernel, 'POLY')
	SVMStruct = svmtrain(Training, Group, 'kernel_function', 'polynomial');
else
	SVMStruct = svmtrain(Training, Group, 'kernel_function', 'linear');
end

end

% Testing.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
gtContactMatrix = contactMapsBag{pairNbr}; %ground truth.
% create positive and negative test sets.
[posTestI posTestJ] = ind2sub(size(gtContactMatrix), ...
                                            find(gtContactMatrix > 0));
[negTestI negTestJ] = ind2sub(size(gtContactMatrix), ...
                                            find(gtContactMatrix == 0));

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
    testFile = ...
        [folderResults FisherMode '_'  SVMMode '_' Kernel '_' num2str(choseNegRatio) '_' num2str(C) '_' num2str(Jj) '_AllVectorsChooseNegRand_aaIndex_test.testOnTrain'];
    resultFile = [folderResults 'AllVectorsChooseNegRand_aaIndex_test_' ...
        FisherMode '_'  SVMMode '_' Kernel '_' num2str(choseNegRatio) '_' num2str(C) '_' num2str(Jj) '_pair' num2str(pairNbr) '.resultOnTrain'];
    write_SVM_file(trainPO, trainNE, testFile);
    command = ...
    [svmlightFolder 'svm_classify ' testFile ' ' modelFile ' ' resultFile];
    system(command);
    trainResult = load(resultFile);
    trainGroundTruth = [ones(size(trainPO, 1), 1); ...
				    -1*ones(size(trainNE, 1), 1)];
    trainAUC = roc(trainResult, trainGroundTruth, 'nofigure');

    % test on test set:
    testFile = ...
        [folderResults FisherMode '_'  SVMMode '_' Kernel '_' num2str(choseNegRatio) '_' num2str(C) '_' num2str(Jj) '_AllVectorsChooseNegRand_aaIndex_test.test'];
    %resultFile = [folderResults 'AllVectorsChooseNegRand_02NOV2011_' ...
     %   FisherMode '_' SVMMode '_' Kernel '_pair' num2str(pairNbr) '.result'];
    resultFile = [folderResults 'AllVectorsChooseNegRand_aaIndex_test_' ...
        FisherMode '_'  SVMMode '_' Kernel '_' num2str(choseNegRatio) '_' num2str(C) '_' num2str(Jj) '_pair' num2str(pairNbr) '.result'];

    write_SVM_file(POTest, NETest, testFile);
    command = ...
    [svmlightFolder 'svm_classify ' testFile ' ' modelFile ' ' resultFile];
    system(command);
    testResult = load(resultFile);
    testGroundTruth = [ones(size(POTest, 1), 1); ...
				    -1*ones(size(NETest, 1), 1)];
    %GTFile = [folderResults 'AllVectorsChooseNegRand_02NOV2011_' ...
    %FisherMode '_' SVMMode '_' Kernel '_pair' num2str(pairNbr) '.groundtruth'];
    GTFile = [folderResults FisherMode '_'  SVMMode '_' Kernel '_' num2str(choseNegRatio) '_AllVectorsChooseNegRand_aaIndex_test.groundtruth'];
    save(GTFile, 'testGroundTruth', '-ascii');
    auc = roc(testResult, testGroundTruth, 'nofigure');


        
else
    % MATLAB. Matlab SVM seems to have problem.
    % test on training set.
    currdir = pwd;
    cd /home/du/Protein_Protein_Interaction_Project/Transductive_SVM_Project/matlab_code;
    [Classif Predict] = svmclassify_alv(SVMStruct, Training);
    % don't know why add negtive may be by accident need to validate it. 
    Predict = -Predict;
    cd(currdir);


    % 

    Sample = [POTest; NETest];
    currdir = pwd;
    cd /home/du/Protein_Protein_Interaction_Project/Transductive_SVM_Project/matlab_code;
    [Classif Predict] = svmclassify_alv(SVMStruct, Sample);
    % don't know why add negtive may be by accident need to validate it. 
    Predict = -Predict;
    cd(currdir);
    resultFile = [folderResults 'AllVectorsChooseNegRand_aaIndex_test_' ...
        FisherMode '_'  SVMMode '_' Kernel '_' num2str(choseNegRatio) '_pair' num2str(pairNbr) '.result'];
        save(resultFile, 'Predict', '-ascii');
    GroundTruth = [ones(size(POTest, 1), 1); ...
                    -1*ones(size(NETest, 1), 1)];
    GTFile = [folderResults FisherMode '_'  SVMMode '_' Kernel '_' num2str(choseNegRatio) '_AllVectorsChooseNegRand_aaIndex_test.groundtruth'];
    save(GTFile, 'GroundTruth', '-ascii');
    auc = roc(Predict, GroundTruth, 'nofigure');

end
% here start asses performance of training and testing data and baseline.
% assess performance.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% get the coordinates of groundtruth (positive and negative).
posTestIdx = [posTestI posTestJ];
negTestIdx = [negTestI negTestJ];
negTestIdxRdcd = negTestIdx(indTestNE, :);
                                    
% baseline.
[blPosI blPosJ] = ind2sub(size(contactMatrixAv), ...
                                            find(contactMatrixAv > 0));
[blNegI blNegJ] = ind2sub(size(contactMatrixAv), ...
                                        find(contactMatrixAv == 0));
blPredForRealPositives = ismember(posTestIdx, [blPosI blPosJ], 'rows');

blPredForRealNegatives = ismember(negTestIdxRdcd, [blPosI blPosJ], 'rows');
blTP = length(find(blPredForRealPositives == 1));
blFP = length(find(blPredForRealNegatives == 1));
blTN = length(find(blPredForRealNegatives == 0));
blFN = length(find(blPredForRealPositives == 0));
blSens = blTP/(blTP+blFN);
blSpec = blTP/(blTP+blFP);
if isnan(blSpec)
    blSpec = 0;
end


blPosPredict = ...
    contactMatrixAv(sub2ind(size(contactMatrixAv), posTestI, posTestJ));
blNegPredict = ...
    contactMatrixAv(sub2ind(size(contactMatrixAv), negTestI, negTestJ));
blNegPredictRdcd = blNegPredict(indTestNE);
blPredict = [blPosPredict; blNegPredictRdcd];
blGroundTruth=[ones(size(blPosPredict, 1), 1); ...
                    -1*ones(size(blNegPredictRdcd, 1), 1)];
aucBL = roc(blPredict, blGroundTruth, 'nofigure');

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
resultsFile = ...
        [folderResults 'AllVectorsChooseNegRand_aaIndex_test_' ...
       FisherMode '_'  SVMMode '_' Kernel '_' num2str(choseNegRatio) '_' num2str(C) '_' num2str(Jj) '_pair_' num2str(pairNbr) '.mat'];
save(resultsFile, 'gtContactMatrix', 'indTestNE', 'contactMatrixAv', ...
                    'testResult', 'avNumbContacts', ...
                    'aucBL', 'blSens', 'blSpec', ...
                    'auc', 'svmSens', 'svmSpec', 'test_f_measure',...
                    'trainAccuracy', 'trainRecall', 'trainPrecision', 'train_f_measure');

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
