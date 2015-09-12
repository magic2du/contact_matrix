function [aucBL, blSens, blSpec, ...
                    auc, svmSens, svmSpec, test_f_measure,...
                    trainAccuracy, trainRecall, trainPrecision, train_f_measure] = runSeqPairAllVectorsChooseNegRand_aaIndex_test ...
                        (ddiName, pairNbr, FisherMode, SVMMode, Kernel,choseNegRatio )
%function auc = runSeqPairAllVectorsChooseNegRand_02NOV2011()
% pairNbr is the current sequence number doing the LOO cross validation.
% SVM mode 'SVMLIGHT', 

folder3did = ['/home/du/Protein_Protein_Interaction_Project/' ...
                                        '3did_20NOV2009/dom_dom_ints/'];
ddiPath = [folder3did ddiName '/'];
folderResults = ...
        ['/home/du/Protein_Protein_Interaction_Project/Contact_Matrix_Project/dom_dom_ints/' ddiName '/'];
if ~exist(folderResults, 'dir')
	mkdir(folderResults);
end

% load the graph clustering data, find the seq. pairs that will be used for
% training.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
percThr = 90;
load([ddiPath 'MC' num2str(percThr) '.mat']);
% iterate through all the clusters. If testPair is in cluster, none of the
% pairs in the cluster is used for training. Ow pick one single example in
% the cluster for training, the first one in the cluster.
trainIdx = [];
for clCtr = 1:size(MC, 2)
    pairsInCl = find(MC(:, clCtr) == 1);
    if ~ismember(pairNbr, pairsInCl)
        %randIdx = randi(length(pairsInCl), 1);
        %trainIdx(end+1, 1) = pairsInCl(randIdx);
        trainIdx(end+1, 1) = pairsInCl(1);
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
load([ddiPath 'FisherA.mat']); % AFisherM0[1]Array, AconstFisherM0[1]Array.
load([ddiPath 'FisherB.mat']); % BFisherM0[1]Array, BconstFisherM0[1]Array.

load([ddiPath 'ddi_str_array.mat']); % AFisherM0[1]Array, AconstFisherM0[1]Array.

% read HMM structures for the two domains involved in the interaction.
domA = ddi_str_array{1}.domainA;
domB = ddi_str_array{1}.domainB;
hmmA = pfamhmmread(['/home/du/Protein_Protein_Interaction_Project/' ...
                        'PFAM_2008/SINGLE_FILES/' domA '.pfam']);
hmmB = pfamhmmread(['/home/du/Protein_Protein_Interaction_Project/' ...
                        'PFAM_2008/SINGLE_FILES/' domB '.pfam']);

FamilyPO = [];
FamilyNE = [];
trainPO = [];
trainNE = [];

%for seqCtr = setdiff(1:length(contactMapsBag), pairNbr)
for idxCtr = 1:length(trainIdx)
    seqCtr = trainIdx(idxCtr);
    
    %compute the HMM aligned sequence ignoring insert, should be HMM length
    seqA = ddi_str_array{seqCtr}.ASequence;
    seqB = ddi_str_array{seqCtr}.BSequence;
    % seqA
    [scoreA, algnA] = hmmprofalign(hmmA, seqA, 'flanks', true);
    indDelA = strfind(algnA, '-');
    indMatchDelA = union(indDelA, regexp(algnA, '[A-Z]'));
    conservedProteinSequenceA = algnA(indMatchDelA); 
    ddi_str_array{seqCtr}.conservedProteinSequenceA=conservedProteinSequenceA;
    % seqB
    [scoreB, algnB] = hmmprofalign(hmmB, seqB, 'flanks', true);
    indDelB = strfind(algnB, '-');
    indMatchDelB = union(indDelB, regexp(algnB, '[A-Z]'));
    conservedProteinSequenceB = algnB(indMatchDelB); 
    ddi_str_array{seqCtr}.conservedProteinSequenceB=conservedProteinSequenceB;
    
    %get FisherVectors
    command = ['AFisherVectors = A' FisherMode 'Array{seqCtr};'];
    eval(command);
    
    command = ['BFisherVectors = B' FisherMode 'Array{seqCtr};'];
    eval(command);
    
    % indexes of positive and negative examples.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [posI posJ] = ind2sub(size(contactMapsBag{seqCtr}), ...
                            find(contactMapsBag{seqCtr} > 0));
    [negI negJ] = ind2sub(size(contactMapsBag{seqCtr}), ...
                            find(contactMapsBag{seqCtr} == 0));
    
    % create positive and negative training sets.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    PO=[];
    NE=[];
    for i=1:length(posI)
        PO = [PO; 
            AFisherVectors(posI(i), :) getAAindexVector(conservedProteinSequenceA, posI(i), 5) ...
            BFisherVectors(posJ(i), :) getAAindexVector(conservedProteinSequenceB, posJ(i), 5)];
    end
    for i=1:length(negI)
    NE = [NE;
        AFisherVectors(negI(i), :) getAAindexVector(conservedProteinSequenceA, negI(i), 5) ...
        BFisherVectors(negJ(i), :)  getAAindexVector(conservedProteinSequenceB, negJ(i), 5)];
    end
    FamilyPO = [FamilyPO; PO];
    FamilyNE = [FamilyNE; NE];
    % chose negatives as in the the ratio default is 1.
    selectedNE=[];
    if choseNegRatio==0
        selectedNE=NE;
    else
        numbNegTrain = size(PO, 1) * choseNegRatio;
        r=randperm(size(NE, 1));
        r=r(1:numbNegTrain);
        indTrainNE=sort(r);
        selectedNE = NE(indTrainNE, :);
    end
    trainNE=[trainNE; selectedNE];


end
trainPO=FamilyPO; %use all the Positive examples as 


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
    [folderResults FisherMode '_'  SVMMode '_' Kernel '_' num2str(choseNegRatio) '_AllVectorsChooseNegRand_aaIndex_test.train'];
modelFile = ...
    [folderResults FisherMode '_'  SVMMode '_' Kernel '_' num2str(choseNegRatio) '_AllVectorsChooseNegRand_aaIndex_test.model'];
% write_SVM_file(FamilyPO, FamilyNE, trainFile);
write_SVM_file(trainPO, trainNE, trainFile);
if strcmp(Kernel, 'RBF')
	command = ...
        [svmlightFolder 'svm_learn -t 2 -g 1 ' trainFile ' ' modelFile];
elseif strcmp(Kernel, 'POLY')
	command = ...
        [svmlightFolder 'svm_learn -t 1 -d 3 ' trainFile ' ' modelFile];
elseif strcmp(Kernel, 'LINEAR')
	command = ...
        [svmlightFolder 'svm_learn -t 0 ' trainFile ' ' modelFile];
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
command = ['ATestFisherVectors = A' FisherMode 'Array{pairNbr};'];
eval(command);
command = ['BTestFisherVectors = B' FisherMode 'Array{pairNbr};'];
eval(command);

% create positive and negative test sets.
[posTestI posTestJ] = ind2sub(size(gtContactMatrix), ...
                                            find(gtContactMatrix > 0));
[negTestI negTestJ] = ind2sub(size(gtContactMatrix), ...
                                            find(gtContactMatrix == 0));
%compute the HMM aligned sequence ignoring insert, should be HMM length
seqA = ddi_str_array{pairNbr}.ASequence;
seqB = ddi_str_array{pairNbr}.BSequence;
% seqA
[scoreA, algnA] = hmmprofalign(hmmA, seqA, 'flanks', true);
indDelA = strfind(algnA, '-');
indMatchDelA = union(indDelA, regexp(algnA, '[A-Z]'));
conservedProteinSequenceA = algnA(indMatchDelA); 
ddi_str_array{pairNbr}.conservedProteinSequenceA=conservedProteinSequenceA;
% seqB
[scoreB, algnB] = hmmprofalign(hmmB, seqB, 'flanks', true);
indDelB = strfind(algnB, '-');
indMatchDelB = union(indDelB, regexp(algnB, '[A-Z]'));
conservedProteinSequenceB = algnB(indMatchDelB); 
ddi_str_array{pairNbr}.conservedProteinSequenceB=conservedProteinSequenceB;
                                          
POTest=[];
NETest=[];
for i=1:length(posTestI)
    POTest = [POTest; 
        ATestFisherVectors(posTestI(i), :) getAAindexVector(conservedProteinSequenceA, posTestI(i), 5) ...
        BTestFisherVectors(posTestJ(i), :) getAAindexVector(conservedProteinSequenceB, posTestJ(i), 5) ];
end
for i=1:length(negTestI)
NETest = [NETest;
    ATestFisherVectors(negTestI(i), :) getAAindexVector(conservedProteinSequenceA, negTestI(i), 5) ...
    BTestFisherVectors(negTestJ(i), :) getAAindexVector(conservedProteinSequenceB, negTestJ(i), 5)];
end                                        

% randomly choose negative examples to test.
if choseNegRatio~=0
    numbNegTest = size(POTest, 1) * choseNegRatio;
    r=randperm(size(NETest, 1));
    r=r(1:numbNegTest);
    indTestNE=sort(r);
    NETest = NETest(indTestNE, :);
end

if strcmp(SVMMode, 'SVMLIGHT')
% SVMLIGHT.
% test on training set.
testFile = ...
    [folderResults FisherMode '_'  SVMMode '_' Kernel '_' num2str(choseNegRatio) '_AllVectorsChooseNegRand_aaIndex_test.testOnTrain'];
resultFile = [folderResults 'AllVectorsChooseNegRand_aaIndex_test_' ...
    FisherMode '_'  SVMMode '_' Kernel '_' num2str(choseNegRatio) '_pair' num2str(pairNbr) '.resultOnTrain'];
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
    [folderResults FisherMode '_'  SVMMode '_' Kernel '_' num2str(choseNegRatio) '_AllVectorsChooseNegRand_aaIndex_test.test'];
%resultFile = [folderResults 'AllVectorsChooseNegRand_02NOV2011_' ...
 %   FisherMode '_' SVMMode '_' Kernel '_pair' num2str(pairNbr) '.result'];
resultFile = [folderResults 'AllVectorsChooseNegRand_aaIndex_test_' ...
    FisherMode '_'  SVMMode '_' Kernel '_' num2str(choseNegRatio) '_pair' num2str(pairNbr) '.result'];

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


% test on test set:

Sample = [POTest; NETest];
currdir = pwd;
cd /home/du/Protein_Protein_Interaction_Project/Transductive_SVM_Project/matlab_code;
[Classif Predict] = svmclassify_alv(SVMStruct, Sample);
% don't know why add negtive may be by accident need to validate it. 
Predict = -Predict;
cd(currdir);
resultFile = [folderResults 'AllVectorsChooseNegRand_aaIndex_test_' ...
    FisherMode '_'  SVMMode '_' Kernel '_' num2str(choseNegRatio) '_pair' num2str(pairNbr) '.result'];save(resultFile, 'Predict', '-ascii');
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
aucBL = roc(blPredict, testGroundTruth, 'nofigure');

% svm.
% for training performance.
trainResult0_1=zeros(length(trainResult), 1);
trainResult0_1(find(trainResult>=0))=1;
trainGroundTruth(find(trainGroundTruth==-1))=0; % change the format to 0 and 1;
[accuracy, sensitivity, specificity, precision, recall, f_measure, gmean]=Evaluate(trainGroundTruth, trainResult0_1); %Evaluate(ACTUAL,PREDICTED)

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
       FisherMode '_'  SVMMode '_' Kernel '_' num2str(choseNegRatio) '_pair_' num2str(pairNbr) '.mat'];
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
                    
return;
