function auc = runSeqPairAllVectorsChooseNegRand_aaIndex_test_backup_Sep4 ...
                        (ddiName, pairNbr, FisherMode, SVMMode, Kernel, chooseNegRatio)
%function auc = runSeqPairAllVectorsChooseNegRand_02NOV2011()
% pairNbr is the current sequence number that is left one out.
% SVMMode chose between algo to do the training and testing. "SVMLIGHT"
% otherwise use matlab SVM.
% Kernel can be 'LINEAR'. "POLY" poly nomial degree =3.
% chooseNegRatio chose random negtive examples as train data.
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
end

% Train SVM.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numbNegTrain = size(FamilyPO, 1);
% question: how many negatives? clustering?

r=randperm(size(FamilyNE, 1));
r=r(1:numbNegTrain);
indTrainNE=sort(r);
FamilyNE = FamilyNE(indTrainNE, :);

if strcmp(SVMMode, 'SVMLIGHT')
% SVMLIGHT.
svmlightFolder = ...
'/home/du/Protein_Protein_Interaction_Project/3did_20NOV2009/svm_light/';

trainFile = ...
    [folderResults FisherMode '_AllVectorsChooseNegRand_02NOV2011.train'];
modelFile = ...
    [folderResults FisherMode '_AllVectorsChooseNegRand_02NOV2011.model'];
write_SVM_file(FamilyPO, FamilyNE, trainFile);
if strcmp(Kernel, 'RBF')
	command = ...
        [svmlightFolder 'svm_learn -t 2 -g 1 ' trainFile ' ' modelFile];
elseif strcmp(Kernel, 'POLY')
	command = ...
        [svmlightFolder 'svm_learn -t 1 -d 3 ' trainFile ' ' modelFile];
elseif strcmp(Kernel, 'LINEAR')
	command = ...
        [svmlightFolder 'svm_learn -t 0 /home/du/Protein_Protein_Interaction_Project/3did_20NOV2009/svm_light/svm_learn -t 0/home/du/Protein_Protein_Interaction_Project/Contact_Matrix_Project/dom_dom_ints/PF00004.21_int_PF09336.2/FisherM1_SVMLIGHT_LINEAR_1_AllVectorsChooseNegRand_aaIndex_test.train /home/du/Protein_Protein_Interaction_Project/Contact_Matrix_Project/dom_dom_ints/PF00004.21_int_PF09336.2/FisherM1_SVMLIGHT_LINEAR_1_AllVectorsChooseNegRand_aaIndex_test.model' trainFile ' ' modelFile];
else
	error('ERROR');
end
system(command);

else
% MATLAB.
Training = [FamilyPO; FamilyNE];
Group = [ones(size(FamilyPO, 1), 1); zeros(size(FamilyNE, 1), 1)];
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
numbNegTest = size(POTest, 1);
r=randperm(size(NETest, 1));
r=r(1:numbNegTest);
indTestNE=sort(r);
NETest = NETest(indTestNE, :);

if strcmp(SVMMode, 'SVMLIGHT')
% SVMLIGHT.
testFile = ...
    [folderResults FisherMode '_AllVectorsChooseNegRand_02NOV2011.test'];
resultFile = [folderResults 'AllVectorsChooseNegRand_02NOV2011_' ...
    FisherMode '_' SVMMode '_' Kernel '_pair' num2str(pairNbr) '.result'];
write_SVM_file(POTest, NETest, testFile);
command = ...
[svmlightFolder 'svm_classify ' testFile ' ' modelFile ' ' resultFile];
system(command);
result = load(resultFile);
GroundTruth = [ones(size(POTest, 1), 1); ...
				-1*ones(size(NETest, 1), 1)];
GTFile = [folderResults 'AllVectorsChooseNegRand_02NOV2011_' ...
FisherMode '_' SVMMode '_' Kernel '_pair' num2str(pairNbr) '.groundtruth'];
save(GTFile, 'GroundTruth', '-ascii');
auc = roc(result, GroundTruth, 'nofigure');
        
else
% MATLAB.
Sample = [POTest; NETest];
currdir = pwd;
cd /home/du/Protein_Protein_Interaction_Project/Transductive_SVM_Project/matlab_code;
[Classif Predict] = svmclassify_alv(SVMStruct, Sample);
Predict = -Predict;
cd(currdir);
resultFile = [folderResults 'AllVectorsChooseNegRand_02NOV2011_' ...
    FisherMode '_' SVMMode '_' Kernel '_pair' num2str(pairNbr) '.result'];
save(resultFile, 'Predict', '-ascii');
GroundTruth = [ones(size(POTest, 1), 1); ...
                -1*ones(size(NETest, 1), 1)];
GTFile = [folderResults 'AllVectorsChooseNegRand_02NOV2011_' ...
FisherMode '_' SVMMode '_' Kernel '_pair' num2str(pairNbr) '.groundtruth'];
save(GTFile, 'GroundTruth', '-ascii');
auc = roc(Predict, GroundTruth, 'nofigure');

end

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
aucBL = roc(blPredict, GroundTruth, 'nofigure');

% svm.
allTestPairs = [posTestIdx; negTestIdxRdcd];
[resultSrt ind] = sort(result, 'descend');
% I realized that there are cases in which avNumbContacts is greater than
% the number of test examples, which makes the following line throw an
% error. Therefore I'll just use the number of known test positives. Even
% though this is a little cheating, we'll use mostly the roc to compare
% performances.

% 11/09/2011 I realized the previous is wrong, because it forces the
% sensitivity to always be equal to the specificity (see notes). Will
% change it in the reportResults function so that we predict positive
% what's on the positive side of the hyperplane, ow we predict negative.

%svmPos = allTestPairs(ind(1:avNumbContacts), :);
svmPos = allTestPairs(ind(1:length(posTestI)), :);
%svmNeg = allTestPairs(ind(avNumbContacts+1:end), :);
svmNeg = allTestPairs(ind(length(posTestI)+1:end), :);
svmTP = intersect([posTestI posTestJ], svmPos, 'rows');
svmFP = setdiff(svmPos, [posTestI posTestJ], 'rows');
svmTN = intersect([negTestI negTestJ], svmNeg, 'rows');
svmFN = setdiff(svmNeg, [negTestI negTestJ], 'rows');
svmSens = size(svmTP, 1)/length(posTestI);
svmSpec = size(svmTP, 1)/size(svmPos, 1);

% save and print results.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
resultsFile = ...
        [folderResults 'AllVectorsChooseNegRand_02NOV2011_results_' ...
        FisherMode '_' SVMMode '_' Kernel '_pair' num2str(pairNbr) '.mat'];
save(resultsFile, 'gtContactMatrix', 'indTestNE', 'contactMatrixAv', ...
                    'result', 'avNumbContacts', ...
                    'aucBL', 'blSens', 'blSpec', ...
                    'auc', 'svmSens', 'svmSpec');

fprintf(['\n\n\n' ddiName ', seqPair ' num2str(pairNbr) ...
            ', auc = ' num2str(auc, '%0.3f') ...
            ', sens = ' num2str(svmSens, '%0.3f') ...
            ', spec = ' num2str(svmSpec, '%0.3f') ...
            ', blAUC = ' num2str(aucBL, '%0.3f') ...
            ', blSens = ' num2str(blSens, '%0.3f') ...
            ', blSpec = ' num2str(blSpec, '%0.3f') '.\n\n\n']);
                    
return;