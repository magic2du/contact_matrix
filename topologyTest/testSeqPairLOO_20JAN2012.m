function [auc Zscore Pvalue] = ...
        testSeqPairLOO_20JAN2012(ddi_folder, pairNbr, FisherMode, SVMMode)

%{
ddi_folder = '/home/alvaro/Protein_Protein_Interaction_Project/3did_15OCT2010/dom_dom_ints/Homoserine_dh_int_NAD_binding_3';
pairNbr = 1;
FisherMode = 'FisherM0';
%}

% positive datasets (train and test).
load([ddi_folder '/FisherA.mat']);
load([ddi_folder '/FisherB.mat']);
trainPO = [];
for FisherCtr = setdiff(1:length(AFisherM0Array), pairNbr)
    command = ...
        ['trainPO(end+1, :) = [(A' FisherMode 'Array{FisherCtr}(:)).'' ' ...
                                '(B' FisherMode 'Array{FisherCtr}(:)).''];'];
    eval(command);
end
command = ['testPO = [(A' FisherMode 'Array{pairNbr}(:)).'' ' ...
                        '(B' FisherMode 'Array{pairNbr}(:)).''];'];
eval(command);
clear *Fisher*Array;

% negative train dataset.
load([ddi_folder '/FisherANegTrain.mat']);
load([ddi_folder '/FisherBNegTrain.mat']);
trainNE = [];
for FisherCtr = setdiff(1:length(AFisherM0ArrayNegTrain), pairNbr)
    command = ...
        ['trainNE(end+1, :) = [(A' FisherMode 'ArrayNegTrain{FisherCtr}(:)).'' ' ...
                                '(B' FisherMode 'ArrayNegTrain{FisherCtr}(:)).''];'];
    eval(command);
end
clear *Fisher*ArrayNegTrain;

% negative test dataset.
load([ddi_folder '/FisherANegTest.mat']);
load([ddi_folder '/FisherBNegTest.mat']);
testNE = [];
for FisherCtr = 1:length(AFisherM0ArrayNegTest)
    command = ...
        ['testNE(end+1, :) = [(A' FisherMode 'ArrayNegTest{FisherCtr}(:)).'' ' ...
                                '(B' FisherMode 'ArrayNegTest{FisherCtr}(:)).''];'];
    eval(command);
end
clear *Fisher*ArrayNegTest;

% svd on positive train dataset, exclusively.
[U S V] = svd(trainPO - repmat(mean(trainPO), size(trainPO, 1), 1));

% reduce dimensionality.

% how many singular components make up for 80% or more of the variance?
singValuesPO = diag(S(1:size(trainPO, 1), 1:size(trainPO, 1)));
relVarPO = (1/sum(singValuesPO.^2))*singValuesPO.^2;
numDim = 1;
addedVar = relVarPO(numDim);
while addedVar < 0.80
    numDim = numDim + 1;
    addedVar = addedVar + relVarPO(numDim);
end

trainPOred = trainPO*V(:, 1:numDim);
testPOred = testPO*V(:, 1:numDim);
trainNEred = trainNE*V(:, 1:numDim);
testNEred = testNE*V(:, 1:numDim);

if strcmp(SVMMode, 'SVMLIGHT')
% SVMLIGHT.

% train.
svmlightFolder = ...
'/home/du/Protein_Protein_Interaction_Project/3did_20NOV2009/svm_light/';
trainFile = [ddi_folder '/' FisherMode '.train'];
modelFile = [ddi_folder '/' FisherMode '.model'];
write_SVM_file(trainPOred, trainNEred, trainFile);
%command = [svmlightFolder 'svm_learn -t 2 -g 2 ' trainFile ' ' modelFile];
command = [svmlightFolder 'svm_learn -t 1 -d 3 ' trainFile ' ' modelFile];
system(command);

% test.
testFile = [ddi_folder '/' FisherMode '.test'];
resultFile = [ddi_folder '/' FisherMode SVMMode 'pair' num2str(pairNbr) '.result'];
write_SVM_file(testPOred, testNEred, testFile);
command = ...
[svmlightFolder 'svm_classify ' testFile ' ' modelFile ' ' resultFile];
system(command);
result = load(resultFile);
GroundTruth = [ones(size(testPOred, 1), 1); ...
                -1*ones(size(testNEred, 1), 1)];
auc = roc(result, GroundTruth, 'nofigure');
dev = result(1) - mean(result);
Zscore = dev/std(result);
distancesToHP = result; % for Pvalue.

else
% MATLAB.

% train.
Training = [trainPOred; trainNEred];
Group = [1*ones(size(trainPOred, 1), 1); ...
        -1*ones(size(trainPOred, 1), 1)];
SVMStruct = svmtrain(Training, Group, 'Kernel_Function', 'polynomial');

% test.
Sample = [testPOred; testNEred];
currdir = pwd;
cd /home/du/Protein_Protein_Interaction_Project/Transductive_SVM_Project/matlab_code;
[Classif Predict] = svmclassify_alv(SVMStruct, Sample);
Predict = -Predict;
cd(currdir);
resultFile = [ddi_folder '/' FisherMode SVMMode 'pair' num2str(pairNbr) '.result'];
save(resultFile, 'Predict', '-ascii');
GroundTruth = [ones(size(testPOred, 1), 1); ...
                -1*ones(size(testNEred, 1), 1)];
auc = roc(Predict, GroundTruth, 'nofigure');
dev = Predict(1) - mean(Predict);
Zscore = dev/std(Predict);
distancesToHP = Predict; % for Pvalue.

end

% let's find the Pvalue (empirically).
minDist = min(distancesToHP);
maxDist = max(distancesToHP);
binWidth = (maxDist-minDist)/100;
edges = minDist:binWidth:maxDist;
distHist = histc(distancesToHP, edges);
posDist = distancesToHP(1);
binsArea = distHist*binWidth;
ptrEdgesBeyondPos = find(edges > posDist);
if ~isempty(ptrEdgesBeyondPos)
    binsAreaBeyondPos = distHist(ptrEdgesBeyondPos)*binWidth;
    Pvalue = sum(binsAreaBeyondPos)/sum(binsArea);
else
    Pvalue = 0;
end

return;
