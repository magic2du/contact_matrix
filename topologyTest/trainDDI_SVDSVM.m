function trainDDI_SVDSVM(ddi_folder, FisherMode, SVMMode)

%{
ddi_folder = '/home/alvaro/Protein_Protein_Interaction_Project/3did_15OCT2010/dom_dom_ints/Homoserine_dh_int_NAD_binding_3';
FisherMode = 'FisherM0';
%}

% positive dataset.
load([ddi_folder '/FisherA.mat']);
load([ddi_folder '/FisherB.mat']);
trainPO = [];
for FisherCtr = 1:length(AFisherM0Array)
    command = ...
        ['trainPO(end+1, :) = [(A' FisherMode 'Array{FisherCtr}(:)).'' ' ...
                                '(B' FisherMode 'Array{FisherCtr}(:)).''];'];
    eval(command);
end
clear *Fisher*Array;

% negative dataset.
load([ddi_folder '/FisherANegTrain.mat']);
load([ddi_folder '/FisherBNegTrain.mat']);
trainNE = [];
for FisherCtr = 1:length(AFisherM0ArrayNegTrain)
    command = ...
        ['trainNE(end+1, :) = [(A' FisherMode 'ArrayNegTrain{FisherCtr}(:)).'' ' ...
                                '(B' FisherMode 'ArrayNegTrain{FisherCtr}(:)).''];'];
    eval(command);
end
clear *Fisher*ArrayNegTrain;

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
SVDFile = [ddi_folder '/' FisherMode 'SVD.mat'];
save(SVDFile, 'U', 'S', 'V', 'numDim');

trainPOred = trainPO*V(:, 1:numDim);
trainNEred = trainNE*V(:, 1:numDim);

if strcmp(SVMMode, 'SVMLIGHT')
% svmlight.

% train.
svmlightFolder = ...
'/home/alvaro/Protein_Protein_Interaction_Project/3did_20NOV2009/svm_light/';
trainFile = [ddi_folder '/' FisherMode 'WholeFamily.train'];
modelFile = [ddi_folder '/' FisherMode 'WholeFamily.model'];
write_SVM_file(trainPOred, trainNEred, trainFile);
%command = [svmlightFolder 'svm_learn -t 2 -g 2 ' trainFile ' ' modelFile];
command = [svmlightFolder 'svm_learn -t 1 -d 3 ' trainFile ' ' modelFile];
system(command);

else
% MATLAB.

Training = [trainPOred; trainNEred];
Group = [1*ones(size(trainPOred, 1), 1); ...
        -1*ones(size(trainPOred, 1), 1)];
SVMStruct = svmtrain(Training, Group, 'Kernel_Function', 'polynomial');
SVMStructFile = [ddi_folder '/' FisherMode SVMMode 'WholeFamily.mat'];
%SVMStructFileError = [ddi_folder '/' FisherMode SVMMode 'WholeFamily.map'];
%system(['rm -fr ' SVMStructFileError]);
save(SVMStructFile, 'SVMStruct');

end

return;