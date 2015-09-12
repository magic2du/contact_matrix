function stackedAE_DDI_f3(ddiName, hiddenSizeL1, hiddenSizeL2, trainRand, testRand, maxIteration, choseNegRatio)
%feature vectors{F0=20,F1=20, Sliding=17*11}*2 and output={0,1}. output 1 means residuepairs are contact pairs. lenght=455
%trainRand=1 train data set chosen random negtive, other wise use whole dataset.
%testRand=1 train data set chosen random negtive

%%======================================================================
%% STEP 0: Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; you do not need to 
%  change the parameters below.

DISPLAY = false;
inputSize = 454;	%feature vectors{F0=20,F1=20, Sliding=17*11}*2 and output={0,1}. output 1 means residuepairs are contact pairs. lenght=455
numClasses = 2;
%hiddenSizeL1 = 200;    % Layer 1 Hidden Size
%hiddenSizeL2 = 200;    % Layer 2 Hidden Size
%hiddenSizeL1 = 210;    % Layer 1 Hidden Size
%hiddenSizeL2 = 220;    % Layer 2 Hidden Size
sparsityParam = 0.1;   % desired average activation of the hidden units.
                       % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		               %  in the lecture notes). 
lambda = 3e-3;         % weight decay parameter       
beta = 3;              % weight of sparsity penalty term       

%%======================================================================
%% STEP 1: Load data from the MNIST database
%
%  This loads our training data from the MNIST database files.

% Load data
rawVectorFolder='/big/du/Protein_Protein_Interaction_Project/Contact_Matrix_Project/Vectors_Fishers_aaIndex_raw/';

trainFile='finishedDDIs_AllVectorsChooseNegRand_FisherM1_SVMLIGHT_POLY_02NOV2011.txt_F0_20_F1_20_Sliding_17_11_F0_20_F1_20_Sliding_17_11_ouput_1_training.txt';
trainFilePath=[rawVectorFolder ddiName '/' trainFile];
% get the whitened trainging data and mu (mean), whMat(whitening matrix) used for precess testData, true for random select negative data.
[trainData , trainLabels, mu, whMat]=processRawData(trainFilePath, trainRand, choseNegRatio);
trainData=trainData';% need to change to columns are examles.
trainData=trainData ./2+0.5;
trainLabels(trainLabels == 0) = numClasses ; % Remap 0 to 2 since our labels need to start from 1

%%======================================================================
%% STEP 2: Train the first sparse autoencoder
%  This trains the first sparse autoencoder on the unlabelled STL training
%  images.
%  If you've correctly implemented sparseAutoencoderCost.m, you don't need
%  to change anything here.


%  Randomly initialize the parameters
sae1Theta = initializeParameters(hiddenSizeL1, inputSize);

%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the first layer sparse autoencoder, this layer has
%                an hidden size of "hiddenSizeL1"
%                You should store the optimal parameters in sae1OptTheta


addpath minFunc/;
options = struct;
options.Method = 'lbfgs';
options.maxIter = maxIteration; %400
options.display = 'on';
[sae1OptTheta, cost] =  minFunc(@(p)sparseAutoencoderCost(p,...
    inputSize,hiddenSizeL1,lambda,sparsityParam,beta,trainData),sae1Theta,options);%训练出第一层网络的参数
%save('saves/step2.mat', 'sae1OptTheta');

if DISPLAY
  W1 = reshape(sae1OptTheta(1:hiddenSizeL1 * inputSize), hiddenSizeL1, inputSize);
  display_network(W1');
end



% -------------------------------------------------------------------------



%%======================================================================
%% STEP 2: Train the second sparse autoencoder
%  This trains the second sparse autoencoder on the first autoencoder
%  featurse.
%  If you've correctly implemented sparseAutoencoderCost.m, you don't need
%  to change anything here.

[sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                        inputSize, trainData);

%  Randomly initialize the parameters
sae2Theta = initializeParameters(hiddenSizeL2, hiddenSizeL1);

%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the second layer sparse autoencoder, this layer has
%                an hidden size of "hiddenSizeL2" and an inputsize of
%                "hiddenSizeL1"
%
%                You should store the optimal parameters in sae2OptTheta

[sae2OptTheta, cost] =  minFunc(@(p)sparseAutoencoderCost(p,...
    hiddenSizeL1,hiddenSizeL2,lambda,sparsityParam,beta,sae1Features),sae2Theta,options);%训练出第二层网络的参数
%save('saves/step3.mat', 'sae2OptTheta');

figure;
if DISPLAY
  W11 = reshape(sae1OptTheta(1:hiddenSizeL1 * inputSize), hiddenSizeL1, inputSize);
  W12 = reshape(sae2OptTheta(1:hiddenSizeL2 * hiddenSizeL1), hiddenSizeL2, hiddenSizeL1);
  % TODO(zellyn): figure out how to display a 2-level network
%  display_network(log(W11' ./ (1-W11')) * W12');
    W2=W12*W11;
   display_network(W2');
%   figure;
%   display_network(W12_temp');
end


% -------------------------------------------------------------------------


%%======================================================================
%% STEP 3: Train the softmax classifier
%  This trains the sparse autoencoder on the second autoencoder features.
%  If you've correctly implemented softmaxCost.m, you don't need
%  to change anything here.


[sae2Features] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
                                        hiddenSizeL1, sae1Features);

%  Randomly initialize the parameters
saeSoftmaxTheta = 0.005 * randn(hiddenSizeL2 * numClasses, 1);


%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the softmax classifier, the classifier takes in
%                input of dimension "hiddenSizeL2" corresponding to the
%                hidden layer size of the 2nd layer.
%
%                You should store the optimal parameters in saeSoftmaxOptTheta 
%
%  NOTE: If you used softmaxTrain to complete this part of the exercise,
%        set saeSoftmaxOptTheta = softmaxModel.optTheta(:);

softmaxLambda = 1e-4;
numClasses = 2;
softoptions = struct;
softoptions.maxIter = 400;
softmaxModel = softmaxTrain(hiddenSizeL2,numClasses,softmaxLambda,...
                            sae2Features,trainLabels,softoptions);
saeSoftmaxOptTheta = softmaxModel.optTheta(:);

%save('saves/step4.mat', 'saeSoftmaxOptTheta');


% -------------------------------------------------------------------------



%%======================================================================
%% STEP 5: Finetune softmax model

% Implement the stackedAECost to give the combined cost of the whole model
% then run this cell.

% Initialize the stack using the parameters learned

stack = cell(2,1);
stack{1}.w = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), ...
                     hiddenSizeL1, inputSize);
stack{1}.b = sae1OptTheta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize+hiddenSizeL1);
stack{2}.w = reshape(sae2OptTheta(1:hiddenSizeL2*hiddenSizeL1), ...
                     hiddenSizeL2, hiddenSizeL1);
stack{2}.b = sae2OptTheta(2*hiddenSizeL2*hiddenSizeL1+1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2);

% Initialize the parameters for the deep model
[stackparams, netconfig] = stack2params(stack);
stackedAETheta = [ saeSoftmaxOptTheta ; stackparams ];%stackedAETheta是个向量，为整个网络的参数，包括分类器那部分，且分类器那部分的参数放前面

%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the deep network, hidden size here refers to the '
%                dimension of the input to the classifier, which corresponds 
%                to "hiddenSizeL2".
%
%

[stackedAEOptTheta, cost] =  minFunc(@(p)stackedAECost(p,inputSize,hiddenSizeL2,...
                         numClasses, netconfig,lambda, trainData, trainLabels),...
                        stackedAETheta,options);%训练出第一层网络的参数
%save('saves/step5.mat', 'stackedAEOptTheta');

figure;
if DISPLAY
  optStack = params2stack(stackedAEOptTheta(hiddenSizeL2*numClasses+1:end), netconfig);
  W11 = optStack{1}.w;After
  W12 = optStack{2}.w;
  % TODO(zellyn): figure out how to display a 2-level network
      W3=W12*W11;
   display_network(W3');
  % display_network(log(1 ./ (1-W11')) * W12');
end



% -------------------------------------------------------------------------



%%======================================================================
%% STEP 6: Test 
%  Instructions: You will need to complete the code in stackedAEPredict.m
%                before running this part of the code
%

% Get labelled test images
% Note that we apply the same kind of preprocessing as the training set

testFile='finishedDDIs_AllVectorsChooseNegRand_FisherM1_SVMLIGHT_POLY_02NOV2011.txt_F0_20_F1_20_Sliding_17_11_F0_20_F1_20_Sliding_17_11_ouput_1_training.txt';
testFilePath=[rawVectorFolder ddiName '/' testFile];
[testData , testLabels]=processTestData(testFilePath, mu, whMat, testRand, choseNegRatio);
testData=testData';
testData=testData ./2+0.5;
%testLabels(testLabels == 0) = numClasses; % Remap 0 to 10

[pred] = stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData);
pred(pred == 2) = 0;
[accuracy, sensitivity, specificity, precision, recall, f_measure, gmean]=Evaluate(testLabels',pred); %Evaluate(ACTUAL,PREDICTED)
acc = mean(testLabels(:) == pred(:));
fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', acc * 100);
fprintf('Before Finetuning Test accuracy: %0.3f%%\n', accuracy * 100);
fprintf('Before Finetuning Test recall: %0.3f%%\n', recall * 100);
fprintf('Before Finetuning Test precision: %0.3f%%\n', precision * 100);
fprintf('Before Finetuning Test sensitivity: %0.3f%%\n', sensitivity * 100);
fprintf('Before Finetuning Test f_measure: %0.3f%%\n', f_measure * 100);

[pred] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData);
pred(pred == 2) = 0;
acc = mean(testLabels(:) == pred(:));
[accuracy, sensitivity, specificity, precision, recall, f_measure, gmean]=Evaluate(testLabels',pred); %Evaluate(ACTUAL,PREDICTED)

fprintf('After Finetuning Test Accuracy: %0.3f%%\n', acc * 100);

fprintf('After Finetuning Test accuracy: %0.3f%%\n', accuracy * 100);
fprintf('After Finetuning Test recall: %0.3f%%\n', recall * 100);
fprintf('After Finetuning Test precision: %0.3f%%\n', precision * 100);
fprintf('After Finetuning Test sensitivity: %0.3f%%\n', sensitivity * 100);
fprintf('After Finetuning Test f_measure: %0.3f%%\n', f_measure * 100);
fprintf('DDI: %s L1: %d L2: %d TrainOnChosenRandom %d TestOnChosenRandom %d\n', ddiName, hiddenSizeL1, hiddenSizeL2, trainRand, testRand);
end
