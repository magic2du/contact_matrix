function [trainResult, testResultOnReducedModel, testResultOnUSModel] = SAE_AllVectorsChooseNegRand_aaIndex_test_DL_RE_US_remote(trainData, trainLabel, testData, testLabel, reducedTrainingData, reducedTrainingLabel)
addpath(genpath('/home/michael/Dropbox/ML/DNN/DeepLearnToolbox/'));

diary off;

%just scale feature to 0-1
[whole_scale, minval, range] = scale_0_1([trainData; testData]);
trainData_scale = (trainData- repmat(minval, size(trainData, 1) ,1)) ./repmat(range, size(trainData, 1) ,1);
testData_scale = (testData- repmat(minval, size(testData, 1) ,1)) ./repmat(range, size(testData, 1) ,1);
reducedTrain_scale= (reducedTrainingData- repmat(minval, size(reducedTrainingData, 1) ,1)) ./repmat(range, size(reducedTrainingData, 1) ,1);

preTrain_x = trainData_scale;

scaled_reduced_x=reducedTrain_scale;
reduced_y=reducedTrainingLabel;

test_x  = testData_scale;
test_y  = testLabel;
train_y = trainLabel;

vx=test_x;
vy=test_y;
%setup parameters
numberOFFeature=size(test_x, 2);
hiddenUnits=20;
batchsize=size(preTrain_x, 1);
numepochs=50000;
inputZeroMaskedFraction=0.2; %default 0.2
weightPenaltyL2=1.00e-4; %default 1.00e-3
learningRate= 0.01; %default is 1
dropoutFraction=0; %default is 0
plot              = 0;            %  enable plotting

diary off;
%% First Train using just reduced example.%%%%%%%%%%%
%  Setup and train a stacked denoising autoencoder (SDAE)
rng(0);
sae = saesetup([numberOFFeature hiddenUnits hiddenUnits]);
sae.ae{1}.activation_function       = 'sigm';
sae.ae{1}.learningRate              = learningRate;
sae.ae{1}.weightPenaltyL2           = weightPenaltyL2;
sae.ae{1}.nonSparsityPenalty        = 3;
sae.ae{1}.sparsityTarget=0.1;
sae.ae{1}.inputZeroMaskedFraction   = inputZeroMaskedFraction;

sae.ae{2}.activation_function       = 'sigm';
sae.ae{2}.learningRate              = learningRate;
sae.ae{2}.weightPenaltyL2           = weightPenaltyL2;
sae.ae{2}.nonSparsityPenalty        = 3;
sae.ae{2}.sparsityTarget=0.1;
sae.ae{2}.inputZeroMaskedFraction   = inputZeroMaskedFraction;

opts.numepochs =   numepochs;
opts.batchsize = batchsize;
sae = saetrain(sae, reducedTrain_scale, opts);
% visualize(sae.ae{1}.W{1}(:,2:end)')

% Use the SDAE to initialize a FFNN
nn = nnsetup([numberOFFeature hiddenUnits hiddenUnits 1]);
nn.activation_function              = 'sigm';
nn.learningRate                     = learningRate;
nn.weightPenaltyL2                  = weightPenaltyL2;
nn.dropoutFraction                  = dropoutFraction;
%add pretrained weights
nn.W{1} = sae.ae{1}.W{1};
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   numepochs;
opts.batchsize = size(reducedTrain_scale, 1);
opts.plot              = plot;
%%%%%%%%%without early stopping
nn = nntrain(nn, reducedTrain_scale, reduced_y, opts);
testResultOnReducedModel = nnpredict(nn, test_x);

%% 2 Second Train using just US for AE and reduced for training%%%%%%%%%%%
%  Setup and train a stacked denoising autoencoder (SDAE)
rng(0);
sae = saesetup([numberOFFeature hiddenUnits hiddenUnits]);
sae.ae{1}.activation_function       = 'sigm';
sae.ae{1}.learningRate              = learningRate;
sae.ae{1}.weightPenaltyL2           = weightPenaltyL2;
sae.ae{1}.nonSparsityPenalty        = 3;
sae.ae{1}.sparsityTarget=0.1;
sae.ae{1}.inputZeroMaskedFraction   = inputZeroMaskedFraction;

sae.ae{2}.activation_function       = 'sigm';
sae.ae{2}.learningRate              = learningRate;
sae.ae{2}.weightPenaltyL2           = weightPenaltyL2;
sae.ae{2}.nonSparsityPenalty        = 3;
sae.ae{2}.sparsityTarget=0.1;
sae.ae{2}.inputZeroMaskedFraction   = inputZeroMaskedFraction;

opts.numepochs =   numepochs;
opts.batchsize = batchsize;
sae = saetrain(sae, trainData_scale, opts);
% visualize(sae.ae{1}.W{1}(:,2:end)')

% Use the SDAE to initialize a FFNN
nn = nnsetup([numberOFFeature hiddenUnits hiddenUnits 1]);
nn.activation_function              = 'sigm';
nn.learningRate                     = learningRate;
nn.weightPenaltyL2                  = weightPenaltyL2;
nn.dropoutFraction                  = dropoutFraction;
%add pretrained weights
nn.W{1} = sae.ae{1}.W{1};
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   numepochs;
opts.batchsize = size(reducedTrain_scale, 1);
opts.plot              = plot;
%%%%%%%%%without early stopping
nn = nntrain(nn, reducedTrain_scale, reduced_y, opts);
testResultOnUSModel = nnpredict(nn, test_x);
trainResult = nnpredict(nn, trainData_scale);

diary on;
return
