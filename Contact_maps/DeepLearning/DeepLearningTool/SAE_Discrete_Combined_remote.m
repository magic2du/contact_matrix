function [trainResult, testResultOnReducedModel, testResultOnUSModel] = SAE_Discrete_Combined_remote.m(train_X1, train_X2, train_y, validationX1, validationX2, validation_y, test_X1, test_X2, test_y)
% train_X1 family A features
% train_X2 family B features
% train_y  train data labels
% pretrain on train_X1, train_X2 seperately. Than combine them together. Using new representation of train_X1, train_X2.
% join
addpath(genpath('/home/michael/Dropbox/ML/DNN/DeepLearnToolbox/'));

diary off;

%setup parameters
numberOFFeature_X1=size(train_X1, 2);
numberOFFeature_X2=size(train_X2, 2);
hiddenUnits_X1=20;% family A pretraining
hiddenUnits_X2=20;
[batchsize, numepochs]=setBatchsizeAndEpochs(size(train_X1, 1), 32, 100000); %32 is default minibatchsize 
inputZeroMaskedFraction=0.5; %default 0.2
weightPenaltyL2=1.00e-4; %default 1.00e-3
learningRate= 0.01; %default is 1
dropoutFraction=0; %default is 0
plot              = 0;            %  enable plotting

diary off;
%% Pretraninig for train_X1.%%%%%%%%%%%
%  Setup and train a stacked denoising autoencoder (SDAE)
rng(0);
sae = saesetup([numberOFFeature hiddenUnits hiddenUnits]);
sae.ae{1}.activation_function       = 'sigm';
sae.ae{1}.learningRate              = learningRate;
sae.ae{1}.weightPenaltyL2           = weightPenaltyL2;
sae.ae{1}.nonSparsityPenalty        = 3; %Default=3
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
sae = saetrain(sae, train_X1, opts);
% visualize(sae.ae{1}.W{1}(:,2:end)')

% Use the SDAE to initialize a FFNN
nn=[];
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
opts.batchsize = batchsize;
opts.plot              = plot;
%%%%%%%%%with early stopping
nn = nntrain(nn, train_x, train_y, opts,vx, vy);
testResultOnReducedModel = nnpredict(nn, test_x);
nn=[];
sae=[];
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
opts.batchsize = batchsize;
opts.plot              = plot;
%%%%%%%%%with early stopping
nn = nntrain(nn, train_x, train_y, opts,vx, vy);
testResultOnUSModel = nnpredict(nn, test_x);
trainResult = nnpredict(nn, trainData_scale);

diary on;
return
