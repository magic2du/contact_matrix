function [trainResult, testResult]= SAE_AllVectorsChooseNegRand_aaIndex_test_load_SAESVM(trainData, trainLabel, testData, testLabel, Kernel, ddiPath)
addpath(genpath('/home/michael/Dropbox/ML/DNN/DeepLearnToolbox/'));

diary off;
[trainData_scale, testData_scale]=preProcessData(trainData, testData);

train_x = trainData_scale;
test_x  = testData_scale;
train_y = trainLabel;
test_y  = testLabel;
%setup parameters
numberOFFeature=size(train_x, 2);
hiddenUnits=10;
batchsize=size(train_x, 1);
numepochs=50000;
inputZeroMaskedFraction=0.2; %default 0.2
weightPenaltyL2=1.00e-3; %default 1.00e-3
learningRate= 0.01; %default is 1
nonSparsityPenalty=0; %default is 3
dropoutFraction=0;
plot              = 0;            %  enable plotting
%%  ex1 train a 100 hidden unit SDAE and use it to initialize a FFNN
%  Setup and train a stacked denoising autoencoder (SDAE)
if strcmp(Kernel, 'L1')
    rng(0);
    %sae = saesetup([784 100]);
    sae = saesetup([numberOFFeature hiddenUnits]);
    sae.ae{1}.activation_function       = 'sigm';
    sae.ae{1}.learningRate              = learningRate;
    sae.ae{1}.weightPenaltyL2           = weightPenaltyL2;
    sae.ae{1}.nonSparsityPenalty        = nonSparsityPenalty;
    sae.ae{1}.sparsityTarget=0.1;
    sae.ae{1}.inputZeroMaskedFraction   =inputZeroMaskedFraction;
    opts.numepochs =   numepochs;
    opts.batchsize = batchsize;
    opts.plot=plot;
    sae = saetrain(sae, train_x, opts);
    % visualize(sae.ae{1}.W{1}(:,2:end)')

    % Use the SDAE to initialize a FFNN
    nn = nnsetup([numberOFFeature hiddenUnits 1]);
    nn.activation_function              = 'sigm';
    nn.learningRate                     = learningRate;
    nn.weightPenaltyL2                  = weightPenaltyL2;
    nn.W{1} = sae.ae{1}.W{1};

    % Use AE to get the new representation of X. get Activations Of Last Hidden Layer
    newTrainX=getNewRepresentationOfX(nn, train_x, train_y);
    newTestX=getNewRepresentationOfX(nn, test_x, testLabel);
    [trainResult, testResult]=runSVMLight(newTrainX,train_y, newTestX, testLabel, ddiPath); 
elseif   strcmp(Kernel, 'L2')   
    rng(0);
    sae = saesetup([numberOFFeature hiddenUnits hiddenUnits]);
    sae.ae{1}.activation_function       = 'sigm';
    sae.ae{1}.learningRate              = learningRate;
    sae.ae{1}.weightPenaltyL2           = weightPenaltyL2;
    sae.ae{1}.nonSparsityPenalty        = nonSparsityPenalty;
    sae.ae{1}.sparsityTarget=0.1;
    sae.ae{1}.inputZeroMaskedFraction   = inputZeroMaskedFraction;

    sae.ae{2}.activation_function       = 'sigm';
    sae.ae{2}.learningRate              = learningRate;
    sae.ae{2}.weightPenaltyL2           = weightPenaltyL2;
    sae.ae{2}.nonSparsityPenalty        = nonSparsityPenalty;
    sae.ae{2}.sparsityTarget=0.1;
    sae.ae{2}.inputZeroMaskedFraction   = inputZeroMaskedFraction;

    opts.numepochs =   numepochs;
    opts.batchsize = batchsize;
    opts.plot=0;
    sae = saetrain(sae, train_x, opts);
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

    % Use AE to get the new representation of X. get Activations Of Last Hidden Layer
    newTrainX=getNewRepresentationOfX(nn, train_x, train_y);
    newTestX=getNewRepresentationOfX(nn, test_x, testLabel);
    [trainResult, testResult]=runSVMLight(newTrainX,train_y, newTestX, testLabel, ddiPath); 
end
diary on;
%% ex2 train a 100-100 hidden unit SDAE and use it to initialize a FFNN
%  Setup and train a stacked denoising autoencoder (SDAE)
return
