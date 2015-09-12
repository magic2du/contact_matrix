function [trainResult, testResult]= SAE_AllVectorsChooseNegRand_aaIndex_test_load_DL_cross(trainData, trainLabel, testData, testLabel)
addpath(genpath('/home/michael/Dropbox/ML/DNN/DeepLearnToolbox/'));

diary off;
% Process data with whitening and scale to 0 1.
%[trainDatawt, mu, invMat, whMat] = whiten(trainData, 0.0001);

%trainDatawt= trainDatawt ./2 +0.5;
%testDatawt= bsxfun(@minus, testData, mu);
%testDatawt=testDatawt*whMat;
%testDatawt= testDatawt ./2 +0.5;
%just scale feature to 0-1
[trainData_scale, minval, range] = scale_0_1(trainData);
testData_scale= (testData- repmat(minval, size(testData, 1) ,1)) ./repmat(range, size(testData, 1) ,1);

train_x = trainData_scale;
test_x  = testData_scale;
train_y = trainLabel;
test_y  = testLabel;
vx=test_x;  %validation set
vy=test_y;
%setup parameters
numberOFFeature=size(train_x, 2);
hiddenUnits=60;
batchsize=100;
numepochs=2000;
inputZeroMaskedFraction=0; %default 0.5
weightPenaltyL2=1.00e-6; %default 1.00e-3
learningRate= 0.01; %default is 1
dropoutFraction=0; %default is 0

plot              = 1;            %  enable plotting
%%  ex1 train a 100 hidden unit SDAE and use it to initialize a FFNN
%  Setup and train a stacked denoising autoencoder (SDAE)
if 0
    rng(0);
    %sae = saesetup([784 100]);
    sae = saesetup([numberOFFeature hiddenUnits]);
    sae.ae{1}.activation_function       = 'sigm';
    sae.ae{1}.learningRate              = learningRate;
    sae.ae{1}.weightPenaltyL2           = weightPenaltyL2;
    sae.ae{1}.nonSparsityPenalty        = 3;
    sae.ae{1}.sparsityTarget=0.1;
    sae.ae{1}.inputZeroMaskedFraction   =inputZeroMaskedFraction;
    opts.numepochs =   numepochs;
    opts.batchsize = batchsize;
    sae = saetrain(sae, train_x, opts);
    % visualize(sae.ae{1}.W{1}(:,2:end)')

    % Use the SDAE to initialize a FFNN
    nn = nnsetup([numberOFFeature hiddenUnits 1]);
    nn.activation_function              = 'sigm';
    nn.learningRate                     = learningRate;
    nn.weightPenaltyL2                  = weightPenaltyL2;
    nn.W{1} = sae.ae{1}.W{1};

    % Train the FFNN
    opts.numepochs =   numepochs;
    opts.batchsize = batchsize;
    nn = nntrain(nn, train_x, train_y, opts);
    testResult = nnpredict(nn, test_x);
    trainResult = nnpredict(nn, train_x);

    trainResult0_1=zeros(length(trainResult), 1);
    trainResult0_1(find(trainResult>=0))=1;
    [accuracy, sensitivity, specificity, precision, recall, f_measure, gmean]=Evaluate(trainLabel, trainResult0_1); %Evaluate(ACTUAL,PREDICTED)

    if isnan(precision)
        precision = 0;
    end
    if isnan(f_measure)
        f_measure = 0;
    end
    diary on;
    fprintf('For one hidden layer\n')
    fprintf('Training accuracy: %0.3f%%\n', accuracy * 100);
    fprintf('Training recall: %0.3f%%\n', recall * 100);
    fprintf('Training precision: %0.3f%%\n', precision * 100);
    fprintf('Training f_measure: %0.3f%%\n', f_measure * 100);
    % for testing set performance.
    testResult0_1=zeros(length(testResult),1);
    testResult0_1(find(testResult>=0))=1;

    [accuracy, sensitivity, specificity, precision, recall, f_measure, gmean]=Evaluate(testLabel, testResult0_1); %Evaluate(ACTUAL,PREDICTED)
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
end
diary off;
%% ex2 train a 100-100 hidden unit SDAE and use it to initialize a FFNN
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

% Train the FFNN
opts.numepochs =   numepochs;
opts.batchsize = batchsize;
opts.plot              = plot;
nn = nntrain(nn, train_x, train_y, opts, vx, vy);
testResult = nnpredict(nn, test_x);
trainResult = nnpredict(nn, train_x);
diary on;
return
