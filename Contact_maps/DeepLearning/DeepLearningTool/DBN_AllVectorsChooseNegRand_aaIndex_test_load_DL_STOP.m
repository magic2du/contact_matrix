function [trainResult, testResult]= DBN_AllVectorsChooseNegRand_aaIndex_test_load_DL_STOP(trainData, trainLabel, testData, testLabel)
addpath(genpath('/home/michael/Dropbox/ML/DNN/DeepLearnToolbox/'));

diary off;
% Process data with whitening and scale to 0 1.
%[trainDatawt, mu, invMat, whMat] = whiten(trainData, 0.0001);

%trainDatawt= trainDatawt ./2 +0.5;
%testDatawt= bsxfun(@minus, testData, mu);
%testDatawt=testDatawt*whMat;
%testDatawt= testDatawt ./2 +0.5;
%just scale feature to 0-1
[trainData_scale, testData_scale]=preProcessData(trainData, testData); %scale the data to 0 1
[newTrainSet, newTrainLable, validationSet, valicationlable]= splitTrainData2TrainAndValidation(trainData_scale, trainLabel);
train_x = newTrainSet;
test_x  = testData_scale;
train_y = newTrainLable;
test_y  = testLabel;
vx      = validationSet;
vy      = valicationlable;

%setup parameters
numberOFFeature=size(train_x, 2);
hiddenUnits=20;
[batchsize, numepochs]=setBatchsizeAndEpochs(size(train_x, 1), 32, 100000); %32 is default minibatchsize 
inputZeroMaskedFraction=0.2; %default 0.2
weightPenaltyL2=1.00e-3; %default 1.00e-3
learningRate= 0.03; %default is 1
dropoutFraction=0; %default is 0
plot              = 0;            %  enable plotting
if 0
	%%  ex1 train a 100 hidden unit RBM and visualize its weights
	rng(0);
	dbn.sizes = [hiddenUnits];
	opts.numepochs =   numepochs;
	opts.batchsize = batchsize;
	opts.momentum  =   0;
	opts.alpha     =   1;
	dbn = dbnsetup(dbn, train_x, opts);
	dbn = dbntrain(dbn, train_x, opts);
	figure; visualize(dbn.rbm{1}.W');   %  Visualize the RBM weights
end
diary off;
%%  ex2 train a 100-100 hidden unit DBN and use its weights to initialize a NN
rng(0);
%train dbn
dbn.sizes = [hiddenUnits hiddenUnits];
opts.numepochs =   numepochs;
opts.batchsize = batchsize;
opts.momentum  =   0;
opts.alpha     =   1;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);

%unfold dbn to nn
nn = dbnunfoldtonn(dbn, 1);
nn.activation_function              = 'sigm';
nn.learningRate                     = learningRate;
nn.weightPenaltyL2                  = weightPenaltyL2;
nn.dropoutFraction                  = dropoutFraction;

%train nn
opts.numepochs =   numepochs;
opts.batchsize = batchsize;
opts.plot              = plot;
nn = nntrain(nn, train_x, train_y, opts, vx, vy);
testResult = nnpredict(nn, test_x);
trainResult = nnpredict(nn, trainData_scale);
diary on;
return
