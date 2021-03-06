function [aucBL_Array, blSens_Array, blSpec_Array, ...
                    auc_Array, svmSens_Array, svmSpec_Array, test_f_measure_Array,...
                    trainAccuracy_Array, trainRecall_Array, trainPrecision_Array, train_f_measure_Array]  = runDDI_CrossValidation_load_DL_remote...
                                (ddiName, FisherMode, SVMMode, Kernel, choseNegRatio, choseNegRatioTest)

folder3did = ['/big/du/Protein_Protein_Interaction_Project/Contact_Matrix_Project/Vectors_Fishers_aaIndex_raw/'];
dataFolder =                                        
ddiPath = [folder3did ddiName '/'];
dataFolder =  ddiPath
folderResults = ddiPath;
if ~exist(folderResults, 'dir')
	mkdir(folderResults);
end

% load sequence numbers  to run.
filePath=[ddiPath 'pairsToRun.txt'];
pairsToRun= load(filePath);


aucBL_Array = [];
blSens_Array = [];
blSpec_Array = [];
auc_Array = [];
svmSens_Array = [];
svmSpec_Array = [];
test_f_measure_Array = [];
trainAccuracy_Array = [];
trainRecall_Array = [];
trainPrecision_Array = [];
train_f_measure_Array = [];
startTime = clock;
% make k fold Partation.
k=10;
partation=cvpartition(length(pairsToRun),'kfold',k)
%index=find(partation.test(1)==1), index=find(partation.training(k)==1) 
for partationCtr = 1:k
    ['fold ' partationCtr ' :\n']
    testPairs = find(partation.test(partationCtr)==1);
    trainingPairs = find(partation.training(k)==1) ;
    
	% Training Fisher vectors. For all the training sequence pairs, go to the
	% corresponding contact matrix, and get from there positive and negative
	% Fisher vector pairs.
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	trainPO = [];
	trainNE = [];
	for idxCtr = 1:length(trainingPairs)
	    seqCtr = trainingPairs(idxCtr);
	    
	    % create positive and negative training sets.
	    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\
	    dataFile=[dataFolder 'F0_20_F1_20_Sliding_17_11_F0_20_F1_20_Sliding_17_11_ouput_' num2str(seqCtr) '.txt'];
	    numPos=load([dataFolder 'numPos_' num2str(seqCtr) '.txt']);
	    numNeg=load([dataFolder 'numNeg_' num2str(seqCtr) '.txt']);
	    [selectedData, label]=chooseAAIndexVectores(dataFile, FisherMode);
	    PO=selectedData(1:numPos, :);
	    NE=selectedData(numPos+1: numPos+numNeg, :);

	    % chose negatives as in the the ratio default is 1.
	    selectedNE=[];
	    if choseNegRatio==0
		selectedNE=NE;
	    else
		numbNegTrain = size(PO, 1) * choseNegRatio;
		if numbNegTrain>numNeg
		    numbNegTrain=numNeg;
		end
		r=randperm(size(NE, 1));
		r=r(1:numbNegTrain);
		indTrainNE=sort(r);
		selectedNE = NE(indTrainNE, :);
	    end
	    trainNE=[trainNE; selectedNE];
	    trainPO=[trainPO; PO];
	end
	    trainGroundTruth = [ones(size(trainPO, 1), 1); ...
		            -1*ones(size(trainNE, 1), 1)];
	% Build Traing Model.
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     	% SVMLIGHT.
	if strcmp(SVMMode, 'SVMLIGHT')

	    svmlightFolder ='/home/du/Protein_Protein_Interaction_Project/svm_light_linux64_2013/';
	  %  '/home/du/Protein_Protein_Interaction_Project/3did_20NOV2009/svm_light/';
	    folderResults = ...
		['/home/du/Protein_Protein_Interaction_Project/Contact_Matrix_Project/dom_dom_ints/' ddiName '/'];
	    trainFile = ...
		[folderResults FisherMode '_'  SVMMode '_' Kernel '_' num2str(choseNegRatio) '_' num2str(choseNegRatioTest) '_CrossValidation_load_DL_remote.train'];
	    modelFile = ...
		[folderResults FisherMode '_'  SVMMode '_' Kernel '_' num2str(choseNegRatio) '_' num2str(choseNegRatioTest) '_CrossValidation_load_DL_remote.model'];
	    % write_SVM_file(FamilyPO, FamilyNE, trainFile);
	    write_SVM_file(trainPO, trainNE, trainFile);
	    if strcmp(Kernel, 'RBF')
		command = ...
		    [svmlightFolder 'svm_learn -t 2 -g 1 -c ' num2str(C) ' -j ' num2str(Jj) ' '  trainFile ' ' modelFile];
	    elseif strcmp(Kernel, 'POLY')
		command = ...
		    [svmlightFolder 'svm_learn -t 1 -d 3 -c ' num2str(C) ' -j ' num2str(Jj) ' '  trainFile ' ' modelFile];
	    elseif strcmp(Kernel, 'LINEAR')
		command = ...
		    [svmlightFolder 'svm_learn -t 0 ' trainFile ' ' modelFile];
	    else
		error('ERROR');
	    end
	    diary off;
	    system(command);
	    diary on;

	    % test on training set.
	    diary off;

	    resultFile = [folderResults FisherMode '_'  SVMMode '_' Kernel '_' num2str(pairNbr) num2str(choseNegRatio) '_' num2str(choseNegRatioTest) '_CrossValidation_load_DL_remote.resultOnTrain'];
	    %write_SVM_file(trainPO, trainNE, testFile);
	    command = ...
	    [svmlightFolder 'svm_classify ' trainFile ' ' modelFile ' ' resultFile];
	    system(command);
	    trainResult = load(resultFile);
		length(trainResult)
	    trainAUC = roc(trainResult, trainGroundTruth, 'nofigure');
	    % for training performance.
	    trainResult0_1=zeros(length(trainResult), 1);
	    trainResult0_1(find(trainResult>=0))=1;
	    trainGroundTruth(find(trainGroundTruth==-1))=0; % change the format to 0 and 1;
	    [trainAccuracy, sensitivity, specificity, trainPrecision, trainRecall, train_f_measure, gmean]=Evaluate(trainGroundTruth, trainResult0_1); %Evaluate(ACTUAL,PREDICTED)
	     
	    if isnan(trainPrecision)
		precision = 0;
	    end
	    if isnan(train_f_measure)
		train_f_measure = 0;
	    end
	    diary on;
	    fprintf('Training AUC: %0.3f%%\n', trainAUC * 100);
	    fprintf('Training accuracy: %0.3f%%\n', trainAccuracy * 100);
	    fprintf('Training recall: %0.3f%%\n', trainRecall * 100);
	    fprintf('Training precision: %0.3f%%\n', trainPrecision * 100);
	    fprintf('Training f_measure: %0.3f%%\n', train_f_measure * 100);


	elseif strcmp(SVMMode, 'DL_RE_US')
	    %%%%%%%%%%%%%%%%%%%%%%This is to reduce the traing sequence pairs for tranning
	    trainData=[trainPO; trainNE];
	    trainLabel=[ones(size(trainPO, 1), 1); ...
					    zeros(size(trainNE, 1), 1)];

	    [reducedTrainingData, reducedTrainingLabel]=getReducedTrainingAndLabel(8, trainingPairs, dataFolder, FisherMode, choseNegRatio);
	%%%% transform data so that  0 1 whitening.
	%just scale feature to 0-1
	[whole_scale, minval, range] = scale_0_1(trainData);
	trainData_scale = (trainData- repmat(minval, size(trainData, 1) ,1)) ./repmat(range, size(trainData, 1) ,1);
	%testData_scale = (testData- repmat(minval, size(testData, 1) ,1)) ./repmat(range, size(testData, 1) ,1);
	reducedTrain_scale= (reducedTrainingData- repmat(minval, size(reducedTrainingData, 1) ,1)) ./repmat(range, size(reducedTrainingData, 1) ,1);
	% split Reduced data to validation set.
	[newTrainSet, newTrainLable, validationSet, valicationlable]= splitTrainData2TrainAndValidation(reducedTrain_scale, reducedTrainingLabel);


	% built reduced SAE model
	reduced_NN = SAE_built_nn_with_DLSTOP(newTrainSet, newTrainLable, validationSet, valicationlable);
	RE_US_NN = SAE_built_nn_with_RE_US(trainData_scale, newTrainSet, newTrainLable, validationSet, valicationlable);
	%%%test on training data %%%%%%
	%%%for unsupervised traing %%%%%
	trainResult = nnpredict(RE_US_NN , newTrainSet);
	    % for training performance.
	    trainResult0_1=zeros(length(trainResult),1);
	    trainResult0_1(find(trainResult>=0))=1;
	    [trainAccuracy, sensitivity, specificity, trainPrecision, trainRecall, train_f_measure, gmean]=Evaluate(newTrainLable, trainResult0_1); %Evaluate(ACTUAL,PREDICTED)
    	    if isnan(trainPrecision)
		trainPrecision = 0;
	    end
	    if isnan(train_f_measure)
		train_f_measure = 0;
	    end
	    diary on;
	    fprintf('Training performance on unsupervised reduced model:\n');
	    fprintf('Training accuracy: %0.3f%%\n', trainAccuracy * 100);
	    fprintf('Training recall: %0.3f%%\n', trainRecall * 100);
	    fprintf('Training precision: %0.3f%%\n', trainPrecision * 100);
	    fprintf('Training f_measure: %0.3f%%\n', train_f_measure * 100);
	end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%testing%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % test on test set:

    for idxCtr = 1:length(testPairs)
	    seqCtr = testPairs(idxCtr);
	    pairNbr = seqCtr;
	POTest=[];
	NETest=[];
	fprintf('Testing for sequence pair %s:\n', pairNbr);
	testFile=[dataFolder 'F0_20_F1_20_Sliding_17_11_F0_20_F1_20_Sliding_17_11_ouput_' num2str(pairNbr) '.txt'];
	numPos=load([dataFolder 'numPos_' num2str(pairNbr) '.txt']);
	numNeg=load([dataFolder 'numNeg_' num2str(pairNbr) '.txt']);
	[selectedData, label]=chooseAAIndexVectores(testFile, FisherMode);
	POTest=selectedData(1:numPos, :);
	NETest=selectedData(numPos+1: numPos+numNeg, :);
    	testGroundTruth = [ones(size(POTest, 1), 1); ...
				    -1*ones(size(NETest, 1), 1)];
	testData = [POTest; NETest];
	testDataLabel =[ones(size(POTest, 1), 1); ...
				    zeros(size(NETest, 1), 1)];
	%Transform testing data to 0 - 1

	testData_scale = (testData- repmat(minval, size(testData, 1) ,1)) ./repmat(range, size(testData, 1) ,1);
	%%%%%%%%%%%%%for SVM%%%%%%%%%%%%%
	if strcmp(SVMMode, 'SVMLIGHT')
	    testFile2 =[folderResults FisherMode '_'  SVMMode '_' Kernel '_' num2str(pairNbr) '_' num2str(choseNegRatio) '_' num2str(choseNegRatioTest) '_test_load_DL_cross.test']
	    resultFile = [folderResults FisherMode '_'  SVMMode '_' Kernel '_' num2str(pairNbr) '_' num2str(choseNegRatio) '_' num2str(choseNegRatioTest) '_test_load_DL_cross.result'];
	    % for each sequence.
		
	    write_SVM_file(POTest, NETest, testFile2);
	    command = ...
	    [svmlightFolder 'svm_classify ' testFile2 ' ' modelFile ' ' resultFile];
	    system(command);
	    testResult = load(resultFile);
	    auc = roc(testResult, testGroundTruth, 'nofigure');
	    % for testing set performance.
	    testResult0_1=zeros(length(testResult),1);
	    testResult0_1(find(testResult>=0))=1;
	    testGroundTruth(find(testGroundTruth==-1))=0; % change the format to 0 and 1;
	    [accuracy, sensitivity, specificity, precision, recall, f_measure, gmean]=Evaluate(testGroundTruth, testResult0_1); %Evaluate(ACTUAL,PREDICTED)
	    if isnan(precision)
		precision = 0;
	    end
	    if isnan(f_measure)
		f_measure = 0;
	    end
	    diary on;
	    svmSens=recall;
	    svmSpec=precision;
	    test_f_measure=f_measure; 

	    aucBL_Array(pairNbr) = 0;
	    blSens_Array(pairNbr) = 0;
	    blSpec_Array(pairNbr) = 0;
	    auc_Array (pairNbr)= auc;
	    svmSens_Array(pairNbr) = svmSens;
	    svmSpec_Array(pairNbr) = svmSpec;
	    test_f_measure_Array(pairNbr) = f_measure;
	    trainAccuracy_Array(pairNbr) = trainAccuracy;
	    trainRecall_Array(pairNbr) = trainRecall;
	    trainPrecision_Array(pairNbr) = trainPrecision;
	    train_f_measure_Array(pairNbr) = train_f_measure;

	    fprintf('Testing AUC: %0.3f%%\n', auc * 100);
	    fprintf('Testing accuracy: %0.3f%%\n', accuracy * 100);
	    fprintf('Testing recall: %0.3f%%\n', recall * 100);
	    fprintf('Testing precision: %0.3f%%\n', precision * 100);
	    fprintf('Testing f_measure: %0.3f%%\n', f_measure * 100);

	elseif strcmp(SVMMode, 'DL_RE_US')

	%%%test on Testing data %%%%%%

	%%%for unsupervised training %%%%%
	    testResultOnUSModel = nnpredict(RE_US_NN , testData_scale);
	    testResult0_1=zeros(length(testData_scale),1);
	    testResult0_1(find(testResultOnUSModel>=0))=1;
	    [accuracy, sensitivity, specificity, precision, recall, f_measure, gmean]=Evaluate(testDataLabel, testResult0_1); %Evaluate(ACTUAL,PREDICTED)
	    auc = roc(testResultOnUSModel, testGroundTruth, 'nofigure');
    	    if isnan(precision)
		precision = 0;
	    end
	    if isnan(f_measure)
		f_measure = 0;
	    end
	    svmSens=recall;
	    svmSpec=precision;
	    test_f_measure=f_measure; 
	%%%%%%%%% store result %%%%%%%%
	    auc_Array (pairNbr)= auc;
	    svmSens_Array(pairNbr) = svmSens;
	    svmSpec_Array(pairNbr) = svmSpec;
	    test_f_measure_Array(pairNbr) = f_measure;

	    trainAccuracy_Array(pairNbr) = trainAccuracy;
	    trainRecall_Array(pairNbr) = trainRecall;
	    trainPrecision_Array(pairNbr) = trainPrecision;
	    train_f_measure_Array(pairNbr) = train_f_measure;
	%%%for reduced model  %%%%%
	    testResultOn_reduced_Model = nnpredict(reduced_NN , testData_scale);
	    testResult0_1=zeros(length(testData_scale),1);
	    testResult0_1(find(testResultOn_reduced_Model>=0))=1;
	    [accuracy, sensitivity, specificity, precision, recall, f_measure, gmean]=Evaluate(testDataLabel, testResult0_1); %Evaluate(ACTUAL,PREDICTED)
	    aucBL = roc(testResultOn_reduced_Model, testGroundTruth, 'nofigure');
    	    if isnan(precision)
		precision = 0;
	    end
	    if isnan(f_measure)
		f_measure = 0;
	    end
	    svmSens=recall;
	    svmSpec=precision;

	    aucBL_Array(pairNbr) = aucBL;
	    blSens_Array(pairNbr) = svmSens;
	    blSpec_Array(pairNbr) = svmSpec;


	end  
    end


end

%fprintf(['\n\n\nFinished ' ddiName ', ...
 %                       meanAUC = ' num2str(meanAUC, '%0.3f') '.\n\n\n']);

% print results to file.
%summaryFile = [folderResults 'AllVectorsChooseNegRand_02NOV2011_' ...
%                            FisherMode '_' SVMMode '_' Kernel '.summary'];
%save(summaryFile, 'AUC_Array', '-ascii');

return;
